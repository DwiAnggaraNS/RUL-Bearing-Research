"""
train_bilstm_physics.py
========================
Physics-Informed BiLSTM Training Script for Cross-Condition RUL Prediction.

Architectural features (ref: "Beyond Accuracy" and "Claude_1may" audits):
    1. Loads 3D .npy tensors natively — no CSV parsing, no reshape bugs.
    2. Asserts tensor is 3D before training to catch preprocessing errors early.
    3. PhysicsInformedBiLSTM: bidirectional LSTM, moderate capacity to prevent
       overfitting on the ~2,400-row XJTU-SY sub-minute dataset.
    4. Physics-Informed Training Loss:
           Total_Loss = MSE + lambda_mono * MonotonicityPenalty
       where MonotonicityPenalty penalises any predicted increase between
       consecutive timesteps within a batch.
    5. Post-processing hard monotonic constraint via np.minimum.accumulate
       to eliminate any residual "healing" predictions at inference time.
    6. Full diagnostic logging: collapse detection, per-fold metrics, EMA plot.

Author: DwiAnggaraNS / PHM Research Team
PEP8: compliant
Emojis: none
"""

import gc
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Publication-standard matplotlib settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   13,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.color":       "#bfbfbf",
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = (
    r"D:\Proyek Dosen\Riset Bearing\XJTU-SY_Bearing_Datasets"
    r"\Processed_Data\LSTM_Inputs_v2"
)
RESULTS_DIR = r"D:\Proyek Dosen\Riset Bearing\Training_Results_BiLSTM"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Window sizes must match preprocessing output.
WINDOW_SIZES = [1, 3, 5]
FOLDS        = 5

# Input dimensionality: 14 features x (mean + std) = 28
NUM_FEATURES = 28

# Model hyperparameters (conservative to avoid overfitting on ~2400 rows)
EPOCHS        = 80
BATCH_SIZE    = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-5
HIDDEN_DIM    = 64      # BiLSTM hidden size per direction; output = 128
NUM_LAYERS    = 2
DROPOUT       = 0.30

# Physics-Informed Loss weight
LAMBDA_MONO   = 0.10    # Monotonicity penalty coefficient

# Post-processing: apply np.minimum.accumulate on final predictions
APPLY_HARD_MONOTONE = True

# Sub-minute hop: each timestep = 0.5 physical minutes
SUBMINUTE_HOP = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("PHYSICS-INFORMED BILSTM TRAINING PIPELINE")
print("=" * 65)
print(f"  Device       : {DEVICE}")
print(f"  NUM_FEATURES : {NUM_FEATURES}")
print(f"  HIDDEN_DIM   : {HIDDEN_DIM}  (BiLSTM output: {HIDDEN_DIM * 2})")
print(f"  NUM_LAYERS   : {NUM_LAYERS}")
print(f"  DROPOUT      : {DROPOUT}")
print(f"  LAMBDA_MONO  : {LAMBDA_MONO}")
print(f"  WINDOW_SIZES : {WINDOW_SIZES}")
print("=" * 65)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PhysicsInformedBiLSTM(nn.Module):
    """
    Bidirectional LSTM regression head for Health Index prediction.

    Architecture:
        BiLSTM (input_size -> hidden_dim, bidirectional)
        -> Dropout
        -> Linear(hidden_dim * 2 -> 32)
        -> ReLU
        -> Linear(32 -> 1)
        -> Sigmoid  [output bounded to (0, 1)]

    The Sigmoid output is physically motivated: HI is normalised to [0, 1]
    by definition, and the Sigmoid prevents the need for hard output clipping
    during training while still allowing smooth gradient flow.

    Capacity is intentionally moderate (hidden_dim=64, 2 layers) to avoid
    overfitting on the relatively small sub-minute aggregated dataset
    (~2,400 rows per operational condition).
    """

    def __init__(
        self,
        input_size:  int   = NUM_FEATURES,
        hidden_dim:  int   = HIDDEN_DIM,
        num_layers:  int   = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ):
        super(PhysicsInformedBiLSTM, self).__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=True,
        )
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_dim * 2, 32)
        self.relu       = nn.ReLU()
        self.fc2        = nn.Linear(32, 1)
        self.output_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (Batch, Window_Size, NUM_FEATURES).

        Returns:
            Tensor of shape (Batch, 1) with HI predictions in (0, 1).
        """
        lstm_out, _ = self.bilstm(x)        # (Batch, Window, hidden*2)
        last_step   = lstm_out[:, -1, :]    # (Batch, hidden*2)
        out         = self.dropout(last_step)
        out         = self.relu(self.fc1(out))
        out         = self.output_act(self.fc2(out))
        return out


# ============================================================================
# PHYSICS-INFORMED LOSS
# ============================================================================

def monotonicity_penalty(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute soft monotonicity penalty for a batch of ordered predictions.

    Penalises any predicted increase from one sample to the next.
    The samples in a batch must be ordered chronologically for this penalty
    to be physically meaningful; the DataLoader must NOT shuffle predictions
    at evaluation time.

    Penalty = mean( clamp(pred[t+1] - pred[t], min=0) )

    Args:
        predictions: Tensor of shape (N,) or (N, 1) of predicted HI values.

    Returns:
        Scalar tensor representing the mean monotonicity violation.
    """
    pred_flat = predictions.view(-1)
    if pred_flat.shape[0] < 2:
        return torch.tensor(0.0, device=predictions.device)
    diff       = pred_flat[1:] - pred_flat[:-1]
    violations = torch.clamp(diff, min=0.0)
    return violations.mean()


def physics_informed_loss(
    predictions: torch.Tensor,
    targets:     torch.Tensor,
    lambda_mono: float = LAMBDA_MONO,
) -> torch.Tensor:
    """
    Compute combined MSE + monotonicity penalty loss.

    Total_Loss = MSE(predictions, targets)
               + lambda_mono * MonotonicityPenalty(predictions)

    Args:
        predictions: Tensor of shape (N, 1) or (N,).
        targets:     Tensor of shape (N, 1) or (N,).
        lambda_mono: Weight for the monotonicity penalty term.

    Returns:
        Scalar total loss tensor.
    """
    mse_loss  = nn.functional.mse_loss(predictions, targets)
    mono_loss = monotonicity_penalty(predictions)
    return mse_loss + lambda_mono * mono_loss


# ============================================================================
# DATA LOADING
# ============================================================================

def load_npy_fold(
    ws: int,
    fold: int,
    base_dir: str = DATASET_DIR,
) -> tuple:
    """
    Load 3D .npy arrays for a given window size and fold.

    Performs an ndim assertion immediately after loading to detect any
    preprocessing corruption early — before tensors enter the model.

    Args:
        ws:       Window size integer.
        fold:     1-based fold index.
        base_dir: Root output directory from the preprocessing pipeline.

    Returns:
        Tuple (X_train, y_train, X_val, y_val) as numpy float32 arrays.

    Raises:
        FileNotFoundError: If any expected .npy file is missing.
        AssertionError:    If any loaded array is not 3D (X) or 1D (y).
    """
    fold_dir = os.path.join(base_dir, f"ws_{ws}", f"fold_{fold}")

    files = {
        "X_train": os.path.join(fold_dir, "X_train.npy"),
        "y_train": os.path.join(fold_dir, "y_train.npy"),
        "X_val":   os.path.join(fold_dir, "X_val.npy"),
        "y_val":   os.path.join(fold_dir, "y_val.npy"),
    }

    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing preprocessed file: {path}\n"
                "Run data_preprocessing_pipeline.py first."
            )

    X_train = np.load(files["X_train"]).astype(np.float32)
    y_train = np.load(files["y_train"]).astype(np.float32)
    X_val   = np.load(files["X_val"]).astype(np.float32)
    y_val   = np.load(files["y_val"]).astype(np.float32)

    # 3D integrity assertions — catch reshape bugs before they corrupt training.
    assert X_train.ndim == 3, (
        f"X_train must be 3D [Samples, Window, Features], "
        f"got shape {X_train.shape}"
    )
    assert X_val.ndim == 3, (
        f"X_val must be 3D [Samples, Window, Features], "
        f"got shape {X_val.shape}"
    )
    assert y_train.ndim == 1, f"y_train must be 1D, got shape {y_train.shape}"
    assert y_val.ndim   == 1, f"y_val must be 1D, got shape {y_val.shape}"

    # Clipping guard: values must be within [-5, 5] from preprocessing.
    assert X_train.max() <= 5.0 + 1e-3, (
        f"X_train max={X_train.max():.4f} exceeds expected clip bound."
    )
    assert X_val.max() <= 5.0 + 1e-3, (
        f"X_val max={X_val.max():.4f} exceeds expected clip bound."
    )

    return X_train, y_train, X_val, y_val


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def apply_hard_monotone(predictions: np.ndarray) -> np.ndarray:
    """
    Apply hard monotonic constraint via cumulative minimum.

    Ensures that HI predictions are strictly non-increasing over time.
    Eliminates any residual "healing" artefacts after soft-constraint training.

    Args:
        predictions: 1-D array of raw model predictions.

    Returns:
        1-D array of monotonically non-increasing predictions.
    """
    return np.minimum.accumulate(predictions)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, and R2 regression metrics."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def check_prediction_collapse(pred: np.ndarray, fold: int, ws: int) -> bool:
    """
    Detect mean prediction collapse (std < 0.05) and log to stdout.

    Returns True if collapse is detected.
    """
    pred_std = float(pred.std())
    collapsed = pred_std < 0.05
    status = "  <<< MEAN COLLAPSE DETECTED >>>" if collapsed else ""
    print(
        f"  [PredDist] Fold={fold} WS={ws} | "
        f"min={pred.min():.4f}  max={pred.max():.4f}  "
        f"mean={pred.mean():.4f}  std={pred_std:.4f}{status}"
    )
    return collapsed


def plot_hi_prediction(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    fold:      int,
    ws:        int,
    out_dir:   str,
) -> None:
    """
    Plot Ground Truth vs Predicted Health Index for all validation bearings.

    Uses EMA smoothing (span=5) on the raw prediction before plotting.
    The hard-monotone post-processed curve is shown separately.

    Args:
        y_true:  Ground truth HI array.
        y_pred:  Raw model prediction array (before monotone constraint).
        fold:    Fold index (for title and filename).
        ws:      Window size (for title and filename).
        out_dir: Directory to save the output PNG.
    """
    x_min = np.arange(len(y_true)) * SUBMINUTE_HOP

    pred_ema      = pd.Series(y_pred).ewm(span=5, adjust=False).mean().values
    pred_monotone = apply_hard_monotone(pred_ema)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x_min, y_true,        color="#1f77b4", lw=2.0,
            label="Ground Truth HI", zorder=3)
    ax.plot(x_min, pred_ema,      color="#d62728", lw=1.5, ls="--",
            label="BiLSTM Prediction (EMA)", zorder=4)
    ax.plot(x_min, pred_monotone, color="#2ca02c", lw=1.5, ls=":",
            label="Hard Monotone (post-process)", zorder=5)

    ax.set_title(
        f"BiLSTM HI Prediction  |  WS={ws}  Fold={fold}",
        fontweight="bold",
    )
    ax.set_xlabel("Operating Time (Minutes)")
    ax.set_ylabel("Health Index")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fname = os.path.join(out_dir, f"hi_plot_ws{ws}_fold{fold}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved: {fname}")


# ============================================================================
# TRAINING AND EVALUATION LOOP
# ============================================================================

summary_metrics = []

for ws in WINDOW_SIZES:
    print(f"\n{'='*65}")
    print(f"  WINDOW SIZE: {ws}  ({ws * SUBMINUTE_HOP:.1f} min physical context)")
    print(f"{'='*65}")

    ws_results_dir = os.path.join(RESULTS_DIR, f"Window_Size_{ws}")
    os.makedirs(ws_results_dir, exist_ok=True)

    fold_metrics   = []
    diagnostic_log = {"window_size": ws}

    for fold in range(1, FOLDS + 1):
        fold_dir = os.path.join(ws_results_dir, f"Fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n  --- Fold {fold} ---")

        # ----------------------------------------------------------------
        # Load 3D .npy data (with assertions)
        # ----------------------------------------------------------------
        try:
            X_train_np, y_train_np, X_val_np, y_val_np = load_npy_fold(ws, fold)
        except (FileNotFoundError, AssertionError) as exc:
            print(f"  [WARNING] {exc}\n  Skipping fold {fold}.")
            continue

        print(
            f"  [INFO] X_train={X_train_np.shape}  y_train={y_train_np.shape}"
        )
        print(
            f"  [INFO] X_val  ={X_val_np.shape}    y_val  ={y_val_np.shape}"
        )

        # ----------------------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------------------
        X_train_t = torch.tensor(X_train_np).to(DEVICE)
        y_train_t = torch.tensor(y_train_np).view(-1, 1).to(DEVICE)
        X_val_t   = torch.tensor(X_val_np).to(DEVICE)
        y_val_t   = torch.tensor(y_val_np).view(-1, 1).to(DEVICE)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        # ----------------------------------------------------------------
        # Model, optimizer, scheduler
        # ----------------------------------------------------------------
        actual_features = X_train_np.shape[2]
        model = PhysicsInformedBiLSTM(
            input_size=actual_features,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)

        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        # ----------------------------------------------------------------
        # Training loop
        # ----------------------------------------------------------------
        t_start = time.time()
        model.train()

        for epoch in range(EPOCHS):
            epoch_mse_total  = 0.0
            epoch_mono_total = 0.0
            n_batches        = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred    = model(batch_x)
                mse_l   = nn.functional.mse_loss(pred, batch_y)
                mono_l  = monotonicity_penalty(pred)
                loss    = mse_l + LAMBDA_MONO * mono_l
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_mse_total  += mse_l.item()
                epoch_mono_total += mono_l.item()
                n_batches        += 1

            avg_mse  = epoch_mse_total  / max(n_batches, 1)
            avg_mono = epoch_mono_total / max(n_batches, 1)
            avg_total = avg_mse + LAMBDA_MONO * avg_mono
            scheduler.step(avg_total)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch [{epoch + 1:3d}/{EPOCHS}] "
                    f"MSE={avg_mse:.6f}  "
                    f"Mono={avg_mono:.6f}  "
                    f"Total={avg_total:.6f}"
                )

        print(f"  Training time: {time.time() - t_start:.1f}s")

        # ----------------------------------------------------------------
        # Batched inference (OOM-safe)
        # ----------------------------------------------------------------
        model.eval()
        val_loader = DataLoader(
            X_val_t, batch_size=BATCH_SIZE, shuffle=False
        )
        pred_list = []
        with torch.no_grad():
            for batch_x in val_loader:
                pred_list.append(model(batch_x).cpu().numpy().flatten())

        pred_raw = np.concatenate(pred_list)

        # ----------------------------------------------------------------
        # Hard monotone post-processing
        # ----------------------------------------------------------------
        if APPLY_HARD_MONOTONE:
            pred_final = apply_hard_monotone(pred_raw)
        else:
            pred_final = pred_raw

        pred_final = np.clip(pred_final, 0.0, 1.0)

        # ----------------------------------------------------------------
        # Collapse detection
        # ----------------------------------------------------------------
        collapsed = check_prediction_collapse(pred_final, fold, ws)
        diagnostic_log[f"fold_{fold}_collapsed"] = collapsed

        # ----------------------------------------------------------------
        # Metrics
        # ----------------------------------------------------------------
        y_val_arr = y_val_np
        metrics   = compute_metrics(y_val_arr, pred_final)
        fold_metrics.append({"Fold": fold, **metrics})
        print(
            f"  [Fold {fold}] "
            f"RMSE={metrics['RMSE']:.4f}  "
            f"MAE={metrics['MAE']:.4f}  "
            f"R2={metrics['R2']:.4f}"
        )

        # ----------------------------------------------------------------
        # Save fold artifacts
        # ----------------------------------------------------------------
        results_df = pd.DataFrame({
            "True_HI":       y_val_arr,
            "Pred_HI_Raw":   pred_raw,
            "Pred_HI_Final": pred_final,
        })
        results_df.to_csv(
            os.path.join(fold_dir, f"predictions_ws{ws}_fold{fold}.csv"),
            index=False,
        )
        torch.save(
            model.state_dict(),
            os.path.join(fold_dir, f"bilstm_model_ws{ws}_fold{fold}.pth"),
        )
        plot_hi_prediction(
            y_true=y_val_arr,
            y_pred=pred_raw,
            fold=fold,
            ws=ws,
            out_dir=fold_dir,
        )

        # ----------------------------------------------------------------
        # OOM prevention: explicit cleanup after each fold
        # ----------------------------------------------------------------
        del X_train_t, y_train_t, X_val_t, y_val_t, model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------------
    # Window-size level aggregation and diagnostic log
    # --------------------------------------------------------------------
    diag_path = os.path.join(ws_results_dir, f"diagnostic_log_ws{ws}.json")
    with open(diag_path, "w", encoding="utf-8") as fh:
        json.dump(diagnostic_log, fh, indent=2)
    print(f"\n  [Diagnostic] Log saved: {diag_path}")

    if fold_metrics:
        df_fold = pd.DataFrame(fold_metrics)
        df_fold.to_csv(
            os.path.join(ws_results_dir, f"fold_metrics_ws{ws}.csv"),
            index=False,
        )
        avg_rmse = df_fold["RMSE"].mean()
        avg_mae  = df_fold["MAE"].mean()
        avg_r2   = df_fold["R2"].mean()

        print(f"\n  => Window Size {ws} Global Performance:")
        print(f"     Mean RMSE : {avg_rmse:.4f}")
        print(f"     Mean MAE  : {avg_mae:.4f}")
        print(f"     Mean R2   : {avg_r2:.4f}")

        summary_metrics.append({
            "Window_Size": ws,
            "Mean_RMSE":   avg_rmse,
            "Mean_MAE":    avg_mae,
            "Mean_R2":     avg_r2,
        })

# ============================================================================
# GLOBAL SUMMARY
# ============================================================================
if summary_metrics:
    summary_df = pd.DataFrame(summary_metrics)
    summary_path = os.path.join(RESULTS_DIR, "BiLSTM_Global_Window_Comparison.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL SUMMARY ACROSS ALL WINDOW SIZES")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")
else:
    print(
        "[WARNING] No metrics collected. "
        "Verify .npy files exist under DATASET_DIR."
    )
