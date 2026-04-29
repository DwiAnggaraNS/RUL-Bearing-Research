"""
3_TRAINING_LSTM.py
==================
LSTM Training and Evaluation Script with:
  - 28-feature input (sub-minute aggregation: 14 means + 14 stds)
  - Hierarchical output: Training_Results/Window_Size_{ws}/Fold_{fold}/
  - Deep diagnostic tracking (collapse detection)
  - Strict OOM prevention with explicit memory cleanup per fold
  - Full life-cycle visualization (X-axis from absolute minute 0)
  - 5-Fold Cross-Validation over XJTU Condition 1 bearings

Author: Auto-generated (refactored)
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
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Publication-standard matplotlib settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#bfbfbf",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})
sns.set_palette("husl")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = r"D:\Proyek Dosen\Riset Bearing\XJTU-SY_Bearing_Datasets\Processed_Data\LSTM_Inputs"
RESULTS_DIR = r"D:\Proyek Dosen\Riset Bearing\Training_Results"

WINDOW_SIZES = [10, 20, 30]
FOLDS = 5

# Input dimensionality: 14 extracted features x (mean + std) = 28 per timestep
NUM_FEATURES = 28

# Model hyperparameters
EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 128
NUM_LAYERS = 2
NOISE_STD = 0.0          # Disabled per preprocessing audit recommendation
DROPOUT = 0.1             # Halved from 0.2

# Subminute hop factor: each row = 0.5 physical minutes
# (integer minute rows + half-minute rows from sub-minute aggregation)
SUBMINUTE_HOP = 0.5       # minutes per timestep index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------
class RobustLSTM(nn.Module):
    """
    Vanilla LSTM regression head for Health Index prediction.

    Architecture:
        LSTM (input_size -> hidden_dim, num_layers) -> Dropout -> Linear -> scalar

    Notes:
        - No activation on output layer (avoids Dying ReLU and output clamping).
        - Optional Gaussian noise injection during training only.
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        noise_std: float = 0.0,
    ):
        super(RobustLSTM, self).__init__()
        self.noise_std = noise_std

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(dropout)
        # Strictly no ReLU on regression output
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.linear(out)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def reconstruct_3d(flat_array: np.ndarray, window_size: int, num_features: int) -> np.ndarray:
    """
    Reshape a 2D flattened sequence array back to 3D topology.

    The preprocessing pipeline stores windows in row-major order:
        flat columns = [feat_0_t0, feat_1_t0, ..., feat_N_t0,
                        feat_0_t1, ..., feat_N_t(ws-1)]
    i.e., shape (samples, window_size * num_features), stored as
        (samples, num_features, window_size) then transposed.

    Args:
        flat_array:   Shape (samples, window_size * num_features).
        window_size:  Number of timesteps per sequence window.
        num_features: Number of features per timestep.

    Returns:
        3D array of shape (samples, window_size, num_features).
    """
    return flat_array.reshape(-1, num_features, window_size).transpose(0, 2, 1)


def run_pre_training_sanity_check(
    feature_cols: list,
    x_train_flat: np.ndarray,
    x_val_flat: np.ndarray,
    log_path: str,
) -> dict:
    """
    Log feature column names, order, and global min/max bounds before training.

    Confirms that per-bearing MinMax scaling in the preprocessing pipeline
    actually bounded feature values to [0, 1].

    Args:
        feature_cols:  Ordered list of feature column names.
        x_train_flat:  2D training feature matrix (samples x flat_features).
        x_val_flat:    2D validation feature matrix (samples x flat_features).
        log_path:      Path to write the sanity check JSON log.

    Returns:
        Dictionary containing the sanity check report.
    """
    report = {
        "feature_count": len(feature_cols),
        "feature_order": feature_cols,
        "train_global_min": float(x_train_flat.min()),
        "train_global_max": float(x_train_flat.max()),
        "val_global_min": float(x_val_flat.min()),
        "val_global_max": float(x_val_flat.max()),
        "train_any_nan": bool(np.isnan(x_train_flat).any()),
        "val_any_nan": bool(np.isnan(x_val_flat).any()),
        "train_any_inf": bool(np.isinf(x_train_flat).any()),
        "val_any_inf": bool(np.isinf(x_val_flat).any()),
    }

    print("[SANITY CHECK] Feature count      :", report["feature_count"])
    print("[SANITY CHECK] Train global min   :", report["train_global_min"])
    print("[SANITY CHECK] Train global max   :", report["train_global_max"])
    print("[SANITY CHECK] Val global min     :", report["val_global_min"])
    print("[SANITY CHECK] Val global max     :", report["val_global_max"])
    print("[SANITY CHECK] Train NaN present  :", report["train_any_nan"])
    print("[SANITY CHECK] Val NaN present    :", report["val_any_nan"])

    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


def log_prediction_distribution(
    pred_hi: np.ndarray,
    log_dict: dict,
    fold: int,
    ws: int,
) -> dict:
    """
    Compute and log prediction distribution statistics to detect mean collapse.

    If the model collapses to a constant prediction, Std -> 0 and Mean ~0.5.
    This function surfaces that condition explicitly.

    Args:
        pred_hi:   Raw (clipped) predicted Health Index array.
        log_dict:  Existing diagnostic log dictionary to update.
        fold:      Current fold index.
        ws:        Current window size.

    Returns:
        Updated log_dict with prediction distribution entry.
    """
    stats = {
        "fold": fold,
        "window_size": ws,
        "pred_min": float(pred_hi.min()),
        "pred_max": float(pred_hi.max()),
        "pred_mean": float(pred_hi.mean()),
        "pred_std": float(pred_hi.std()),
    }
    collapsed = stats["pred_std"] < 0.05
    stats["collapse_warning"] = collapsed

    print(
        f"[PRED DIST] Fold {fold} | WS {ws} | "
        f"Min={stats['pred_min']:.4f}  Max={stats['pred_max']:.4f}  "
        f"Mean={stats['pred_mean']:.4f}  Std={stats['pred_std']:.4f}"
        + ("  <<< COLLAPSE DETECTED >>>" if collapsed else "")
    )

    key = f"fold_{fold}_pred_distribution"
    log_dict[key] = stats
    return log_dict


def plot_health_index(
    smoothed_results: pd.DataFrame,
    ws: int,
    fold: int,
    output_dir: str,
) -> None:
    """
    Plot full life-cycle Ground Truth vs Smoothed Predicted Health Index.

    Requirements enforced:
      - X-axis starts at absolute minute 0 (full life from machine start-up).
      - X-axis label: "Operating Time (Minutes)".
      - Mapping: step_index * SUBMINUTE_HOP -> absolute minutes.
      - True_HI plotted as solid blue line.
      - Smoothed_Predicted_HI plotted as dashed red line.
      - Horizontal plateau (True_HI == 1.0) clearly visible.

    Args:
        smoothed_results: DataFrame with columns:
                          [Bearing_ID, Step_Index, True_HI, Smoothed_Predicted_HI].
        ws:               Window size (used in title and filename).
        fold:             Fold index (used in title and filename).
        output_dir:       Directory to save the plot PNG.
    """
    if "Bearing_ID" in smoothed_results.columns:
        bearings = smoothed_results["Bearing_ID"].unique()
    else:
        bearings = ["Unknown_Bearing"]

    n_bearings = len(bearings)
    fig, axes = plt.subplots(
        n_bearings, 1,
        figsize=(12, 4 * n_bearings),
        squeeze=False,
    )

    for idx, bearing in enumerate(sorted(bearings)):
        ax = axes[idx, 0]
        b_df = smoothed_results[
            smoothed_results["Bearing_ID"] == bearing
        ].copy()

        # Reconstruct full operating time axis from absolute step index
        if "Step_Index" in b_df.columns:
            b_df = b_df.sort_values("Step_Index")
            x_minutes = b_df["Step_Index"].values * SUBMINUTE_HOP
        elif "Original_Minute" in b_df.columns:
            b_df = b_df.sort_values("Original_Minute")
            x_minutes = b_df["Original_Minute"].values
        else:
            b_df = b_df.reset_index(drop=True)
            x_minutes = b_df.index.values * SUBMINUTE_HOP

        true_hi = b_df["True_HI"].values
        pred_col = (
            "Smoothed_Predicted_HI"
            if "Smoothed_Predicted_HI" in b_df.columns
            else "Predicted_HI"
        )
        pred_hi = b_df[pred_col].values

        # Solid blue for ground truth (plateau + descent clearly visible)
        ax.plot(x_minutes, true_hi, color="#1f77b4", linewidth=2.0,
                label="Ground Truth HI", zorder=3)
        # Dashed red for prediction
        ax.plot(x_minutes, pred_hi, color="#d62728", linewidth=1.8,
                linestyle="--", label="LSTM Prediction (EMA Smoothed)", zorder=4)

        # Shade the healthy plateau region
        if (true_hi == 1.0).any():
            plateau_end_min = x_minutes[true_hi == 1.0][-1]
            ax.axvspan(
                x_minutes[0], plateau_end_min,
                alpha=0.07, color="#1f77b4", label="Healthy Phase",
            )

        ax.set_title(
            f"LSTM HI Prediction  |  WS={ws}  Fold={fold}  |  {bearing}",
            pad=8,
        )
        ax.set_xlabel("Operating Time (Minutes)")
        ax.set_ylabel("Health Index (HI)")
        ax.set_xlim(left=0)
        ax.set_ylim(-0.05, 1.10)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9,
                  edgecolor="black")

    plt.tight_layout()
    fname = f"hi_plot_ws{ws}_fold{fold}.png"
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved: {fpath}")


# ---------------------------------------------------------------------------
# Main Training and Evaluation Loop
# ---------------------------------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
summary_metrics = []

print("=" * 70)
print("LSTM TRAINING AND EVALUATION PIPELINE")
print(f"Device       : {DEVICE}")
print(f"NUM_FEATURES : {NUM_FEATURES}")
print(f"HIDDEN_DIM   : {HIDDEN_DIM}   NUM_LAYERS : {NUM_LAYERS}")
print(f"EPOCHS       : {EPOCHS}   BATCH_SIZE : {BATCH_SIZE}")
print(f"LEARNING_RATE: {LEARNING_RATE}   WEIGHT_DECAY: {WEIGHT_DECAY}")
print(f"DROPOUT      : {DROPOUT}   NOISE_STD  : {NOISE_STD}")
print("=" * 70)

for ws in WINDOW_SIZES:
    print(f"\n{'='*70}")
    print(f"  WINDOW SIZE: {ws}  ({ws * SUBMINUTE_HOP:.1f} physical minutes context)")
    print(f"{'='*70}")

    ws_data_dir = os.path.join(DATASET_DIR, f"window_size_{ws}")
    ws_results_dir = os.path.join(RESULTS_DIR, f"Window_Size_{ws}")
    os.makedirs(ws_results_dir, exist_ok=True)

    fold_metrics = []
    ws_all_predictions = pd.DataFrame()
    diagnostic_log = {"window_size": ws}

    for fold in range(1, FOLDS + 1):
        fold_dir = os.path.join(ws_results_dir, f"Fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        print(f"\n  --- Fold {fold} ---")

        train_file = os.path.join(ws_data_dir, f"processed_train_fold{fold}.csv")
        val_file = os.path.join(ws_data_dir, f"processed_val_fold{fold}.csv")

        if not os.path.exists(train_file) or not os.path.exists(val_file):
            print(f"  [Warning] Missing CSV for Fold {fold}. Skipping.")
            continue

        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)

        # ---------------------------------------------------------------
        # Feature extraction: drop non-feature metadata columns
        # ---------------------------------------------------------------
        meta_cols = [
            "Health_Index", "Target_RUL", "Bearing_ID",
            "Change_Point", "Original_Minute", "Minute", "Step_Index",
        ]
        drop_cols = [c for c in meta_cols if c in df_train.columns]

        df_train_num = df_train.select_dtypes(include=[np.number])
        df_val_num = df_val.select_dtypes(include=[np.number])

        feature_cols_train = [
            c for c in df_train_num.columns
            if c not in drop_cols
        ]
        feature_cols_val = [
            c for c in df_val_num.columns
            if c not in drop_cols
        ]

        x_train_flat = df_train_num[feature_cols_train].values.astype(np.float32)
        x_val_flat = df_val_num[feature_cols_val].values.astype(np.float32)
        y_train = df_train["Health_Index"].values.astype(np.float32)
        y_val = df_val["Health_Index"].values.astype(np.float32)

        # ---------------------------------------------------------------
        # Pre-Training Sanity Check
        # ---------------------------------------------------------------
        sanity_log_path = os.path.join(fold_dir, "sanity_check.json")
        run_pre_training_sanity_check(
            feature_cols=feature_cols_train,
            x_train_flat=x_train_flat,
            x_val_flat=x_val_flat,
            log_path=sanity_log_path,
        )

        # Derive actual_num_features from flat array (guards against CSV drift)
        actual_num_features = x_train_flat.shape[1] // ws
        print(f"  [INFO] Flat features per sample : {x_train_flat.shape[1]}")
        print(f"  [INFO] Derived num_features     : {actual_num_features}")

        if actual_num_features != NUM_FEATURES:
            print(
                f"  [WARNING] Expected NUM_FEATURES={NUM_FEATURES} "
                f"but derived {actual_num_features}. "
                "Proceeding with derived value."
            )

        # ---------------------------------------------------------------
        # 3D Reconstruction: (samples, ws, num_features)
        # ---------------------------------------------------------------
        x_train_3d = reconstruct_3d(x_train_flat, ws, actual_num_features)
        x_val_3d = reconstruct_3d(x_val_flat, ws, actual_num_features)

        # ---------------------------------------------------------------
        # PyTorch Tensor Conversion
        # ---------------------------------------------------------------
        X_train_tensor = torch.tensor(x_train_3d, dtype=torch.float32).to(DEVICE)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
        X_val_tensor = torch.tensor(x_val_3d, dtype=torch.float32).to(DEVICE)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(DEVICE)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=False,  # Already on DEVICE; pin_memory for CPU->GPU only
        )

        # ---------------------------------------------------------------
        # Model, Optimizer, Loss
        # ---------------------------------------------------------------
        model = RobustLSTM(
            input_size=actual_num_features,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            noise_std=NOISE_STD,
        ).to(DEVICE)

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.MSELoss()

        # ---------------------------------------------------------------
        # Training Loop
        # ---------------------------------------------------------------
        t_start = time.time()
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(len(train_loader), 1)
                print(
                    f"  Epoch [{epoch + 1:3d}/{EPOCHS}] "
                    f"Train MSE Loss: {avg_loss:.6f}"
                )
        print(f"  Training time: {time.time() - t_start:.1f}s")

        # ---------------------------------------------------------------
        # Batched Inference (OOM-safe)
        # ---------------------------------------------------------------
        model.eval()
        val_loader = DataLoader(
            X_val_tensor,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        pred_hi_list = []
        with torch.no_grad():
            for batch_x in val_loader:
                batch_pred = model(batch_x).cpu().numpy().flatten()
                pred_hi_list.append(batch_pred)

        pred_hi_raw = np.concatenate(pred_hi_list)
        pred_hi = np.clip(pred_hi_raw, 0.0, 1.0)

        # ---------------------------------------------------------------
        # Prediction Distribution Logging (Collapse Detection)
        # ---------------------------------------------------------------
        diagnostic_log = log_prediction_distribution(
            pred_hi=pred_hi,
            log_dict=diagnostic_log,
            fold=fold,
            ws=ws,
        )

        # ---------------------------------------------------------------
        # Build validation results DataFrame
        # ---------------------------------------------------------------
        val_results = pd.DataFrame()
        val_results["Bearing_ID"] = (
            df_val["Bearing_ID"].values
            if "Bearing_ID" in df_val.columns
            else "Unknown"
        )

        if "Step_Index" in df_val.columns:
            val_results["Step_Index"] = df_val["Step_Index"].values
        elif "Original_Minute" in df_val.columns:
            val_results["Step_Index"] = (
                df_val["Original_Minute"].values / SUBMINUTE_HOP
            ).astype(int)
        else:
            val_results["Step_Index"] = np.arange(len(df_val))

        val_results["Original_Minute"] = val_results["Step_Index"] * SUBMINUTE_HOP

        if "Change_Point" in df_val.columns:
            val_results["Change_Point"] = df_val["Change_Point"].values

        val_results["True_HI"] = y_val
        val_results["Predicted_HI"] = pred_hi
        val_results["Fold"] = fold
        val_results["Window_Size"] = ws

        # ---------------------------------------------------------------
        # Ensemble Averaging (minute-level grouping) + EMA Smoothing
        # ---------------------------------------------------------------
        smoothed_dfs = []
        for b_id, b_df in val_results.groupby("Bearing_ID"):
            b_df = b_df.sort_values("Step_Index").copy()
            # EMA smoothing (span=5 ~ 2.5 physical minutes)
            b_df["Smoothed_Predicted_HI"] = (
                b_df["Predicted_HI"].ewm(span=5, adjust=False).mean()
            )
            smoothed_dfs.append(b_df)

        smoothed_results = pd.concat(smoothed_dfs, ignore_index=True)

        # ---------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------
        rmse_val = calculate_rmse(
            smoothed_results["True_HI"],
            smoothed_results["Smoothed_Predicted_HI"],
        )
        mae_val = mean_absolute_error(
            smoothed_results["True_HI"],
            smoothed_results["Smoothed_Predicted_HI"],
        )
        r2_val = r2_score(
            smoothed_results["True_HI"],
            smoothed_results["Smoothed_Predicted_HI"],
        )

        fold_metrics.append({"Fold": fold, "RMSE": rmse_val, "MAE": mae_val, "R2": r2_val})
        print(
            f"  [Fold {fold}] RMSE={rmse_val:.4f}  "
            f"MAE={mae_val:.4f}  R2={r2_val:.4f}"
        )

        # ---------------------------------------------------------------
        # Per-Fold Artifact Saving
        # ---------------------------------------------------------------
        smoothed_results.to_csv(
            os.path.join(fold_dir, f"predictions_ws{ws}_fold{fold}.csv"),
            index=False,
        )
        torch.save(
            model.state_dict(),
            os.path.join(fold_dir, f"lstm_model_ws{ws}_fold{fold}.pth"),
        )

        # Plot (all folds for diagnostic visibility)
        plot_health_index(
            smoothed_results=smoothed_results,
            ws=ws,
            fold=fold,
            output_dir=fold_dir,
        )

        ws_all_predictions = pd.concat(
            [ws_all_predictions, smoothed_results], ignore_index=True
        )

        # ---------------------------------------------------------------
        # Strict OOM Prevention: explicit cleanup after EACH fold
        # ---------------------------------------------------------------
        del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # Diagnostic Log: write consolidated JSON for this window size
    # -------------------------------------------------------------------
    diag_path = os.path.join(ws_results_dir, f"diagnostic_log_ws{ws}.json")
    with open(diag_path, "w", encoding="utf-8") as fh:
        json.dump(diagnostic_log, fh, indent=2)
    print(f"\n[DIAGNOSTIC] Log saved: {diag_path}")

    # -------------------------------------------------------------------
    # Window-Size Level Aggregation
    # -------------------------------------------------------------------
    if fold_metrics:
        df_fold_metrics = pd.DataFrame(fold_metrics)
        df_fold_metrics.to_csv(
            os.path.join(ws_results_dir, f"fold_metrics_ws{ws}.csv"),
            index=False,
        )
        if not ws_all_predictions.empty:
            ws_all_predictions.to_csv(
                os.path.join(ws_results_dir, f"all_predictions_ws{ws}.csv"),
                index=False,
            )

        avg_rmse = df_fold_metrics["RMSE"].mean()
        avg_mae = df_fold_metrics["MAE"].mean()
        avg_r2 = df_fold_metrics["R2"].mean()

        print(f"\n  => Window Size {ws} Global Performance:")
        print(f"     Mean RMSE : {avg_rmse:.4f}")
        print(f"     Mean MAE  : {avg_mae:.4f}")
        print(f"     Mean R2   : {avg_r2:.4f}")

        summary_metrics.append({
            "Window_Size": ws,
            "Mean_RMSE": avg_rmse,
            "Mean_MAE": avg_mae,
            "Mean_R2": avg_r2,
        })

# ---------------------------------------------------------------------------
# Global Summary
# ---------------------------------------------------------------------------
if summary_metrics:
    summary_df = pd.DataFrame(summary_metrics)
    summary_path = os.path.join(RESULTS_DIR, "LSTM_Global_Window_Comparison.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY ACROSS ALL WINDOW SIZES")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        summary_df["Window_Size"].astype(str),
        summary_df["Mean_RMSE"],
        color=sns.color_palette("viridis", len(summary_df)),
        edgecolor="black",
        linewidth=1.2,
    )
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.4f}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            ha="center", va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title(
        "Mean Validation RMSE Across Window Sizes (LSTM, 5-Fold CV)",
        pad=12,
    )
    ax.set_xlabel("Window Size (Timesteps)")
    ax.set_ylabel("Mean Validation RMSE (HI scale, 0-1)")
    ax.set_ylim(0, summary_df["Mean_RMSE"].max() * 1.25)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "LSTM_RMSE_Comparison_BarChart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Bar chart saved: {chart_path}")
else:
    print("[Warning] No metrics collected. Review dataset and fold CSV files.")
