"""
data_preprocessing_pipeline.py
================================
Refactored Data Preprocessing Pipeline for Cross-Condition RUL Prediction.

Architectural corrections applied (ref: "Beyond Accuracy" and "Claude_1may" audits):
    1. Frequency cap removed: full 25.6 kHz spectrum used in feature extraction.
    2. 30-second hop fixed: sub-minute aggregation is strictly intra-file.
       Each CSV file produces exactly two aggregated rows (first-half / second-half).
    3. Causal HI construction: CUSUM and HI labeling execute AFTER aggregation.
    4. Scaling + clipping: Per-bearing MinMaxScaler followed by .clip(-5.0, 5.0)
       to prevent gradient saturation.
    5. Phase 3 visual validation: Spearman rank correlation check with halt warning.
    6. Native 3D saving: sliding windows saved as .npy [Samples, Window, 28_Features].

Pipeline phases:
    Phase 1  - Intra-file raw feature extraction (1024-sample non-overlapping windows)
    Phase 2  - Intra-file sub-minute aggregation  (2 rows per CSV file)
    Phase 3  - Causal ground truth construction   (CUSUM on aggregated RMS)
    Phase 4  - Mandatory visual validation        (Spearman correlation + plots)
    Phase 5  - Per-bearing MinMax scaling + clip  (.clip(-5.0, 5.0))
    Phase 6  - Sliding windows + native 3D .npy save

Author: DwiAnggaraNS / PHM Research Team
PEP8: compliant
Emojis: none
"""

import os
import sys
import glob
import warnings
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(__file__))
from CrossDomainFeatureExtractor import CrossDomainFeatureExtractor
from ConstructGroundTruthUsingCUSUM import UnivariateCUSUMDetector

# ============================================================================
# CONFIGURATION
# ============================================================================
# Root folder containing one sub-folder per bearing (each sub-folder holds
# the per-minute CSV files, e.g. acc_00001.csv ... acc_00123.csv).
INPUT_PATH = (
    r"D:\Proyek Dosen\Riset Bearing\XJTU-SY_Bearing_Datasets"
    r"\Processed_Data\Downsampled"
)
OUTPUT_PATH = (
    r"D:\Proyek Dosen\Riset Bearing\XJTU-SY_Bearing_Datasets"
    r"\Processed_Data\LSTM_Inputs_v2"
)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Signal processing
XJTU_SAMPLING_RATE: float = 25600.0   # Hz
SEGMENT_LENGTH: int = 1024            # samples per non-overlapping window
# NOTE: MAX_FREQ_HZ is intentionally absent. Full spectrum is used.

# Sliding window sizes for LSTM sequences (in sub-minute timestep units).
# WS=1 -> 0.5 min context; WS=3 -> 1.5 min; WS=5 -> 2.5 min.
WINDOW_SIZES: List[int] = [1, 3, 5]

# Scaling
CLIP_LOW:  float = -5.0
CLIP_HIGH: float =  5.0

# Spearman threshold: warn if median |rho| falls below this value.
SPEARMAN_WARN_THRESHOLD: float = 0.30

# CUSUM parameters (conservative)
CUSUM_BASELINE_RATIO: float = 0.20
CUSUM_K_FACTOR:       float = 0.50
CUSUM_H_FACTOR:       float = 8.00
CUSUM_SMOOTHING_WIN:  int   = 5     # rolling mean applied to aggregated RMS

# Cross-validation fold definitions (XJTU-SY 3-condition, 5-bearing layout)
ALL_BEARINGS: List[str] = [
    f"Bearing{c}_{i}" for c in range(1, 4) for i in range(1, 6)
]
FOLDS: List[Dict[str, List[str]]] = [
    {"val": [f"Bearing{c}_{i}" for c in range(1, 4)]}
    for i in range(1, 6)
]
for _fold in FOLDS:
    _fold["train"] = [b for b in ALL_BEARINGS if b not in _fold["val"]]

# 14 raw feature column names (must match CrossDomainFeatureExtractor output)
RAW_FEATURE_COLS: List[str] = [
    "td_mean", "td_std", "td_rms", "td_peak_value", "td_p2p",
    "td_skewness", "td_kurtosis", "td_peak_factor",
    "td_clearance_factor", "td_shape_factor", "td_impulse_factor",
    "fd_mean_freq", "fd_centroid_freq", "fd_fft_p2p",
]

# 28 aggregated feature column names (mean + std of each raw feature)
AGG_FEATURE_COLS: List[str] = (
    [f"{f}_mean" for f in RAW_FEATURE_COLS]
    + [f"{f}_std"  for f in RAW_FEATURE_COLS]
)

META_COLS: List[str] = [
    "Bearing_ID", "File_Index", "Half_Index",
    "Time_Index", "hi_target", "cusum_change_point",
]

print("=" * 65)
print("DATA PREPROCESSING PIPELINE  (v2 — full audit fixes applied)")
print("=" * 65)
print(f"  Input  : {INPUT_PATH}")
print(f"  Output : {OUTPUT_PATH}")
print(f"  Windows: {WINDOW_SIZES}")
print(f"  Clip   : [{CLIP_LOW}, {CLIP_HIGH}]")
print("=" * 65)


# ============================================================================
# PHASE 1 — INTRA-FILE RAW FEATURE EXTRACTION
# ============================================================================

def extract_features_from_csv(
    csv_path: str,
    bearing_id: str,
    file_index: int,
) -> Optional[pd.DataFrame]:
    """
    Extract 14 raw features from a single per-minute CSV file.

    The file is sliced into non-overlapping 1024-sample windows.
    Features are extracted per window from the horizontal (H) vibration channel.
    No aggregation, no scaling, no labeling occurs here.

    Args:
        csv_path:   Absolute path to the per-minute CSV file.
        bearing_id: Canonical bearing identifier (e.g., "Bearing1_3").
        file_index: 1-based integer index of the CSV file in the bearing's
                    chronological sequence.

    Returns:
        DataFrame of shape (n_windows, 14+meta) or None on failure.
    """
    try:
        df_raw = pd.read_csv(csv_path, header=None)
    except Exception as exc:
        print(f"    [WARNING] Cannot read {csv_path}: {exc}")
        return None

    # XJTU-SY format: column 0 = horizontal, column 1 = vertical.
    if df_raw.shape[1] < 1:
        return None

    h_signal = df_raw.iloc[:, 0].to_numpy(dtype=np.float64)
    n_total   = len(h_signal)

    if n_total < SEGMENT_LENGTH:
        return None

    extractor = CrossDomainFeatureExtractor(sampling_rate=XJTU_SAMPLING_RATE)

    n_windows = n_total // SEGMENT_LENGTH
    rows = []
    for w in range(n_windows):
        start = w * SEGMENT_LENGTH
        seg   = h_signal[start: start + SEGMENT_LENGTH]
        feats = extractor.extract_all_features(seg)
        feats["Bearing_ID"]  = bearing_id
        feats["File_Index"]  = file_index
        feats["Window_Index"] = w
        rows.append(feats)

    if not rows:
        return None

    return pd.DataFrame(rows)


# ============================================================================
# PHASE 2 — INTRA-FILE SUB-MINUTE AGGREGATION (2 rows per CSV file)
# ============================================================================

def aggregate_intrafile(df_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate window-level features within a single CSV file into two rows.

    Split the windows into two equal halves (first half = row 0,
    second half = row 1).  For each of the 14 raw features, compute
    the mean and std across the windows in each half -> 28 features per row.

    This is the corrected "30-second hop" logic.  Aggregation never
    crosses a file boundary, eliminating the data leakage present in the
    previous implementation.

    Args:
        df_windows: All window-level features for one CSV file.
                    Must contain Bearing_ID, File_Index columns.

    Returns:
        DataFrame with exactly 2 rows and 28 aggregated features + meta.
    """
    bearing_id = df_windows["Bearing_ID"].iloc[0]
    file_index = df_windows["File_Index"].iloc[0]
    feat_vals  = df_windows[RAW_FEATURE_COLS].values
    n_windows  = len(feat_vals)

    midpoint = n_windows // 2 if n_windows >= 2 else 1

    halves = [
        feat_vals[:midpoint],
        feat_vals[midpoint:] if n_windows >= 2 else feat_vals,
    ]

    rows = []
    for half_idx, half_vals in enumerate(halves):
        if len(half_vals) == 0:
            continue
        row: Dict[str, object] = {"Bearing_ID": bearing_id,
                                   "File_Index":  file_index,
                                   "Half_Index":  half_idx}
        for j, feat in enumerate(RAW_FEATURE_COLS):
            col_vals = half_vals[:, j]
            row[f"{feat}_mean"] = float(np.mean(col_vals))
            row[f"{feat}_std"]  = float(np.std(col_vals, ddof=1)
                                        if len(col_vals) > 1 else 0.0)
        rows.append(row)

    df_agg = pd.DataFrame(rows)
    # Sequential time index: (file_index-1)*2 + half_index
    df_agg["Time_Index"] = (
        (df_agg["File_Index"] - 1) * 2 + df_agg["Half_Index"]
    ).astype(float)
    return df_agg


def run_phase1_and_phase2(
    bearing_csv_dir: str,
    bearing_id: str,
) -> Optional[pd.DataFrame]:
    """
    Execute Phases 1 and 2 for a single bearing.

    Discovers all per-minute CSV files, processes them in chronological order,
    and returns the intra-file aggregated time-series.

    Args:
        bearing_csv_dir: Directory containing the per-minute CSV files.
        bearing_id:      Canonical bearing identifier.

    Returns:
        Aggregated DataFrame (sorted by Time_Index) or None if no data found.
    """
    csv_files = sorted(glob.glob(os.path.join(bearing_csv_dir, "*.csv")))
    if not csv_files:
        print(f"  [WARNING] No CSV files in {bearing_csv_dir}")
        return None

    all_agg_rows: List[pd.DataFrame] = []

    for file_idx, csv_path in enumerate(csv_files, start=1):
        df_windows = extract_features_from_csv(csv_path, bearing_id, file_idx)
        if df_windows is None or df_windows.empty:
            continue
        df_agg = aggregate_intrafile(df_windows)
        all_agg_rows.append(df_agg)

    if not all_agg_rows:
        return None

    df_bearing = pd.concat(all_agg_rows, ignore_index=True)
    df_bearing = df_bearing.sort_values("Time_Index").reset_index(drop=True)
    df_bearing = df_bearing.fillna(0.0)

    print(
        f"  {bearing_id}: {len(csv_files)} files -> "
        f"{len(df_bearing)} sub-minute rows"
    )
    return df_bearing


# ============================================================================
# PHASE 3 — CAUSAL GROUND TRUTH CONSTRUCTION
# ============================================================================

def construct_causal_hi(df_agg: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Construct piecewise-linear Health Index (HI) targets on the aggregated series.

    Execution order is strictly causal: CUSUM is applied to the aggregated RMS
    column (never to raw windows), and the HI target is derived from the
    detected change point.

    HI definition (piecewise linear):
        t < T_cp  -> HI = 1.0  (healthy plateau)
        t >= T_cp -> HI = 1.0 - (t - T_cp) / (T_total - T_cp)  (linear decay)
        final row -> HI = 0.0  (guaranteed failure)

    Args:
        df_agg: Intra-file aggregated DataFrame with AGG_FEATURE_COLS columns.

    Returns:
        Tuple of (df with hi_target + cusum_change_point columns, metadata dict).
    """
    df_out      = df_agg.copy()
    n_total     = len(df_out)
    bearing_id  = df_out["Bearing_ID"].iloc[0]

    rms_col = "td_rms_mean"
    if rms_col not in df_out.columns:
        rms_candidates = [c for c in df_out.columns if "rms" in c.lower()]
        raise ValueError(
            f"Column '{rms_col}' not found in {bearing_id}. "
            f"Available RMS-related columns: {rms_candidates}"
        )

    smoothed_rms = (
        df_out[rms_col]
        .rolling(window=CUSUM_SMOOTHING_WIN, min_periods=1, center=True)
        .mean()
        .values
    )

    detector = UnivariateCUSUMDetector(
        baseline_ratio=CUSUM_BASELINE_RATIO,
        k_factor=CUSUM_K_FACTOR,
        h_factor=CUSUM_H_FACTOR,
    )
    change_point, _ = detector.fit_predict(smoothed_rms)

    # Piecewise linear HI
    hi = np.ones(n_total, dtype=np.float64)
    degradation_len = n_total - change_point
    if degradation_len > 1:
        hi[change_point:] = np.linspace(1.0, 0.0, degradation_len)
    elif degradation_len == 1:
        hi[change_point] = 0.0

    hi[-1] = 0.0  # Hard boundary: final sample is always failure.

    df_out["hi_target"]           = hi
    df_out["cusum_change_point"]  = int(change_point)

    metadata = {
        "Bearing_ID":           bearing_id,
        "Total_Timesteps":      n_total,
        "Change_Point_Index":   int(change_point),
        "Change_Point_Minute":  float(df_out["Time_Index"].iloc[change_point])
                                if change_point < n_total else -1.0,
        "HI_Min":               float(hi.min()),
        "HI_Max":               float(hi.max()),
    }
    return df_out, metadata


# ============================================================================
# PHASE 4 — MANDATORY VISUAL VALIDATION (Spearman correlation)
# ============================================================================

def validate_and_plot(
    df_gt: pd.DataFrame,
    bearing_id: str,
    output_dir: str,
) -> float:
    """
    Plot 28 aggregated features vs HI target and compute Spearman correlations.

    Emits a WARNING banner to stdout if the median absolute Spearman rho
    falls below SPEARMAN_WARN_THRESHOLD.  A poor score means the extracted
    features are not correlated with degradation; training should be halted.

    Args:
        df_gt:      DataFrame with AGG_FEATURE_COLS and 'hi_target'.
        bearing_id: Identifier used in plot titles and filenames.
        output_dir: Directory to save the validation plot.

    Returns:
        Median absolute Spearman rho across all 28 features.
    """
    hi     = df_gt["hi_target"].values
    time   = df_gt["Time_Index"].values

    rho_records = []
    for feat in AGG_FEATURE_COLS:
        if feat not in df_gt.columns:
            continue
        rho, pval = scipy_stats.spearmanr(df_gt[feat].values, hi)
        rho_records.append({"feature": feat, "rho": rho, "pval": pval})

    rho_df = pd.DataFrame(rho_records).sort_values("rho", key=abs, ascending=False)
    median_abs_rho = float(rho_df["rho"].abs().median())

    print(f"\n  [Phase 4 — {bearing_id}] Spearman Rank Correlation vs HI Target")
    print(f"  {'Feature':<35} {'rho':>8}  {'p-value':>10}")
    print("  " + "-" * 58)
    for _, r in rho_df.iterrows():
        print(f"  {r['feature']:<35} {r['rho']:>8.4f}  {r['pval']:>10.4e}")
    print(f"\n  Median |rho| = {median_abs_rho:.4f}")

    if median_abs_rho < SPEARMAN_WARN_THRESHOLD:
        warnings.warn(
            f"\n{'='*65}\n"
            f"  VALIDATION WARNING — {bearing_id}\n"
            f"  Median |Spearman rho| = {median_abs_rho:.4f} "
            f"< threshold {SPEARMAN_WARN_THRESHOLD}.\n"
            f"  Visual correlation and Spearman score are poor.\n"
            f"  The model will likely fail to learn meaningful patterns.\n"
            f"  ACTION REQUIRED: Halt training and audit feature engineering.\n"
            f"{'='*65}",
            UserWarning,
            stacklevel=2,
        )

    # --- Plot ---
    n_feats  = len(AGG_FEATURE_COLS)
    n_cols   = 4
    n_rows   = (n_feats + n_cols - 1) // n_cols + 1  # +1 row for HI overlay

    fig = plt.figure(figsize=(n_cols * 5, n_rows * 3))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    # Top full-width: HI target curve
    ax_hi = fig.add_subplot(gs[0, :])
    ax_hi.plot(time, hi, color="#c0392b", linewidth=2, label="HI Target")
    ax_hi.set_title(
        f"{bearing_id} — Health Index Target  "
        f"(CUSUM CP = {df_gt['cusum_change_point'].iloc[0]})"
    )
    ax_hi.set_xlabel("Time Index (sub-minute steps)")
    ax_hi.set_ylabel("HI")
    ax_hi.legend()
    ax_hi.grid(True, linestyle="--", alpha=0.5)

    # Remaining rows: one subplot per feature
    for feat_idx, feat in enumerate(AGG_FEATURE_COLS):
        row = (feat_idx // n_cols) + 1
        col = feat_idx % n_cols
        ax  = fig.add_subplot(gs[row, col])

        rho_row = rho_df[rho_df["feature"] == feat]
        rho_val = rho_row["rho"].values[0] if len(rho_row) > 0 else float("nan")

        ax.plot(time, df_gt[feat].values, linewidth=1.0, alpha=0.85)
        ax.set_title(f"{feat}\nrho={rho_val:.3f}", fontsize=8)
        ax.set_xlabel("Time", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(
        f"Feature-HI Validation — {bearing_id}  "
        f"(Median |rho| = {median_abs_rho:.3f})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"validation_{bearing_id}.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Phase 4] Validation plot saved: {plot_path}")

    # Save Spearman table as CSV alongside the plot
    csv_path = os.path.join(output_dir, f"spearman_{bearing_id}.csv")
    rho_df.to_csv(csv_path, index=False)

    return median_abs_rho


# ============================================================================
# PHASE 5 — PER-BEARING MINMAX SCALING + CLIP
# ============================================================================

def scale_and_clip(
    df_gt: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply per-bearing MinMaxScaler to the 28 aggregated features, then clip.

    If scaler is None (training mode), fit a new scaler on df_gt and return it.
    If scaler is provided (validation mode), transform only — do not refit.

    Clipping to [-5.0, 5.0] is applied immediately after transform to prevent
    sigmoid/tanh saturation caused by extreme out-of-distribution values.

    Args:
        df_gt:  DataFrame with AGG_FEATURE_COLS and label columns.
        scaler: Pre-fitted MinMaxScaler (for val set) or None (for train set).

    Returns:
        Tuple of (scaled+clipped DataFrame, fitted MinMaxScaler).
    """
    df_scaled = df_gt.copy()

    present_feats = [c for c in AGG_FEATURE_COLS if c in df_scaled.columns]

    if scaler is None:
        scaler = MinMaxScaler()
        df_scaled[present_feats] = scaler.fit_transform(
            df_scaled[present_feats]
        )
    else:
        df_scaled[present_feats] = scaler.transform(
            df_scaled[present_feats]
        )

    # Critical: clip immediately after scaling.
    df_scaled[present_feats] = df_scaled[present_feats].clip(
        lower=CLIP_LOW, upper=CLIP_HIGH
    )

    return df_scaled, scaler


# ============================================================================
# PHASE 6 — SLIDING WINDOWS + NATIVE 3D .npy SAVING
# ============================================================================

def create_3d_windows(
    df_scaled: pd.DataFrame,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D sliding windows from scaled aggregated features.

    Args:
        df_scaled:   Scaled DataFrame with AGG_FEATURE_COLS and 'hi_target'.
        window_size: Number of consecutive sub-minute timesteps per window.

    Returns:
        Tuple (X, y) where:
            X shape: (N_windows, window_size, 28)
            y shape: (N_windows,)  — HI value of the last timestep in window.
    """
    present_feats = [c for c in AGG_FEATURE_COLS if c in df_scaled.columns]
    feature_matrix = df_scaled[present_feats].values.astype(np.float32)
    hi_values      = df_scaled["hi_target"].values.astype(np.float32)

    n_samples = len(feature_matrix)
    if n_samples <= window_size:
        return np.empty((0, window_size, len(present_feats)), dtype=np.float32), \
               np.empty((0,), dtype=np.float32)

    n_windows = n_samples - window_size + 1
    X = np.lib.stride_tricks.sliding_window_view(
        feature_matrix, window_shape=window_size, axis=0
    )  # Shape: (n_windows, n_features, window_size)
    X = X.transpose(0, 2, 1)  # -> (n_windows, window_size, n_features)
    y = hi_values[window_size - 1:]

    assert X.shape == (n_windows, window_size, len(present_feats)), \
        f"Unexpected X shape: {X.shape}"
    assert y.shape == (n_windows,), f"Unexpected y shape: {y.shape}"

    return X.astype(np.float32), y.astype(np.float32)


def save_npy(array: np.ndarray, path: str) -> None:
    """Save a numpy array as a .npy file, creating parent directories."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)
    print(f"  [Saved] {path}  shape={array.shape}")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def discover_bearing_dirs(root: str) -> Dict[str, str]:
    """
    Walk the dataset root and map bearing IDs to their CSV directories.

    Expected layout:
        root/
            35Hz/
                Bearing1_1/  <- contains acc_xxxxx.csv files
                Bearing1_2/
                ...
            37.5Hz/
                Bearing2_1/
                ...
            40Hz/
                Bearing3_1/
                ...

    Returns:
        Dict mapping bearing_id (e.g. "Bearing1_3") to its directory path.
    """
    mapping: Dict[str, str] = {}
    condition_map = {"35": 1, "37": 2, "40": 3}

    for cond_dir in sorted(os.listdir(root)):
        cond_path = os.path.join(root, cond_dir)
        if not os.path.isdir(cond_path):
            continue
        c_idx = None
        for key, val in condition_map.items():
            if key in cond_dir:
                c_idx = val
                break
        if c_idx is None:
            continue
        for bearing_dir in sorted(os.listdir(cond_path)):
            b_path = os.path.join(cond_path, bearing_dir)
            if not os.path.isdir(b_path):
                continue
            # Extract bearing number from folder name (e.g. "Bearing1" -> "1")
            digits = "".join(filter(str.isdigit, bearing_dir))
            b_num  = digits[-1] if digits else "1"
            b_id   = f"Bearing{c_idx}_{b_num}"
            mapping[b_id] = b_path

    return mapping


def run_pipeline() -> None:
    """
    Main preprocessing pipeline orchestrator.

    Execution sequence:
        1. Discover bearing directories.
        2. Phase 1 + 2: intra-file feature extraction and aggregation.
        3. Phase 3: causal CUSUM + piecewise-linear HI target construction.
        4. Phase 4: mandatory Spearman visual validation (with halt warning).
        5. Phase 5: per-bearing MinMax scaling + clip for each fold.
        6. Phase 6: sliding windows, 3D assertion, native .npy saving.
    """
    bearing_dirs = discover_bearing_dirs(INPUT_PATH)
    if not bearing_dirs:
        print(f"ERROR: No bearing directories found under {INPUT_PATH}")
        return

    print(f"\nFound {len(bearing_dirs)} bearing directories.")

    # ------------------------------------------------------------------
    # Phases 1 & 2: Extract + Aggregate per bearing
    # ------------------------------------------------------------------
    print("\n--- Phase 1 + 2: Feature Extraction and Intra-File Aggregation ---")
    all_aggregated: Dict[str, pd.DataFrame] = {}

    for b_id, b_dir in tqdm(bearing_dirs.items(), desc="Bearings"):
        df_agg = run_phase1_and_phase2(b_dir, b_id)
        if df_agg is not None and not df_agg.empty:
            all_aggregated[b_id] = df_agg

    if not all_aggregated:
        print("ERROR: No aggregated data produced. Check INPUT_PATH and CSV format.")
        return

    # ------------------------------------------------------------------
    # Phase 3: Causal ground truth construction
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Causal CUSUM + HI Target Construction ---")
    all_labeled: Dict[str, pd.DataFrame] = {}
    all_metadata = []

    for b_id, df_agg in all_aggregated.items():
        df_gt, meta = construct_causal_hi(df_agg)
        all_labeled[b_id] = df_gt
        all_metadata.append(meta)
        print(
            f"  {b_id}: CP index={meta['Change_Point_Index']}, "
            f"CP minute={meta['Change_Point_Minute']:.1f}, "
            f"HI=[{meta['HI_Min']:.3f}, {meta['HI_Max']:.3f}]"
        )

    meta_df = pd.DataFrame(all_metadata)
    meta_path = os.path.join(OUTPUT_PATH, "bearing_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"\nMetadata saved: {meta_path}")

    # ------------------------------------------------------------------
    # Phase 4: Mandatory visual validation (Spearman)
    # ------------------------------------------------------------------
    print("\n--- Phase 4: Mandatory Visual Validation (Spearman Correlation) ---")
    validation_dir = os.path.join(OUTPUT_PATH, "validation_plots")
    validation_summary = []

    for b_id, df_gt in all_labeled.items():
        median_rho = validate_and_plot(df_gt, b_id, validation_dir)
        validation_summary.append({"bearing": b_id, "median_abs_rho": median_rho})

    val_df = pd.DataFrame(validation_summary)
    val_path = os.path.join(validation_dir, "spearman_summary.csv")
    val_df.to_csv(val_path, index=False)
    print(f"\nSpearman summary saved: {val_path}")

    overall_poor = val_df[val_df["median_abs_rho"] < SPEARMAN_WARN_THRESHOLD]
    if not overall_poor.empty:
        print(
            "\n  *** HALT RECOMMENDATION ***\n"
            "  The following bearings have poor feature-HI correlation:\n"
            f"{overall_poor.to_string(index=False)}\n"
            "  Investigate feature engineering before proceeding to training."
        )

    # ------------------------------------------------------------------
    # Phases 5 & 6: Scaling, windowing, 3D .npy saving per fold
    # ------------------------------------------------------------------
    print("\n--- Phase 5 + 6: Scaling, Windowing, and 3D .npy Saving ---")

    for ws in WINDOW_SIZES:
        print(f"\n  Window size = {ws} ({ws * 0.5:.1f} min physical context)")
        ws_dir = os.path.join(OUTPUT_PATH, f"ws_{ws}")
        os.makedirs(ws_dir, exist_ok=True)

        for fold_idx, fold in enumerate(FOLDS, start=1):
            train_ids = fold["train"]
            val_ids   = fold["val"]

            # --- Scale train set per-bearing (fit on train, transform val) ---
            train_X_parts, train_y_parts = [], []
            val_X_parts,   val_y_parts   = [], []

            for b_id in train_ids:
                if b_id not in all_labeled:
                    continue
                df_scaled, fitted_scaler = scale_and_clip(
                    all_labeled[b_id], scaler=None
                )
                X, y = create_3d_windows(df_scaled, ws)
                if X.shape[0] > 0:
                    train_X_parts.append(X)
                    train_y_parts.append(y)

            for b_id in val_ids:
                if b_id not in all_labeled:
                    continue
                # For val: use each bearing's own scaler to avoid data leakage
                df_scaled_val, _ = scale_and_clip(
                    all_labeled[b_id], scaler=None
                )
                X_v, y_v = create_3d_windows(df_scaled_val, ws)
                if X_v.shape[0] > 0:
                    val_X_parts.append(X_v)
                    val_y_parts.append(y_v)

            if not train_X_parts or not val_X_parts:
                print(f"  [WARNING] Fold {fold_idx} WS={ws}: empty split. Skipping.")
                continue

            X_train = np.concatenate(train_X_parts, axis=0)
            y_train = np.concatenate(train_y_parts, axis=0)
            X_val   = np.concatenate(val_X_parts,   axis=0)
            y_val   = np.concatenate(val_y_parts,   axis=0)

            # Shuffle training set
            rng  = np.random.default_rng(seed=42)
            perm = rng.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]

            assert X_train.ndim == 3, f"X_train must be 3D, got {X_train.ndim}D"
            assert X_val.ndim   == 3, f"X_val must be 3D, got {X_val.ndim}D"
            assert X_train.max() <= CLIP_HIGH + 1e-3, "Clipping violated on train"
            assert X_val.max()   <= CLIP_HIGH + 1e-3, "Clipping violated on val"

            fold_dir = os.path.join(ws_dir, f"fold_{fold_idx}")
            save_npy(X_train, os.path.join(fold_dir, "X_train.npy"))
            save_npy(y_train, os.path.join(fold_dir, "y_train.npy"))
            save_npy(X_val,   os.path.join(fold_dir, "X_val.npy"))
            save_npy(y_val,   os.path.join(fold_dir, "y_val.npy"))

            print(
                f"  Fold {fold_idx} WS={ws}: "
                f"train={X_train.shape}, val={X_val.shape}"
            )

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETE — all .npy files written to:")
    print(f"  {OUTPUT_PATH}")
    print("=" * 65)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_pipeline()
