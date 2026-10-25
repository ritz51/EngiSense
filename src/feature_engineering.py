"""Feature engineering, data preparation, and sequence generation for EngiSense.

Ensures strict separation between training and test sets, engine-level splits
to prevent data leakage, and identical transformations for training and inference.
"""

from __future__ import annotations

import pathlib
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import load_subset
from .rul import compute_rul_train, compute_rul_test, clip_rul


# Known non-informative constant sensors for FD001 & FD003
CONSTANT_FEATURES_MAP = {
    "FD001": ["setting_3", "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"],
    "FD002": [],
    "FD003": ["setting_3", "sensor_1", "sensor_5", "sensor_16", "sensor_18", "sensor_19"],
    "FD004": [],
}


def get_feature_columns(df: pd.DataFrame, subset: str = "FD001", drop_constants: bool = True) -> List[str]:
    """Return informative feature columns for modeling, excluding engine metadata and constant sensors."""
    metadata_cols = {"unit_id", "unit", "cycle", "RUL"}
    all_features = [c for c in df.columns if c not in metadata_cols]

    if drop_constants:
        constants = CONSTANT_FEATURES_MAP.get(subset.upper(), [])
        # Also drop any columns that have zero variance in this dataset
        zero_var = [c for c in all_features if df[c].nunique(dropna=False) <= 1]
        drop_set = set(constants).union(zero_var)
        return [c for c in all_features if c not in drop_set]

    return all_features


def split_engine_ids(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split engines into training and validation groups to prevent data leakage."""
    engine_col = "unit_id" if "unit_id" in df.columns else "unit"
    engine_ids = df[engine_col].drop_duplicates().to_numpy(copy=True)

    rng = np.random.default_rng(seed)
    rng.shuffle(engine_ids)

    split = int(len(engine_ids) * (1.0 - val_frac))
    if split <= 0 or split >= len(engine_ids):
        raise ValueError("val_frac results in empty train or validation set.")

    return engine_ids[:split].tolist(), engine_ids[split:].tolist()


# ---------------------------------------------------------------------------
# Tabular Feature Engineering for XGBoost
# ---------------------------------------------------------------------------

def extract_tabular_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    rolling_windows: List[int] = (5, 10),
) -> pd.DataFrame:
    """Generate tabular features including rolling statistics per engine trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with unit_id, cycle, and base features.
    feature_cols : List[str]
        List of sensor and setting columns to extract features from.
    rolling_windows : List[int]
        Window sizes for rolling mean and standard deviation.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with raw and rolling features.
    """
    engine_col = "unit_id" if "unit_id" in df.columns else "unit"
    df_sorted = df.sort_values([engine_col, "cycle"]).copy()

    feature_dfs = [df_sorted[[engine_col, "cycle"] + [c for c in ["RUL"] if c in df_sorted.columns]]]
    # Add raw base features
    feature_dfs.append(df_sorted[feature_cols])

    # Add rolling statistics per engine
    grouped = df_sorted.groupby(engine_col)[feature_cols]

    for w in rolling_windows:
        mean_df = grouped.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        mean_df.columns = [f"{c}_roll_mean_{w}" for c in feature_cols]
        feature_dfs.append(mean_df)

        std_df = grouped.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
        std_df.columns = [f"{c}_roll_std_{w}" for c in feature_cols]
        feature_dfs.append(std_df)

    enriched = pd.concat(feature_dfs, axis=1)
    return enriched


# ---------------------------------------------------------------------------
# Sequence Generation for LSTM & GRU
# ---------------------------------------------------------------------------

def pad_or_truncate_sequence(sensor_matrix: np.ndarray, seq_len: int) -> np.ndarray:
    """Ensure a sensor matrix has exactly seq_len rows using edge-padding if needed."""
    n_rows, n_cols = sensor_matrix.shape
    if n_rows >= seq_len:
        return sensor_matrix[-seq_len:]

    # Repeat the first available row at the beginning (edge padding)
    pad_len = seq_len - n_rows
    padding = np.repeat(sensor_matrix[:1, :], pad_len, axis=0)
    return np.vstack([padding, sensor_matrix])


def create_sequences_for_engine(
    group: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate all sliding windows of length seq_len for a single engine.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        seqs: shape (N, seq_len, len(feature_cols))
        targets: shape (N,)
    """
    group_sorted = group.sort_values("cycle")
    sensor_vals = group_sorted[feature_cols].to_numpy(dtype=np.float32)
    rul_vals = group_sorted["RUL"].to_numpy(dtype=np.float32) if "RUL" in group_sorted.columns else None

    n_rows = len(group_sorted)
    if n_rows < seq_len:
        # Pad shorter engine
        seq = pad_or_truncate_sequence(sensor_vals, seq_len)[np.newaxis, ...]
        target = np.array([rul_vals[-1]], dtype=np.float32) if rul_vals is not None else np.array([0.0])
        return seq, target

    seqs = []
    targets = []
    for start in range(n_rows - seq_len + 1):
        end = start + seq_len
        seqs.append(sensor_vals[start:end])
        if rul_vals is not None:
            targets.append(rul_vals[end - 1])

    seq_arr = np.stack(seqs, axis=0)
    target_arr = np.array(targets, dtype=np.float32) if rul_vals is not None else np.zeros(len(seqs), dtype=np.float32)
    return seq_arr, target_arr


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Build full sliding-window sequence array and scale features consistently.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing engine sensor observations.
    feature_cols : List[str]
        List of features to include in the sequence.
    seq_len : int
        Window length.
    scaler : StandardScaler, optional
        Pre-fitted scaler to apply.
    fit_scaler : bool
        If True, fit a new scaler on the raw features.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        (sequences, targets, scaler)
    """
    engine_col = "unit_id" if "unit_id" in df.columns else "unit"
    all_seqs = []
    all_targets = []

    for _, group in df.groupby(engine_col):
        seqs, targets = create_sequences_for_engine(group, feature_cols, seq_len)
        all_seqs.append(seqs)
        all_targets.append(targets)

    X_raw = np.concatenate(all_seqs, axis=0)  # (N, seq_len, F)
    y = np.concatenate(all_targets, axis=0)   # (N,)

    # Scale across feature dimension
    N, T, F = X_raw.shape
    flattened = X_raw.reshape(-1, F)

    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        scaler.fit(flattened)

    scaled_flattened = scaler.transform(flattened)
    X_scaled = scaled_flattened.reshape(N, T, F).astype(np.float32)

    return X_scaled, y, scaler


def extract_last_sequence_per_engine(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    scaler: StandardScaler,
) -> np.ndarray:
    """Extract the single final sequence window for each test engine.

    Returns
    -------
    np.ndarray
        Array of shape (n_engines, seq_len, n_features), ordered by engine ID.
    """
    engine_col = "unit_id" if "unit_id" in df.columns else "unit"
    engine_ids = df[engine_col].drop_duplicates().tolist()

    last_windows = []
    for eng_id in engine_ids:
        group = df[df[engine_col] == eng_id].sort_values("cycle")
        sensor_vals = group[feature_cols].to_numpy(dtype=np.float32)
        window = pad_or_truncate_sequence(sensor_vals, seq_len)
        last_windows.append(window)

    X_raw = np.stack(last_windows, axis=0)  # (n_engines, seq_len, F)
    N, T, F = X_raw.shape
    scaled = scaler.transform(X_raw.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
    return scaled
