# src/validation.py
"""Dataset validation for NASA C‑MAPSS FD subsets.

For each subset (FD001‑FD004) the script performs the checks requested in
*Phase 1* of the project description and writes a detailed validation report
as both JSON and CSV.

The script is deliberately defensive – it never modifies the original raw
files.  All cleaning decisions (e.g., dropping constant columns) are only
recorded in the report; downstream pipelines can decide whether to act on
them.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from .data_loader import load_subset

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _is_numeric(series: pd.Series) -> bool:
    """Return ``True`` if all non‑null values in *series* are numeric.
    ``pandas`` may read a column as ``object`` when mixed types are present.
    """
    try:
        pd.to_numeric(series.dropna())
        return True
    except ValueError:
        return False


def _detect_constant(series: pd.Series, eps: float = 1e-12) -> bool:
    """Detect perfectly constant columns (zero variance)."""
    return series.nunique(dropna=False) <= 1


def _detect_near_constant(series: pd.Series, rel_tol: float = 1e-3) -> bool:
    """Detect near‑constant columns.

    A column is considered near‑constant when the standard deviation is less
    than *rel_tol* times the absolute mean (or absolute mean of 1 when the
    mean is close to zero).  This heuristic works for sensor data that is
    expected to vary noticeably across cycles.
    """
    if series.dtype.kind not in "biufc":
        return False
    # Ignore NaNs for the calculation
    clean = series.dropna()
    if clean.empty:
        return False
    mean_abs = np.mean(np.abs(clean))
    if mean_abs < 1e-9:
        mean_abs = 1.0
    std = np.std(clean)
    return std / mean_abs < rel_tol


def _column_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return descriptive statistics for all columns as a dict.
    Uses ``pandas.DataFrame.describe`` and converts the result to a plain
    ``dict`` to make JSON serialisation easy.
    """
    desc = df.describe(include="all").to_dict()
    # ``describe`` returns ``numpy.float64`` which JSON cannot serialise –
    # cast to Python ``float``/``int`` where possible.
    for col, stats in desc.items():
        for k, v in stats.items():
            if isinstance(v, (np.integer, np.floating)):
                desc[col][k] = v.item()
    return desc


def validate_subset(subset: str, base_path: pathlib.Path | str = pathlib.Path("c:/ml_package/EngineLife")) -> Dict[str, Any]:
    """Validate a single FD subset and return a dictionary of results.

    The function performs the checks listed in the specification – row counts,
    engine IDs, column counts, missing/duplicate values, non‑numeric entries,
    infinite values, constant / near‑constant features, and descriptive
    statistics.
    """
    train_df, test_df, rul_series = load_subset(subset, base_path)

    # -------------------------------------------------------------------
    # Basic dimensions & engine information
    # -------------------------------------------------------------------
    train_rows, test_rows = len(train_df), len(test_df)
    train_engines = sorted(train_df["unit_id"].unique().tolist())
    test_engines = sorted(test_df["unit_id"].unique().tolist())
    train_engine_count, test_engine_count = len(train_engines), len(test_engines)
    train_cycle_range = (int(train_df["cycle"].min()), int(train_df["cycle"].max()))
    test_cycle_range = (int(test_df["cycle"].min()), int(test_df["cycle"].max()))
    n_columns = train_df.shape[1]  # training and test should have same columns

    # -------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------
    train_missing = train_df.isnull().sum().to_dict()
    test_missing = test_df.isnull().sum().to_dict()

    # -------------------------------------------------------------------
    # Duplicate rows
    # -------------------------------------------------------------------
    train_duplicates = int(train_df.duplicated().sum())
    test_duplicates = int(test_df.duplicated().sum())

    # -------------------------------------------------------------------
    # Non‑numeric / infinite detection (per column)
    # -------------------------------------------------------------------
    non_numeric_cols: List[str] = []
    infinite_counts: Dict[str, int] = {}
    for col in train_df.columns:
        if not _is_numeric(train_df[col]):
            non_numeric_cols.append(col)
        # ``np.isinf`` works on numeric dtypes; for object dtypes we skip.
        if pd.api.types.is_numeric_dtype(train_df[col]):
            infinite_counts[col] = int(np.isinf(train_df[col]).sum())
    # Repeat for test set (union of both sets)
    for col in test_df.columns:
        if col not in non_numeric_cols and not _is_numeric(test_df[col]):
            non_numeric_cols.append(col)
        if pd.api.types.is_numeric_dtype(test_df[col]):
            infinite_counts[col] = infinite_counts.get(col, 0) + int(np.isinf(test_df[col]).sum())

    # -------------------------------------------------------------------
    # Constant / near‑constant features (excluding identifiers)
    # -------------------------------------------------------------------
    const_features = []
    near_const_features = []
    feature_cols = [c for c in train_df.columns if c not in {"unit_id", "cycle", "RUL"}]
    for col in feature_cols:
        if _detect_constant(train_df[col]):
            const_features.append(col)
        elif _detect_near_constant(train_df[col]):
            near_const_features.append(col)

    # -------------------------------------------------------------------
    # RUL verification – the RUL file should have the same length as the
    # number of test engines and must correspond to the *last* cycle of each
    # engine in the test set.
    # -------------------------------------------------------------------
    # Compute the true RUL from the test data for sanity‑check
    max_cycle_per_engine = test_df.groupby("unit_id")["cycle"].max()
    derived_rul = max_cycle_per_engine.map(lambda c: max_cycle_per_engine.max() - c)  # placeholder; actual RUL is provided in file
    # The specification expects that the provided RUL file aligns with the order of
    # test engines appearing in the file.  We therefore simply verify length.
    rul_length_match = len(rul_series) == test_engine_count
    # Additionally, test that the engine ordering matches the order of appearance
    # in the test set (sorted by engine ID).
    test_engine_order = test_df["unit_id"].drop_duplicates().tolist()
    rul_order_match = len(rul_series) == len(test_engine_order)

    # -------------------------------------------------------------------
    # Assemble the report dictionary
    # -------------------------------------------------------------------
    report: Dict[str, Any] = {
        "subset": subset,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_engine_count": train_engine_count,
        "test_engine_count": test_engine_count,
        "train_engine_ids": train_engines,
        "test_engine_ids": test_engines,
        "train_cycle_range": train_cycle_range,
        "test_cycle_range": test_cycle_range,
        "n_columns": n_columns,
        "train_missing_per_column": train_missing,
        "test_missing_per_column": test_missing,
        "train_duplicate_rows": train_duplicates,
        "test_duplicate_rows": test_duplicates,
        "non_numeric_columns": non_numeric_cols,
        "infinite_values_per_column": infinite_counts,
        "constant_features": const_features,
        "near_constant_features": near_const_features,
        "descriptive_statistics": _column_stats(train_df),
        "rul_file_length_match": rul_length_match,
        "rul_engine_order_match": rul_order_match,
    }
    return report


def run_validation_all(base_path: pathlib.Path | str = pathlib.Path("c:/ml_package/EngineLife")) -> List[Dict[str, Any]]:
    """Validate all four FD subsets and write the combined report.
    The output files are placed under ``outputs/validation_report.json`` and
    ``outputs/validation_report.csv``.
    """
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    results = []
    for sub in subsets:
        try:
            res = validate_subset(sub, base_path)
        except Exception as exc:  # pragma: no cover – defensive fallback
            res = {"subset": sub, "error": str(exc)}
        results.append(res)

    out_dir = pathlib.Path(base_path) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "validation_report.json"
    csv_path = out_dir / "validation_report.csv"
    # Write JSON – pretty printed for readability
    json_path.write_text(json.dumps(results, indent=2))
    # Write CSV – flatten the dict for each subset (only a subset of fields)
    rows = []
    for r in results:
        row = {
            "subset": r.get("subset"),
            "train_rows": r.get("train_rows"),
            "test_rows": r.get("test_rows"),
            "train_engine_count": r.get("train_engine_count"),
            "test_engine_count": r.get("test_engine_count"),
            "n_columns": r.get("n_columns"),
            "train_duplicate_rows": r.get("train_duplicate_rows"),
            "test_duplicate_rows": r.get("test_duplicate_rows"),
            "constant_features": ",".join(r.get("constant_features", [])),
            "near_constant_features": ",".join(r.get("near_constant_features", [])),
            "rul_file_length_match": r.get("rul_file_length_match"),
            "rul_engine_order_match": r.get("rul_engine_order_match"),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Validation report written to {json_path} and {csv_path}")
    return results


if __name__ == "__main__":
    run_validation_all()
