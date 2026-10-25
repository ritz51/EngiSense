# src/data_loader.py
"""Utility functions to load NASA C-MAPSS dataset files.

The raw files are space‑separated text files that may contain a varying
number of columns.  Each file has no header.  The expected column layout is

    0: unit_id            (engine identifier)
    1: cycle              (operating cycle)
    2‑4: setting_1‑3      (operating‑setting variables)
    5‑25: sensor_1‑21    (sensor measurements)

The loader does **not** modify the original files – it only reads them into
a :class:`pandas.DataFrame` and assigns sensible column names.

Functions
---------
- :func:`load_dataset(subset, kind, base_path)` – Load a specific training or
  test file for a given subset (e.g. ``FD001``).  Returns a DataFrame.
- :func:`load_rul_file(subset, base_path)` – Load the RUL reference file for the
  test set.
- :func:`assign_column_names(df)` – Attach column names based on the number of
  columns that were actually read.
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_PATH = PROJECT_ROOT

# Expected column count for the standard C‑MAPSS files
EXPECTED_COLS = 26

def assign_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add column names to ``df``.

    The function works for the standard 26‑column layout as well as for files
    that happen to contain fewer columns (e.g., some sensor columns are omitted
    in certain publications).  Missing sensor columns are simply omitted from
    the DataFrame; they will be handled later during validation.
    """
    col_count = df.shape[1]
    # Base columns always present
    base_cols = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    # Add as many sensor columns as we have remaining columns
    sensor_cols = [f"sensor_{i}" for i in range(1, col_count - len(base_cols) + 1)]
    df.columns = base_cols + sensor_cols
    return df


def load_dataset(subset: str, kind: str, base_path: pathlib.Path | str = BASE_PATH) -> pd.DataFrame:
    """Load *train* or *test* data for a given FD subset.

    Parameters
    ----------
    subset: str
        One of ``"FD001"``, ``"FD002"``, ``"FD003"``, ``"FD004"``.
    kind: str
        Either ``"train"`` or ``"test"``.
    base_path: Path or str, optional
        Root directory that contains the ``archive`` folder.

    Returns
    -------
    pd.DataFrame
        The raw data with column names assigned.
    """
    if kind not in {"train", "test"}:
        raise ValueError("kind must be 'train' or 'test'")
    file_path = pathlib.Path(base_path) / "archive" / f"{kind}_{subset}.txt"
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    # ``sep='\s+'`` handles any amount of whitespace. ``engine='python'``
    # avoids pandas trying to guess the number of columns from a fixed-width
    # parser.
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = assign_column_names(df)
    return df


def load_rul_file(subset: str, base_path: pathlib.Path | str = BASE_PATH) -> pd.Series:
    """Load the RUL reference file for the test set of a subset.

    The file contains a single integer per line – the remaining useful life for
    the **last** cycle of each test engine.
    """
    file_path = pathlib.Path(base_path) / "archive" / f"RUL_{subset}.txt"
    if not file_path.is_file():
        raise FileNotFoundError(f"RUL file not found: {file_path}")
    # ``header=None`` forces a single‑column DataFrame; ``squeeze=True`` turns it
    # into a Series of ints.
    rul_series = pd.read_csv(file_path, header=None).iloc[:, 0]
    # Ensure integer type
    rul_series = rul_series.astype(int)
    return rul_series


def load_subset(subset: str, base_path: pathlib.Path | str = BASE_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Convenience wrapper returning ``train_df, test_df, rul_series``.
    """
    train_df = load_dataset(subset, "train", base_path)
    test_df = load_dataset(subset, "test", base_path)
    rul_series = load_rul_file(subset, base_path)
    return train_df, test_df, rul_series

# When executed as a script, provide a quick sanity check.
if __name__ == "__main__":
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        train, test, rul = load_subset(sub)
        print(f"{sub}: train rows={len(train):,}, test rows={len(test):,}, RUL entries={len(rul)}")
