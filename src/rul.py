
"""
EngineLife - Remaining Useful Life (RUL) computation and visualisation
for the NASA C-MAPSS turbofan degradation dataset.

Functions:
    compute_rul_train()       - Calculate RUL for training engines.
    compute_rul_test()        - Calculate RUL for test engines using official
                                reference RUL values.
    plot_rul_distribution()  - Save an RUL distribution plot.
    generate_rul_summary()   - Save RUL summary statistics as CSV or JSON.
    clip_rul()               - Clip RUL to a specified maximum value.

Supports either 'unit_id' or 'unit' as the engine ID column.
Original DataFrames are never modified.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DEFAULT_PLOT_DPI = 150
DEFAULT_FIGSIZE = (10, 6)


def _get_engine_column(df: pd.DataFrame) -> str:
    """Return the engine ID column used by the DataFrame."""

    if "unit_id" in df.columns:
        return "unit_id"

    if "unit" in df.columns:
        return "unit"

    raise ValueError(
        "DataFrame must contain an engine ID column named "
        "'unit_id' or 'unit'."
    )


def _validate_cycle_column(df: pd.DataFrame) -> str:
    """Validate required columns and return the engine ID column."""

    engine_col = _get_engine_column(df)

    if "cycle" not in df.columns:
        raise ValueError("DataFrame must contain a 'cycle' column.")

    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    if df[[engine_col, "cycle"]].isnull().any().any():
        raise ValueError(
            "Engine IDs and cycle values must not contain missing values."
        )

    if not pd.api.types.is_numeric_dtype(df["cycle"]):
        raise ValueError("'cycle' must contain numeric values.")

    if (df["cycle"] < 0).any():
        raise ValueError("'cycle' values cannot be negative.")

    return engine_col


def _ensure_output_dir(out_dir: Path) -> None:
    """Create an output directory if it does not exist."""

    out_dir.mkdir(parents=True, exist_ok=True)


def compute_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate row-level RUL for training data.

    Formula:
        RUL = maximum observed cycle for the engine - current cycle

    The final observed training cycle of each engine has RUL = 0.
    """

    engine_col = _validate_cycle_column(df)
    result = df.copy()

    max_cycle = result.groupby(engine_col)["cycle"].transform("max")
    result["RUL"] = (max_cycle - result["cycle"]).astype(int)

    return result


def compute_rul_test(
    df: pd.DataFrame,
    rul_ref: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate row-level RUL for test data using NASA reference RUL values.

    Formula:
        RUL = last observed cycle + reference RUL - current cycle

    The reference values must correspond to the test engines in the same
    order as their first appearance in the test DataFrame.
    """

    engine_col = _validate_cycle_column(df)

    if rul_ref is None:
        raise ValueError("Official reference RUL values must be provided.")

    references = np.asarray(rul_ref).reshape(-1)

    if len(references) == 0:
        raise ValueError("Reference RUL array cannot be empty.")

    if not np.issubdtype(references.dtype, np.number):
        raise ValueError("Reference RUL values must be numeric.")

    if not np.isfinite(references).all():
        raise ValueError("Reference RUL values must be finite.")

    if (references < 0).any():
        raise ValueError("Reference RUL values cannot be negative.")

    if not np.equal(references, np.floor(references)).all():
        raise ValueError("Reference RUL values must be integers.")

    result = df.copy()

    # Preserve the engine order in which IDs first appear.
    engine_ids = result[engine_col].drop_duplicates().tolist()

    if len(engine_ids) != len(references):
        raise ValueError(
            f"Mismatch: found {len(engine_ids)} test engines, "
            f"but received {len(references)} reference RUL values."
        )

    # Associate each test engine with its official reference RUL.
    engine_to_ref = dict(zip(engine_ids, references.astype(int)))

    # Last cycle observed for each engine in the test data.
    last_cycle = result.groupby(engine_col)["cycle"].transform("max")

    # Estimate RUL at every observed test cycle.
    result["RUL"] = (
        last_cycle
        + result[engine_col].map(engine_to_ref)
        - result["cycle"]
    ).astype(int)

    return result


def plot_rul_distribution(
    df: pd.DataFrame,
    subset_name: str,
    out_dir: str | Path,
) -> None:
    """Save a histogram and KDE plot of the RUL distribution."""

    if "RUL" not in df.columns:
        raise ValueError(
            "DataFrame must contain an 'RUL' column before plotting."
        )

    if df["RUL"].dropna().empty:
        raise ValueError("Cannot plot RUL because all values are missing.")

    output_dir = Path(out_dir)
    _ensure_output_dir(output_dir)

    plt.figure(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_PLOT_DPI)

    try:
        sns.histplot(
            data=df,
            x="RUL",
            kde=df["RUL"].nunique() > 1,
            bins=30,
            color="#4A90E2",
        )

        plt.title(f"RUL Distribution - {subset_name}", fontsize=14)
        plt.xlabel("Remaining Useful Life (cycles)")
        plt.ylabel("Count")
        plt.tight_layout()

        output_file = output_dir / f"rul_distribution_{subset_name}.png"
        plt.savefig(output_file, dpi=DEFAULT_PLOT_DPI)
    finally:
        plt.close()


def generate_rul_summary(
    df: pd.DataFrame,
    out_path: str | Path,
) -> dict:
    """
    Generate summary statistics for the RUL column and save them as CSV or JSON.

    Returns the summary dictionary.
    """

    if "RUL" not in df.columns:
        raise ValueError(
            "DataFrame must contain an 'RUL' column to summarise."
        )

    rul = pd.to_numeric(df["RUL"], errors="raise").dropna()

    if rul.empty:
        raise ValueError("Cannot summarise an empty RUL column.")

    summary = {
        "count": int(rul.count()),
        "mean": float(rul.mean()),
        "median": float(rul.median()),
        "std": float(rul.std()) if len(rul) > 1 else None,
        "min": float(rul.min()),
        "25%": float(rul.quantile(0.25)),
        "75%": float(rul.quantile(0.75)),
        "max": float(rul.max()),
    }

    output_path = Path(out_path)
    _ensure_output_dir(output_path.parent)

    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        pd.DataFrame([summary]).to_csv(output_path, index=False)

    elif suffix == ".json":
        output_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    else:
        raise ValueError(
            "Unsupported output extension. Use '.csv' or '.json'."
        )

    return summary


def clip_rul(
    df: pd.DataFrame,
    max_rul: int,
) -> pd.DataFrame:
    """Return a copy with RUL values capped at max_rul."""

    if "RUL" not in df.columns:
        raise ValueError(
            "DataFrame must contain an 'RUL' column to clip."
        )

    if not isinstance(max_rul, (int, np.integer)) or isinstance(
        max_rul, (bool, np.bool_)
    ):
        raise ValueError("max_rul must be a positive integer.")

    if max_rul <= 0:
        raise ValueError("max_rul must be a positive integer.")

    result = df.copy()
    result["RUL"] = result["RUL"].clip(upper=max_rul)

    return result


def _run_sanity_tests() -> None:
    """Run small deterministic checks of the RUL calculations."""

    # Training data: supports the unit_id column used by EngineLife.
    train = pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "sensor_1": [10, 11, 12, 20, 21],
    })

    train_result = compute_rul_train(train)

    assert train_result["RUL"].tolist() == [2, 1, 0, 1, 0]
    assert "RUL" not in train.columns  # Original DataFrame is unchanged.

    # Also support datasets using the unit column.
    train_unit = pd.DataFrame({
        "unit": [1, 1, 2, 2],
        "cycle": [1, 3, 1, 4],
    })

    result_unit = compute_rul_train(train_unit)
    assert result_unit["RUL"].tolist() == [2, 0, 3, 0]

    # Test data: engine 1 ends at cycle 3 with reference RUL 5;
    # engine 2 ends at cycle 2 with reference RUL 10.
    test = pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
    })

    test_result = compute_rul_test(test, np.array([5, 10]))

    assert test_result["RUL"].tolist() == [7, 6, 5, 11, 10]

    # A reference-count mismatch must be rejected.
    try:
        compute_rul_test(test, np.array([5]))
    except ValueError:
        pass
    else:
        raise AssertionError("Reference-count mismatch was not detected.")

    print("All rul.py sanity tests passed.")



def _run_dataset_pipeline() -> None:
    """
    Generate RUL plots and summary CSV files for FD001-FD004.

    Run from C:\\ml_package using:
        python -m EngineLife.src.rul
    """
    from .data_loader import load_subset

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "rul"
    output_dir.mkdir(parents=True, exist_ok=True)

    for subset in ("FD001", "FD002", "FD003", "FD004"):
        print(f"\nProcessing {subset}...")

        # Load all three items using the actual data_loader interface.
        train_df, test_df, rul_ref = load_subset(subset)

        # Calculate training RUL.
        train_df = compute_rul_train(train_df)

        plot_rul_distribution(
            train_df,
            f"{subset}_train",
            output_dir,
        )

        train_summary = generate_rul_summary(
            train_df,
            output_dir / f"{subset}_train_summary.csv",
        )

        # Calculate test RUL using the official reference values.
        test_df = compute_rul_test(test_df, rul_ref)

        plot_rul_distribution(
            test_df,
            f"{subset}_test",
            output_dir,
        )

        test_summary = generate_rul_summary(
            test_df,
            output_dir / f"{subset}_test_summary.csv",
        )

        print(f"{subset} training RUL summary: {train_summary}")
        print(f"{subset} test RUL summary: {test_summary}")

    print(f"\nRUL outputs saved to: {output_dir}")


if __name__ == "__main__":
    _run_sanity_tests()
    _run_dataset_pipeline()


if __name__ == "__main__":
    _run_sanity_tests()
    _run_dataset_pipeline()
