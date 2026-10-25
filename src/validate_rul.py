
"""Validate the RUL workflow for NASA C-MAPSS FD001-FD004.

Run from C:\\ml_package using:
    python -m EngineLife.src.validate_rul
"""

from pathlib import Path

from .data_loader import load_subset
from .rul import (
    compute_rul_train,
    compute_rul_test,
    plot_rul_distribution,
    generate_rul_summary,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "rul"
    output_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["FD001", "FD002", "FD003", "FD004"]

    for subset in subsets:
        print(f"\nProcessing {subset}...")

        # Load training data, test data, and official test RUL references.
        train_df, test_df, rul_ref = load_subset(subset)

        # ----- Training data -----
        train_df = compute_rul_train(train_df)

        plot_rul_distribution(
            train_df,
            f"{subset}_train",
            output_dir,
        )

        generate_rul_summary(
            train_df,
            output_dir / f"{subset}_train_summary.csv",
        )

        # ----- Test data -----
        test_df = compute_rul_test(test_df, rul_ref)

        plot_rul_distribution(
            test_df,
            f"{subset}_test",
            output_dir,
        )

        generate_rul_summary(
            test_df,
            output_dir / f"{subset}_test_summary.csv",
        )

        # ----- Sanity check -----
        first_engine = test_df["unit_id"].iloc[0]
        engine_rows = test_df[
            test_df["unit_id"] == first_engine
        ]

        print(
            f"{subset} - Engine {first_engine}: "
            f"first RUL={engine_rows['RUL'].iloc[0]}, "
            f"last RUL={engine_rows['RUL'].iloc[-1]}"
        )

    print("\nRUL validation completed. Outputs saved to: outputs/rul")


if __name__ == "__main__":
    main()
