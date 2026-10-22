
"""Statistical inference utilities for the EngineLife project.

Analyses:
1. Descriptive statistics
2. Engine lifetime comparisons across FD001-FD004
3. Sensor distribution comparisons using engine-level means
4. Sensor-RUL correlations
5. Sensor trends across operating cycles using mixed-effects models
6. Paired comparisons of model errors

Training RUL = engine maximum cycle - current cycle.

Lifetime and row-level RUL are kept separate. Test RUL must be
calculated using the official NASA reference RUL values.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


DEFAULT_OUTPUT_DIR = pathlib.Path(
    "C:/ml_package/EngineLife/outputs"
)


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def _engine_column(df: pd.DataFrame) -> str:
    """Support either unit_id or unit as the engine ID column."""
    if "unit_id" in df.columns:
        return "unit_id"
    if "unit" in df.columns:
        return "unit"
    raise ValueError("DataFrame must contain 'unit_id' or 'unit'.")


def _group_columns(df: pd.DataFrame) -> list[str]:
    """Identify engines uniquely, even after combining subsets."""
    engine_col = _engine_column(df)

    if "subset" in df.columns:
        return ["subset", engine_col]

    return [engine_col]


def _json_safe(value: Any) -> Any:
    """Convert values recursively into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (pd.Timestamp, pathlib.Path)):
        return str(value)

    if value is pd.NA or value is pd.NaT:
        return None

    return value


def _bh_adjust(p_values: list[float | None]) -> list[float | None]:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    adjusted: list[float | None] = [None] * len(p_values)

    valid = [
        (i, float(p))
        for i, p in enumerate(p_values)
        if p is not None and np.isfinite(p)
    ]

    if not valid:
        return adjusted

    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    running_min = 1.0

    for rank_index in range(m - 1, -1, -1):
        original_index, p_value = ordered[rank_index]
        rank = rank_index + 1
        running_min = min(running_min, p_value * m / rank)
        adjusted[original_index] = float(min(running_min, 1.0))

    return adjusted


def add_training_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate row-level RUL for complete training trajectories."""
    engine_col = _engine_column(df)

    if "cycle" not in df.columns:
        raise ValueError("DataFrame must contain a 'cycle' column.")

    result = df.copy()
    group_cols = _group_columns(result)

    max_cycle = result.groupby(
        group_cols, sort=False
    )["cycle"].transform("max")

    result["RUL"] = (max_cycle - result["cycle"]).astype(int)
    return result


def descriptive_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Return descriptive statistics for every numeric column."""
    output: dict[str, Any] = {}

    for column in df.columns:
        series = df[column]

        entry: dict[str, Any] = {
            "missing_count": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean() * 100),
            "n_distinct": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            entry.update({
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(ddof=1),
                "variance": series.var(ddof=1),
                "min": series.min(),
                "max": series.max(),
                "q1": series.quantile(0.25),
                "q3": series.quantile(0.75),
                "iqr": (
                    series.quantile(0.75)
                    - series.quantile(0.25)
                ),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
            })

        output[column] = _json_safe(entry)

    return output


# ---------------------------------------------------------------------
# A. Engine lifetime comparisons
# ---------------------------------------------------------------------

def lifetime_aggregation(
    train_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return one observed lifetime per engine and subset.

    Observed lifetime is the maximum recorded training cycle.
    It is not row-level remaining useful life.
    """
    rows = []

    for subset, df in train_dfs.items():
        engine_col = _engine_column(df)

        if "cycle" not in df.columns:
            raise ValueError(f"{subset} has no 'cycle' column.")

        max_cycles = df.groupby(engine_col)["cycle"].max()

        for engine_id, lifetime in max_cycles.items():
            rows.append({
                "subset": subset,
                "unit_id": int(engine_id),
                "lifetime": float(lifetime),
            })

    return pd.DataFrame(
        rows,
        columns=["subset", "unit_id", "lifetime"],
    )


def test_lifetime_differences(
    lifetimes: pd.DataFrame,
) -> dict[str, Any]:
    """Compare observed engine lifetimes across subsets.

    Uses ordinary one-way ANOVA when normality and equal-variance
    checks are acceptable; otherwise uses Kruskal-Wallis.
    """
    required = {"subset", "lifetime"}
    if not required.issubset(lifetimes.columns):
        raise ValueError(f"Required columns: {required}")

    clean = lifetimes[["subset", "lifetime"]].dropna()
    grouped = list(clean.groupby("subset", sort=True))
    names = [name for name, _ in grouped]
    groups = [
        group["lifetime"].to_numpy(dtype=float)
        for _, group in grouped
    ]

    if len(groups) < 2:
        return {
            "error": "At least two subsets are required for comparison."
        }

    if any(len(group) < 2 for group in groups):
        return {
            "error": "Each subset needs at least two engines."
        }

    normality = {}
    for name, group in zip(names, groups):
        # Shapiro-Wilk is not reliable for extremely large samples.
        sample = group
        if len(sample) > 5000:
            sample = np.random.default_rng(42).choice(
                sample, size=5000, replace=False
            )
        normality[str(name)] = float(stats.shapiro(sample).pvalue)

    levene_stat, levene_p = stats.levene(
        *groups, center="median"
    )

    all_normal = all(p > 0.05 for p in normality.values())
    equal_variance = bool(levene_p > 0.05)

    if all_normal and equal_variance:
        statistic, p_value = stats.f_oneway(*groups)
        test_name = "One-way ANOVA"

        grand_mean = clean["lifetime"].mean()
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2
            for g in groups
        )
        ss_total = sum(
            (x - grand_mean) ** 2 for g in groups for x in g
        )
        effect_size = (
            float(ss_between / ss_total)
            if ss_total > 0 else None
        )
        effect_name = "eta_squared"
    else:
        statistic, p_value = stats.kruskal(*groups)
        test_name = "Kruskal-Wallis"

        n = sum(len(g) for g in groups)
        k = len(groups)
        effect_size = (
            float((statistic - k + 1) / (n - k))
            if n > k else None
        )
        effect_name = "epsilon_squared"

    # Bootstrap confidence intervals for pairwise median differences.
    rng = np.random.default_rng(42)
    pairwise_ci = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = groups[i], groups[j]
            differences = []

            for _ in range(1000):
                sample_a = rng.choice(a, size=len(a), replace=True)
                sample_b = rng.choice(b, size=len(b), replace=True)
                differences.append(
                    float(np.median(sample_a) - np.median(sample_b))
                )

            lower, upper = np.percentile(
                differences, [2.5, 97.5]
            )
            pairwise_ci[f"{names[i]}_vs_{names[j]}"] = [
                float(lower), float(upper)
            ]

    p_value = float(p_value)

    return _json_safe({
        "research_question": "Observed engine lifetime differences",
        "group_sizes": {
            str(name): int(len(group))
            for name, group in zip(names, groups)
        },
        "normality_pvalues": normality,
        "levene": {
            "statistic": float(levene_stat),
            "pvalue": float(levene_p),
        },
        "chosen_test": test_name,
        "test_statistic": float(statistic),
        "p_value": p_value,
        "effect_size_name": effect_name,
        "effect_size": effect_size,
        "alpha": 0.05,
        "reject_null": bool(p_value < 0.05),
        "pairwise_median_difference_bootstrap_ci": pairwise_ci,
    })


# ---------------------------------------------------------------------
# B. Sensor distribution comparisons
# ---------------------------------------------------------------------

def sensor_distribution_tests(
    train_dfs: dict[str, pd.DataFrame],
    sensors: list[str],
) -> dict[str, Any]:
    """Compare subset sensor distributions using engine-level means.

    Each engine contributes one value per sensor, avoiding treating
    repeated cycles from the same engine as independent engines.
    """
    results = []

    for sensor in sensors:
        groups = []
        names = []

        for subset, df in train_dfs.items():
            if sensor not in df.columns:
                continue

            engine_col = _engine_column(df)
            engine_means = (
                df.groupby(engine_col)[sensor]
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )

            if len(engine_means) >= 2:
                groups.append(engine_means)
                names.append(subset)

        if len(groups) < 2:
            results.append({
                "sensor": sensor,
                "test": None,
                "p_raw": None,
                "note": "Insufficient data in at least two subsets.",
            })
            continue

        normality_p = []
        for group in groups:
            sample = group
            if len(sample) > 5000:
                sample = np.random.default_rng(42).choice(
                    sample, 5000, replace=False
                )
            normality_p.append(
                float(stats.shapiro(sample).pvalue)
            )

        levene_stat, levene_p = stats.levene(
            *groups, center="median"
        )

        if all(p > 0.05 for p in normality_p) and levene_p > 0.05:
            statistic, p_value = stats.f_oneway(*groups)
            test_name = "One-way ANOVA"

            combined = np.concatenate(groups)
            grand_mean = combined.mean()
            ss_between = sum(
                len(g) * (g.mean() - grand_mean) ** 2
                for g in groups
            )
            ss_total = np.sum((combined - grand_mean) ** 2)
            effect = (
                float(ss_between / ss_total)
                if ss_total > 0 else None
            )
            effect_name = "eta_squared"
        else:
            statistic, p_value = stats.kruskal(*groups)
            n = sum(len(g) for g in groups)
            k = len(groups)
            effect = (
                float((statistic - k + 1) / (n - k))
                if n > k else None
            )
            effect_name = "epsilon_squared"
            test_name = "Kruskal-Wallis"

        results.append({
            "sensor": sensor,
            "subsets": names,
            "engine_counts": [len(g) for g in groups],
            "test": test_name,
            "statistic": float(statistic),
            "p_raw": float(p_value),
            "effect_size_name": effect_name,
            "effect_size": effect,
            "normality_pvalues": normality_p,
            "levene_p": float(levene_p),
        })

    adjusted = _bh_adjust([
        item.get("p_raw") for item in results
    ])

    for item, p_adjusted in zip(results, adjusted):
        item["p_adjusted_bh"] = p_adjusted

    return _json_safe({
        "sensor_tests": results,
        "correction": "Benjamini-Hochberg FDR",
        "unit_of_analysis": "engine-level sensor means",
    })


# ---------------------------------------------------------------------
# C. Sensor and RUL association
# ---------------------------------------------------------------------

def sensor_rul_correlations(
    df: pd.DataFrame,
    sensors: list[str],
) -> dict[str, Any]:
    """Compute sensor-RUL correlations.

    If RUL is absent, calculate training RUL as maximum cycle minus
    current cycle, grouped by subset and engine where available.
    """
    data = df.copy()

    if "RUL" not in data.columns:
        data = add_training_rul(data)

    results = []

    for sensor in sensors:
        if sensor not in data.columns:
            results.append({
                "sensor": sensor,
                "note": "Sensor column not found.",
            })
            continue

        valid = data[[sensor, "RUL"]].dropna()
        x = valid[sensor].to_numpy(dtype=float)
        y = valid["RUL"].to_numpy(dtype=float)

        if (
            len(x) < 4
            or np.unique(x).size < 2
            or np.unique(y).size < 2
        ):
            results.append({
                "sensor": sensor,
                "pearson_r": None,
                "pearson_p": None,
                "spearman_rho": None,
                "spearman_p": None,
                "note": "Insufficient data or constant values.",
            })
            continue

        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)

        z = np.arctanh(np.clip(pearson.statistic, -0.999999, 0.999999))
        standard_error = 1 / np.sqrt(len(x) - 3)

        ci = [
            float(np.tanh(z - 1.96 * standard_error)),
            float(np.tanh(z + 1.96 * standard_error)),
        ]

        results.append({
            "sensor": sensor,
            "n_rows": len(x),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "pearson_ci95": ci,
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        })

    adjusted = _bh_adjust([
        item.get("pearson_p") for item in results
    ])

    for item, p_adjusted in zip(results, adjusted):
        item["pearson_p_adjusted_bh"] = p_adjusted

    return _json_safe({
        "sensor_rul_corr": results,
        "correction": "Benjamini-Hochberg FDR",
        "rul_definition": "max_cycle_for_engine - current_cycle",
        "note": (
            "Row-level correlations can be influenced by repeated "
            "measurements within engines; interpret as exploratory."
        ),
    })


# ---------------------------------------------------------------------
# D. Sensor trends versus operating cycle
# ---------------------------------------------------------------------

def sensor_cycle_trends(
    df: pd.DataFrame,
    sensors: list[str],
) -> dict[str, Any]:
    """Fit mixed-effects models with engine-specific random intercepts."""
    results = []
    data_all = df.copy()

    engine_col = _engine_column(data_all)

    # Ensure engines with the same numeric ID in different subsets
    # are treated as different engines.
    if "subset" in data_all.columns:
        data_all["_engine_group"] = (
            data_all["subset"].astype(str)
            + "_"
            + data_all[engine_col].astype(str)
        )
    else:
        data_all["_engine_group"] = data_all[engine_col].astype(str)

    for sensor in sensors:
        if sensor not in data_all.columns:
            results.append({
                "sensor": sensor,
                "note": "Sensor column not found.",
            })
            continue

        model_data = data_all[
            ["_engine_group", "cycle", sensor]
        ].dropna()

        try:
            model = smf.mixedlm(
                f'Q("{sensor}") ~ cycle',
                data=model_data,
                groups=model_data["_engine_group"],
            )
            fitted = model.fit(reml=False)

            random_variance = float(fitted.cov_re.iloc[0, 0])
            residual_variance = float(fitted.scale)
            denominator = random_variance + residual_variance

            results.append({
                "sensor": sensor,
                "slope": float(fitted.params["cycle"]),
                "std_error": float(fitted.bse["cycle"]),
                "p_value": float(fitted.pvalues["cycle"]),
                "icc": (
                    random_variance / denominator
                    if denominator > 0 else None
                ),
                "converged": bool(fitted.converged),
            })
        except Exception as exc:
            results.append({
                "sensor": sensor,
                "error": str(exc),
            })

    return _json_safe({"sensor_cycle_mixed": results})


# ---------------------------------------------------------------------
# E. Model prediction error comparison
# ---------------------------------------------------------------------

def compare_model_errors(
    error_dict: dict[str, list[float]],
) -> dict[str, Any]:
    """Compare aligned per-engine errors for pairs of models."""
    names = list(error_dict)
    comparisons = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.asarray(error_dict[names[i]], dtype=float)
            b = np.asarray(error_dict[names[j]], dtype=float)

            if len(a) != len(b):
                raise ValueError(
                    "Model error arrays must refer to the same engines "
                    "and have equal lengths."
                )

            valid = np.isfinite(a) & np.isfinite(b)
            a, b = a[valid], b[valid]
            diff = a - b

            if len(diff) < 2:
                comparisons.append({
                    "model_a": names[i],
                    "model_b": names[j],
                    "error": "Insufficient paired observations.",
                })
                continue

            normality_p = (
                float(stats.shapiro(diff).pvalue)
                if len(diff) >= 3 else None
            )

            if normality_p is not None and normality_p > 0.05:
                test = stats.ttest_rel(a, b)
                sd = float(np.std(diff, ddof=1))
                effect = float(np.mean(diff) / sd) if sd > 0 else None
                test_name = "Paired t-test"
                effect_name = "cohens_d_paired"
            else:
                if np.allclose(diff, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    test = stats.wilcoxon(a, b)
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)

                if normality_p is not None:
                    statistic = float(test.statistic) if not np.allclose(diff, 0) else 0.0
                effect = None
                effect_name = "not_calculated_for_wilcoxon"
                test_name = "Wilcoxon signed-rank"

            if normality_p is not None and normality_p > 0.05:
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            comparisons.append({
                "model_a": names[i],
                "model_b": names[j],
                "test": test_name,
                "statistic": statistic,
                "p_value": p_value,
                "normality_p": normality_p,
                "effect_size_name": effect_name,
                "effect_size": effect,
            })

    adjusted = _bh_adjust([
        item.get("p_value") for item in comparisons
    ])

    for item, p_adjusted in zip(comparisons, adjusted):
        item["p_adjusted_bh"] = p_adjusted

    return _json_safe({
        "pairwise_error_comparisons": comparisons,
        "correction": "Benjamini-Hochberg FDR",
    })


# ---------------------------------------------------------------------
# Full report orchestration
# ---------------------------------------------------------------------

def run_full_inference(
    train_dfs: dict[str, pd.DataFrame],
    test_dfs: dict[str, pd.DataFrame] | None = None,
    model_error_dict: dict[str, list[float]] | None = None,
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the analyses and write the statistical inference report."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {}

    report["descriptive_statistics"] = {
        subset: descriptive_statistics(df)
        for subset, df in train_dfs.items()
    }

    lifetimes = lifetime_aggregation(train_dfs)
    report["lifetime_comparison"] = test_lifetime_differences(
        lifetimes
    )

    sensor_list = [
        f"sensor_{i}" for i in range(1, 6)
        if all(f"sensor_{i}" in df.columns for df in train_dfs.values())
    ]

    report["sensor_distribution"] = sensor_distribution_tests(
        train_dfs, sensor_list
    )

    # Add subset labels before combining so repeated engine IDs remain unique.
    combined_parts = []
    for subset, df in train_dfs.items():
        part = df.copy()
        part["subset"] = subset
        combined_parts.append(part)

    combined_train = pd.concat(
        combined_parts, ignore_index=True
    )

    report["sensor_rul_association"] = sensor_rul_correlations(
        combined_train, sensor_list
    )

    report["sensor_cycle_trends"] = sensor_cycle_trends(
        combined_train, sensor_list
    )

    if model_error_dict:
        report["model_error_comparison"] = compare_model_errors(
            model_error_dict
        )

    json_path = output_dir / "statistical_inference_report.json"
    json_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Statistical inference report written to {json_path}")
    return report


# ---------------------------------------------------------------------
# Sanity tests and command-line entry point
# ---------------------------------------------------------------------

def _test_lifetime_aggregation() -> None:
    """Check lifetime aggregation on a small known dataset."""
    sample = pd.DataFrame({
        "unit_id": [1, 1, 2, 2, 2],
        "cycle": [10, 15, 5, 7, 12],
        "sensor_1": [0.1, 0.2, 0.3, 0.4, 0.5],
    })

    result = lifetime_aggregation({"FD001": sample})
    actual = dict(zip(result["unit_id"], result["lifetime"]))

    assert actual == {1: 15.0, 2: 12.0}
    assert "final_rul" not in result.columns

    with_rul = add_training_rul(sample)
    assert with_rul["RUL"].tolist() == [5, 0, 7, 5, 0]

    print("Lifetime aggregation and training RUL tests passed.")


if __name__ == "__main__":
    _test_lifetime_aggregation()

    from .data_loader import load_subset

    subsets = {
        subset: load_subset(subset)[0]
        for subset in ("FD001", "FD002", "FD003", "FD004")
    }

    run_full_inference(subsets)
