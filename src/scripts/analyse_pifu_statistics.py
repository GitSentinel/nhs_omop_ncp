"""
Paired statistical analysis for frozen FastPIFU predictions.

Compares the base and fine-tuned FastPIFU models using:
- paired class-stratified non-parametric bootstrap confidence intervals;
- exact two-sided McNemar tests; and
- Wilson score intervals for selected proportions.

The external evaluation set is treated as the primary held-out evaluation.
Challenge-set inference is explicitly labelled exploratory.

Example:
uv run python src/scripts/analyse_pifu_statistics.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

# Numerical Libraries
import numpy as np

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


# FastPIFU Label Space
LABELS = (0, 1, 2)

# Statistical Analysis Defaults
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_SEED = 20260826
CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class PredictionRow:
    """One case-level model prediction."""

    sample_id: str
    true_label: int
    predicted_label: int


@dataclass(frozen=True)
class AlignedPredictions:
    """Paired base and fine-tuned predictions for identical cases."""

    sample_ids: list[str]
    true_labels: np.ndarray
    base_predictions: np.ndarray
    finetuned_predictions: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    # Project Root
    project_root = Path(__file__).resolve().parents[2]

    # CLI Parser
    parser = argparse.ArgumentParser(
        description="Paired bootstrap and McNemar analysis for FastPIFU."
    )

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=(
            project_root
            / "results"
            / "dissertation_results"
            / "02_pifu_model"
            / "artifacts"
        ),
        help="Directory containing the frozen PIFU prediction CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "results"
            / "dissertation_results"
            / "07_statistical_analysis"
        ),
        help="Directory for statistical-analysis outputs.",
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
        help="Number of paired bootstrap replicates.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    # Bootstrap Validation
    if args.bootstraps < 100:
        raise ValueError("--bootstraps must be at least 100.")

    return args


def read_prediction_csv(
    path: Path,
) -> list[PredictionRow]:
    """Load and validate one case-level prediction CSV."""

    # File Validation
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    rows: list[PredictionRow] = []

    # Prediction Loading
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required_columns = {
            "sample_id",
            "true_label",
            "predicted_label",
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        seen_ids: set[str] = set()

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            sample_id = str(
                row["sample_id"]
            ).strip()

            if not sample_id:
                raise ValueError(
                    f"Blank sample_id in {path} at row {row_number}."
                )

            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate sample_id {sample_id!r} in {path}."
                )

            seen_ids.add(sample_id)

            true_label = int(
                row["true_label"]
            )

            predicted_label = int(
                row["predicted_label"]
            )

            if true_label not in LABELS:
                raise ValueError(
                    f"Unexpected true label {true_label} "
                    f"for {sample_id} in {path}."
                )

            if predicted_label not in LABELS:
                raise ValueError(
                    f"Unexpected predicted label {predicted_label} "
                    f"for {sample_id} in {path}."
                )

            rows.append(
                PredictionRow(
                    sample_id=sample_id,
                    true_label=true_label,
                    predicted_label=predicted_label,
                )
            )

    if not rows:
        raise ValueError(
            f"No prediction rows found in {path}."
        )

    return rows


def align_predictions(
    base_rows: list[PredictionRow],
    finetuned_rows: list[PredictionRow],
) -> AlignedPredictions:
    """Align base and fine-tuned predictions by sample ID."""

    # Case Lookup
    base_by_id = {
        row.sample_id: row
        for row in base_rows
    }

    finetuned_by_id = {
        row.sample_id: row
        for row in finetuned_rows
    }

    base_ids = set(base_by_id)
    finetuned_ids = set(finetuned_by_id)

    # Case-Set Validation
    if base_ids != finetuned_ids:
        only_base = sorted(
            base_ids - finetuned_ids
        )

        only_finetuned = sorted(
            finetuned_ids - base_ids
        )

        raise ValueError(
            "Base and fine-tuned prediction files do not contain "
            "the same cases. "
            f"Only base: {only_base[:10]}; "
            f"only fine-tuned: {only_finetuned[:10]}"
        )

    # Preserve Base-File Order
    sample_ids = [
        row.sample_id
        for row in base_rows
    ]

    true_labels: list[int] = []
    base_predictions: list[int] = []
    finetuned_predictions: list[int] = []

    # Pair Construction
    for sample_id in sample_ids:
        base_row = base_by_id[sample_id]
        finetuned_row = finetuned_by_id[sample_id]

        if base_row.true_label != finetuned_row.true_label:
            raise ValueError(
                f"Ground-truth mismatch for {sample_id}: "
                f"base={base_row.true_label}, "
                f"fine-tuned={finetuned_row.true_label}."
            )

        true_labels.append(
            base_row.true_label
        )

        base_predictions.append(
            base_row.predicted_label
        )

        finetuned_predictions.append(
            finetuned_row.predicted_label
        )

    return AlignedPredictions(
        sample_ids=sample_ids,
        true_labels=np.asarray(
            true_labels,
            dtype=int,
        ),
        base_predictions=np.asarray(
            base_predictions,
            dtype=int,
        ),
        finetuned_predictions=np.asarray(
            finetuned_predictions,
            dtype=int,
        ),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate the selected FastPIFU statistical-analysis metrics."""

    # NOT_ELIGIBLE Cases
    not_eligible_mask = y_true == 0

    if not np.any(not_eligible_mask):
        not_eligible_recall = float("nan")
        unsafe_eligible_rate = float("nan")

    else:
        not_eligible_recall = float(
            np.mean(
                y_pred[not_eligible_mask] == 0
            )
        )

        unsafe_eligible_rate = float(
            np.mean(
                y_pred[not_eligible_mask] == 2
            )
        )

    # Classification Metrics
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "not_eligible_recall": not_eligible_recall,
        "unsafe_eligible_rate": unsafe_eligible_rate,
    }


def stratified_bootstrap_indices(
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample cases with replacement within each observed true class.

    The same sampled indices are subsequently applied to the base and
    fine-tuned predictions, preserving both pairing and class counts.
    """

    # Stratified Resampling
    sampled_groups: list[np.ndarray] = []

    for label in np.unique(y_true):
        label_indices = np.flatnonzero(
            y_true == label
        )

        sampled_indices = rng.choice(
            label_indices,
            size=len(label_indices),
            replace=True,
        )

        sampled_groups.append(
            sampled_indices
        )

    return np.concatenate(
        sampled_groups
    )


def percentile_interval(
    values: np.ndarray,
    confidence: float = CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Calculate a two-sided percentile confidence interval."""

    # Quantile Limits
    alpha = 1.0 - confidence

    low, high = np.quantile(
        values,
        [
            alpha / 2.0,
            1.0 - alpha / 2.0,
        ],
    )

    return (
        float(low),
        float(high),
    )


def paired_bootstrap(
    aligned: AlignedPredictions,
    n_bootstraps: int,
    seed: int,
) -> list[dict[str, float | str | int]]:
    """Run paired class-stratified bootstrap analysis."""

    # Bootstrap Validation
    if n_bootstraps < 100:
        raise ValueError(
            "At least 100 bootstrap replicates are required."
        )

    # Random Generator
    rng = np.random.default_rng(
        seed
    )

    y_true = aligned.true_labels
    base_predictions = aligned.base_predictions
    finetuned_predictions = aligned.finetuned_predictions

    # Point Estimates
    base_point = compute_metrics(
        y_true,
        base_predictions,
    )

    finetuned_point = compute_metrics(
        y_true,
        finetuned_predictions,
    )

    metric_names = list(
        base_point
    )

    # Bootstrap Storage
    base_samples = {
        metric: np.empty(
            n_bootstraps,
            dtype=float,
        )
        for metric in metric_names
    }

    finetuned_samples = {
        metric: np.empty(
            n_bootstraps,
            dtype=float,
        )
        for metric in metric_names
    }

    delta_samples = {
        metric: np.empty(
            n_bootstraps,
            dtype=float,
        )
        for metric in metric_names
    }

    # Bootstrap Replicates
    for bootstrap_index in range(
        n_bootstraps
    ):
        indices = stratified_bootstrap_indices(
            y_true,
            rng,
        )

        y_bootstrap = y_true[indices]

        base_bootstrap = base_predictions[
            indices
        ]

        finetuned_bootstrap = finetuned_predictions[
            indices
        ]

        base_metrics = compute_metrics(
            y_bootstrap,
            base_bootstrap,
        )

        finetuned_metrics = compute_metrics(
            y_bootstrap,
            finetuned_bootstrap,
        )

        for metric in metric_names:
            base_value = base_metrics[
                metric
            ]

            finetuned_value = finetuned_metrics[
                metric
            ]

            base_samples[metric][
                bootstrap_index
            ] = base_value

            finetuned_samples[metric][
                bootstrap_index
            ] = finetuned_value

            delta_samples[metric][
                bootstrap_index
            ] = (
                finetuned_value
                - base_value
            )

    # Confidence Intervals
    results: list[
        dict[str, float | str | int]
    ] = []

    for metric in metric_names:
        base_low, base_high = percentile_interval(
            base_samples[metric]
        )

        finetuned_low, finetuned_high = percentile_interval(
            finetuned_samples[metric]
        )

        delta_low, delta_high = percentile_interval(
            delta_samples[metric]
        )

        results.append({
            "metric": metric,
            "base": base_point[metric],
            "base_ci_low": base_low,
            "base_ci_high": base_high,
            "finetuned": finetuned_point[metric],
            "finetuned_ci_low": finetuned_low,
            "finetuned_ci_high": finetuned_high,
            "delta_ft_minus_base": (
                finetuned_point[metric]
                - base_point[metric]
            ),
            "delta_ci_low": delta_low,
            "delta_ci_high": delta_high,
            "bootstrap_replicates": n_bootstraps,
            "bootstrap_seed": seed,
            "resampling": (
                "paired_stratified_by_true_class"
            ),
        })

    return results


def exact_mcnemar(
    aligned: AlignedPredictions,
) -> dict[str, int | float]:
    """
    Calculate the exact two-sided McNemar test.

    The test uses the conditional binomial distribution of discordant pairs:

    b = base correct and fine-tuned wrong
    c = base wrong and fine-tuned correct

    Under the null hypothesis, conditional on b + c, each discordant
    direction has probability 0.5.
    """

    # Paired Correctness
    y_true = aligned.true_labels

    base_correct = (
        aligned.base_predictions
        == y_true
    )

    finetuned_correct = (
        aligned.finetuned_predictions
        == y_true
    )

    # Contingency Counts
    both_correct = int(
        np.sum(
            base_correct
            & finetuned_correct
        )
    )

    base_only_correct = int(
        np.sum(
            base_correct
            & ~finetuned_correct
        )
    )

    finetuned_only_correct = int(
        np.sum(
            ~base_correct
            & finetuned_correct
        )
    )

    both_wrong = int(
        np.sum(
            ~base_correct
            & ~finetuned_correct
        )
    )

    discordant_pairs = (
        base_only_correct
        + finetuned_only_correct
    )

    # Exact Two-Sided P-Value
    if discordant_pairs == 0:
        p_value = 1.0

    else:
        smaller_count = min(
            base_only_correct,
            finetuned_only_correct,
        )

        one_tail = (
            sum(
                math.comb(
                    discordant_pairs,
                    k,
                )
                for k in range(
                    smaller_count + 1
                )
            )
            / (2**discordant_pairs)
        )

        p_value = min(
            1.0,
            2.0 * one_tail,
        )

    return {
        "both_correct": both_correct,
        "base_only_correct": base_only_correct,
        "finetuned_only_correct": finetuned_only_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant_pairs,
        "exact_mcnemar_p": float(p_value),
    }


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Calculate a 95% Wilson score interval for a binomial proportion."""

    # Empty Denominator
    if total <= 0:
        return (
            float("nan"),
            float("nan"),
        )

    # Wilson Interval
    proportion = successes / total
    z_squared = z * z

    denominator = (
        1.0
        + z_squared / total
    )

    centre = (
        proportion
        + z_squared / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            (
                proportion
                * (1.0 - proportion)
                / total
            )
            + (
                z_squared
                / (
                    4.0
                    * total
                    * total
                )
            )
        )
        / denominator
    )

    return (
        max(
            0.0,
            centre - half_width,
        ),
        min(
            1.0,
            centre + half_width,
        ),
    )


def proportion_intervals(
    aligned: AlignedPredictions,
) -> list[dict[str, float | str | int]]:
    """Calculate Wilson intervals for selected model proportions."""

    y_true = aligned.true_labels

    output: list[
        dict[str, float | str | int]
    ] = []

    for model_name, predictions in (
        (
            "base",
            aligned.base_predictions,
        ),
        (
            "finetuned",
            aligned.finetuned_predictions,
        ),
    ):
        # Accuracy
        accuracy_successes = int(
            np.sum(
                predictions == y_true
            )
        )

        accuracy_total = len(
            y_true
        )

        low, high = wilson_interval(
            accuracy_successes,
            accuracy_total,
        )

        output.append({
            "model": model_name,
            "metric": "accuracy",
            "successes": accuracy_successes,
            "total": accuracy_total,
            "estimate": (
                accuracy_successes
                / accuracy_total
            ),
            "wilson_ci_low": low,
            "wilson_ci_high": high,
        })

        # NOT_ELIGIBLE Subset
        not_eligible_mask = (
            y_true == 0
        )

        not_eligible_total = int(
            np.sum(
                not_eligible_mask
            )
        )

        if not_eligible_total <= 0:
            continue

        # NOT_ELIGIBLE Recall
        recall_successes = int(
            np.sum(
                predictions[
                    not_eligible_mask
                ] == 0
            )
        )

        low, high = wilson_interval(
            recall_successes,
            not_eligible_total,
        )

        output.append({
            "model": model_name,
            "metric": "not_eligible_recall",
            "successes": recall_successes,
            "total": not_eligible_total,
            "estimate": (
                recall_successes
                / not_eligible_total
            ),
            "wilson_ci_low": low,
            "wilson_ci_high": high,
        })

        # Unsafe-Eligible Rate
        unsafe_successes = int(
            np.sum(
                predictions[
                    not_eligible_mask
                ] == 2
            )
        )

        low, high = wilson_interval(
            unsafe_successes,
            not_eligible_total,
        )

        output.append({
            "model": model_name,
            "metric": "unsafe_eligible_rate",
            "successes": unsafe_successes,
            "total": not_eligible_total,
            "estimate": (
                unsafe_successes
                / not_eligible_total
            ),
            "wilson_ci_low": low,
            "wilson_ci_high": high,
        })

    return output


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write a list of dictionaries to CSV."""

    # Row Validation
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    # Output Directory
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CSV Saving
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def format_ci(
    estimate: float,
    low: float,
    high: float,
) -> str:
    """Format a point estimate and 95% confidence interval."""

    return (
        f"{estimate:.4f} "
        f"[{low:.4f}, {high:.4f}]"
    )


def latex_escape(
    text: str,
) -> str:
    """Escape underscores for LaTeX output."""

    return text.replace(
        "_",
        r"\_",
    )


def write_latex_bootstrap_table(
    path: Path,
    dataset_name: str,
    results: list[dict],
    exploratory: bool,
) -> None:
    """Write a dissertation-ready paired-bootstrap LaTeX table."""

    # Metric Labels
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "macro_f1": "Macro F1",
        "not_eligible_recall": (
            r"\texttt{NOT\_ELIGIBLE} recall"
        ),
        "unsafe_eligible_rate": (
            "Unsafe-eligible rate"
        ),
    }

    rows: list[str] = []

    # Table Rows
    for result in results:
        metric = str(
            result["metric"]
        )

        rows.append(
            " & ".join([
                metric_labels.get(
                    metric,
                    latex_escape(metric),
                ),
                format_ci(
                    float(result["base"]),
                    float(result["base_ci_low"]),
                    float(result["base_ci_high"]),
                ),
                format_ci(
                    float(result["finetuned"]),
                    float(result["finetuned_ci_low"]),
                    float(result["finetuned_ci_high"]),
                ),
                format_ci(
                    float(result["delta_ft_minus_base"]),
                    float(result["delta_ci_low"]),
                    float(result["delta_ci_high"]),
                ),
            ])
            + r" \\"
        )

    # Challenge-Set Qualification
    caution = (
        " These intervals are exploratory because the challenge set "
        "was deliberately selected and is not prevalence-representative."
        if exploratory
        else ""
    )

    # Table Caption
    caption = (
        f"Paired stratified bootstrap estimates for the FastPIFU "
        f"{dataset_name}. Values are point estimate [95\\% percentile CI]; "
        f"$\\Delta$ is fine-tuned minus base.{caution}"
    )

    # LaTeX Table
    table = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:pifu_{dataset_name.replace(' ', '_')}_bootstrap}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & Base & Fine-tuned & $\Delta$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])

    path.write_text(
        table,
        encoding="utf-8",
    )


def write_methods_tex(
    path: Path,
    n_bootstraps: int,
    external_seed: int,
    challenge_seed: int,
) -> None:
    """Write the dissertation statistical-methods paragraph."""

    # Methods Text
    text = rf"""
Statistical uncertainty for the FastPIFU model comparison was examined using
{n_bootstraps:,} paired non-parametric bootstrap resamples. Resampling was
performed at the case level and stratified by ground-truth class so that each
replicate retained the class counts of the corresponding evaluation set. The
same sampled case indices were applied to the base and fine-tuned predictions,
thereby preserving the paired design. Two-sided 95\% percentile confidence
intervals were calculated for each model and for the fine-tuned-minus-base
difference in accuracy, balanced accuracy, macro F1,
\texttt{{NOT\_ELIGIBLE}} recall, and the unsafe-eligible rate. Reproducible
random-number streams were generated using seeds {external_seed} for the
external evaluation set and {challenge_seed} for the challenge set. Overall
paired classification correctness was additionally compared using an exact
two-sided McNemar test on the discordant case pairs. Wilson score intervals
were also calculated for simple proportions, including the unsafe-eligible
rate, because a conventional non-parametric bootstrap interval can be
degenerate when no events are observed. Statistical inference from the
24-case challenge set was treated as exploratory because that set was
deliberately selected, was small, and contained no ground-truth
\texttt{{BORDERLINE}} cases.
""".strip()

    path.write_text(
        text + "\n",
        encoding="utf-8",
    )


def analyse_dataset(
    dataset_key: str,
    dataset_label: str,
    base_path: Path,
    finetuned_path: Path,
    output_dir: Path,
    n_bootstraps: int,
    seed: int,
    exploratory: bool,
) -> dict:
    """Run all statistical analyses for one FastPIFU evaluation set."""

    # Prediction Loading
    base_rows = read_prediction_csv(
        base_path
    )

    finetuned_rows = read_prediction_csv(
        finetuned_path
    )

    # Paired Alignment
    aligned = align_predictions(
        base_rows,
        finetuned_rows,
    )

    # Paired Bootstrap
    bootstrap_results = paired_bootstrap(
        aligned=aligned,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Exact McNemar Test
    mcnemar_result = exact_mcnemar(
        aligned
    )

    # Wilson Intervals
    wilson_results = proportion_intervals(
        aligned
    )

    # Dataset Metadata
    for row in bootstrap_results:
        row["dataset"] = dataset_key

    for row in wilson_results:
        row["dataset"] = dataset_key

    mcnemar_row = {
        "dataset": dataset_key,
        "n_cases": len(
            aligned.true_labels
        ),
        **mcnemar_result,
        "interpretation": (
            "exploratory_only"
            if exploratory
            else "primary_external_evaluation"
        ),
    }

    # CSV Outputs
    write_csv(
        output_dir
        / f"{dataset_key}_paired_bootstrap.csv",
        bootstrap_results,
    )

    write_csv(
        output_dir
        / f"{dataset_key}_wilson_intervals.csv",
        wilson_results,
    )

    write_csv(
        output_dir
        / f"{dataset_key}_mcnemar.csv",
        [mcnemar_row],
    )

    # LaTeX Table
    write_latex_bootstrap_table(
        output_dir
        / f"{dataset_key}_bootstrap_table.tex",
        dataset_name=dataset_label,
        results=bootstrap_results,
        exploratory=exploratory,
    )

    return {
        "dataset": dataset_key,
        "n_cases": len(
            aligned.true_labels
        ),
        "class_counts": {
            str(label): int(
                np.sum(
                    aligned.true_labels
                    == label
                )
            )
            for label in LABELS
        },
        "bootstrap": bootstrap_results,
        "wilson": wilson_results,
        "mcnemar": mcnemar_row,
    }


def print_summary(
    result: dict,
) -> None:
    """Print a concise statistical-analysis summary."""

    # Dataset Header
    print()
    print("=" * 78)
    print(
        result["dataset"].upper()
    )
    print("=" * 78)

    print(
        f"Cases: {result['n_cases']}"
    )

    print(
        "Ground-truth class counts: "
        f"{result['class_counts']}"
    )

    print()

    # Bootstrap Results Header
    print(
        f"{'Metric':<26}"
        f"{'Base [95% CI]':>25}"
        f"{'Fine-tuned [95% CI]':>25}"
        f"{'Delta [95% CI]':>25}"
    )

    print("-" * 101)

    # Bootstrap Results
    for row in result["bootstrap"]:
        base = format_ci(
            float(row["base"]),
            float(row["base_ci_low"]),
            float(row["base_ci_high"]),
        )

        finetuned = format_ci(
            float(row["finetuned"]),
            float(row["finetuned_ci_low"]),
            float(row["finetuned_ci_high"]),
        )

        delta = format_ci(
            float(row["delta_ft_minus_base"]),
            float(row["delta_ci_low"]),
            float(row["delta_ci_high"]),
        )

        print(
            f"{str(row['metric']):<26}"
            f"{base:>25}"
            f"{finetuned:>25}"
            f"{delta:>25}"
        )

    # McNemar Summary
    mcnemar = result["mcnemar"]

    print()
    print("Exact McNemar test:")
    print(
        f"  both correct           : "
        f"{mcnemar['both_correct']}"
    )
    print(
        f"  base only correct      : "
        f"{mcnemar['base_only_correct']}"
    )
    print(
        f"  fine-tuned only correct: "
        f"{mcnemar['finetuned_only_correct']}"
    )
    print(
        f"  both wrong             : "
        f"{mcnemar['both_wrong']}"
    )
    print(
        f"  discordant pairs       : "
        f"{mcnemar['discordant_pairs']}"
    )
    print(
        f"  exact two-sided p      : "
        f"{mcnemar['exact_mcnemar_p']:.8g}"
    )


def main() -> None:
    """Run FastPIFU statistical analysis."""

    # CLI Setup
    args = parse_args()

    # Directory Resolution
    artifacts_dir = (
        args.artifacts_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not artifacts_dir.exists():
        raise FileNotFoundError(
            f"Artifacts directory does not exist: {artifacts_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # External Evaluation Analysis
    external_result = analyse_dataset(
        dataset_key="external_test",
        dataset_label="external test",
        base_path=(
            artifacts_dir
            / "base_external_test_predictions.csv"
        ),
        finetuned_path=(
            artifacts_dir
            / "finetuned_external_test_predictions.csv"
        ),
        output_dir=output_dir,
        n_bootstraps=args.bootstraps,
        seed=args.seed,
        exploratory=False,
    )

    # Challenge-Set Analysis
    challenge_seed = (
        args.seed + 1
    )

    challenge_result = analyse_dataset(
        dataset_key="challenge",
        dataset_label="challenge set",
        base_path=(
            artifacts_dir
            / "base_challenge_predictions.csv"
        ),
        finetuned_path=(
            artifacts_dir
            / "finetuned_challenge_predictions.csv"
        ),
        output_dir=output_dir,
        n_bootstraps=args.bootstraps,
        seed=challenge_seed,
        exploratory=True,
    )

    # Combined Results
    combined = {
        "analysis": {
            "bootstrap_replicates": args.bootstraps,
            "external_seed": args.seed,
            "challenge_seed": challenge_seed,
            "bootstrap_type": (
                "paired_stratified_nonparametric"
            ),
            "confidence_level": CONFIDENCE_LEVEL,
            "mcnemar": "exact_two_sided",
        },
        "external_test": external_result,
        "challenge": challenge_result,
    }

    # Combined JSON Output
    results_path = (
        output_dir
        / "pifu_statistical_results.json"
    )

    results_path.write_text(
        json.dumps(
            combined,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Dissertation Methods Output
    methods_path = (
        output_dir
        / "pifu_statistical_methods.tex"
    )

    write_methods_tex(
        methods_path,
        n_bootstraps=args.bootstraps,
        external_seed=args.seed,
        challenge_seed=challenge_seed,
    )

    # Console Summaries
    print_summary(
        external_result
    )

    print_summary(
        challenge_result
    )

    # Final Output Summary
    print()
    print(
        f"Outputs written to: {output_dir}"
    )
    print("Primary files:")
    print("  external_test_paired_bootstrap.csv")
    print("  external_test_mcnemar.csv")
    print("  external_test_wilson_intervals.csv")
    print("  external_test_bootstrap_table.tex")
    print("  challenge_paired_bootstrap.csv")
    print("  challenge_mcnemar.csv")
    print("  challenge_wilson_intervals.csv")
    print("  challenge_bootstrap_table.tex")
    print("  pifu_statistical_methods.tex")
    print("  pifu_statistical_results.json")


if __name__ == "__main__":
    main()