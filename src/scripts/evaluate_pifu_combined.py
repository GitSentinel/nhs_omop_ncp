"""
Supplementary Combined FastPIFU Evaluation

Pools the held-out external evaluation set and challenge set for an
evaluation-only analysis of the existing fine-tuned FastPIFU classifier.

No model training, threshold tuning, or parameter selection is performed.

Run:
uv run python src/scripts/evaluate_pifu_combined.py
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

# Experiment Tracking
import mlflow

# Numerical Libraries
import numpy as np

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# Project Configuration
from src.config.pifu_settings import (  # noqa: E402
    MLFLOW_EXPERIMENT_FINETUNE,
    PIFU_CHALLENGE_PATH,
    PIFU_EVALUATION_DIR,
    PIFU_ID_TO_LABEL,
    PIFU_LABEL_IDS,
    PIFU_OUTPUT_DIR,
    PIFU_TEST_PATH,
)
from src.config.settings import settings  # noqa: E402

# Authoritative FastPIFU Classifier
from src.inference.pifu_classifier import (  # noqa: E402
    load_finetuned_model,
    score_probabilities,
)

# Existing Evaluation Helpers
from src.scripts.evaluate_pifu import (  # noqa: E402
    compute_metrics,
    load_samples,
    metric_subset,
    unload_model,
    write_split_artifacts,
)

# Analysis Configuration
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 42

OUTPUT_PREFIX = "finetuned_combined_external_challenge"

BOOTSTRAP_METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "not_eligible_precision",
    "not_eligible_recall",
    "borderline_precision",
    "borderline_recall",
    "eligible_precision",
    "eligible_recall",
    "unsafe_eligible_rate",
    "manual_review_rate",
)

# Dissertation Result Export Paths
DISSERTATION_RESULTS_ROOT = PROJECT_ROOT / "results" / "dissertation_results"

COMBINED_ARTIFACTS_DIR = (
    DISSERTATION_RESULTS_ROOT
    / "02_pifu_model"
    / "artifacts"
    / "combined_external_challenge"
)

DISSERTATION_TABLES_DIR = DISSERTATION_RESULTS_ROOT / "04_dissertation_tables"

STATISTICAL_RESULTS_DIR = DISSERTATION_RESULTS_ROOT / "07_statistical_analysis"


def annotate_source(
    samples: list[dict],
    evaluation_source: str,
) -> list[dict]:
    """Copy samples and record their original evaluation split."""

    # Source Annotation
    annotated_samples: list[dict] = []

    for sample in samples:
        copied = dict(sample)

        copied["original_source"] = copied.get("source")

        copied["source"] = evaluation_source

        annotated_samples.append(copied)

    return annotated_samples


def class_counts(
    labels: np.ndarray,
) -> dict[str, int]:
    """Return class counts using FastPIFU class names."""

    # Label Counting
    counts = Counter(int(value) for value in labels)

    return {
        PIFU_ID_TO_LABEL[label]: counts.get(
            label,
            0,
        )
        for label in PIFU_LABEL_IDS
    }


def stratified_bootstrap_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one class-stratified bootstrap sample.

    Each observed ground-truth class is independently resampled with
    replacement using the same number of cases as in the original pooled
    evaluation set.
    """

    # Class-Stratified Resampling
    sampled_groups: list[np.ndarray] = []

    for label in PIFU_LABEL_IDS:
        class_indices = np.flatnonzero(labels == label)

        if len(class_indices) == 0:
            continue

        sampled_indices = rng.choice(
            class_indices,
            size=len(class_indices),
            replace=True,
        )

        sampled_groups.append(sampled_indices)

    if not sampled_groups:
        raise ValueError("No labelled cases were available for bootstrap resampling.")

    # Combined Bootstrap Indices
    combined_indices = np.concatenate(sampled_groups)

    rng.shuffle(combined_indices)

    return combined_indices


def percentile_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate a two-sided percentile confidence interval."""

    # Confidence Limits
    alpha = 1.0 - confidence

    lower, upper = np.quantile(
        values,
        [
            alpha / 2.0,
            1.0 - alpha / 2.0,
        ],
    )

    return (
        float(lower),
        float(upper),
    )


def calculate_stratified_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Estimate stratified-bootstrap confidence intervals."""

    # Bootstrap Validation
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")

    if len(labels) != len(predictions):
        raise ValueError(
            "Labels and predictions must contain the same number of cases."
        )

    if len(labels) == 0:
        raise ValueError("At least one case is required for bootstrap analysis.")

    # Random Generator
    rng = np.random.default_rng(seed)

    # Metric Storage
    bootstrap_values: dict[str, list[float]] = {
        metric: [] for metric in BOOTSTRAP_METRIC_NAMES
    }

    # Bootstrap Replicates
    for replicate_index in range(replicates):
        indices = stratified_bootstrap_indices(
            labels,
            rng,
        )

        metrics = compute_metrics(
            labels[indices],
            predictions[indices],
        )

        for metric in BOOTSTRAP_METRIC_NAMES:
            value = metrics.get(metric)

            if isinstance(
                value,
                (int, float),
            ) and math.isfinite(float(value)):
                bootstrap_values[metric].append(float(value))

        # Progress Reporting
        if (replicate_index + 1) % 100 == 0 or replicate_index + 1 == replicates:
            print(f"Bootstrap {replicate_index + 1:,}/{replicates:,}")

    # Bootstrap Summary
    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for metric, values in bootstrap_values.items():
        if not values:
            continue

        array = np.asarray(
            values,
            dtype=float,
        )

        lower, upper = percentile_interval(array)

        summary[metric] = {
            "bootstrap_mean": float(np.mean(array)),
            "ci_95_lower": lower,
            "ci_95_upper": upper,
        }

    return summary


def build_combined_dataset_payload(
    samples: list[dict],
    labels: np.ndarray,
) -> dict:
    """Build an auditable description of the pooled evaluation set."""

    # Dataset Description
    return {
        "task": "pifu_eligibility_3class",
        "analysis": "combined_external_challenge",
        "evaluation_only": True,
        "training_used": False,
        "threshold_tuning_used": False,
        "n_samples": len(samples),
        "class_counts": class_counts(labels),
        "source_counts": {
            "external_test": sum(
                sample["source"] == "external_test" for sample in samples
            ),
            "challenge": sum(sample["source"] == "challenge" for sample in samples),
        },
        "samples": samples,
    }


def finite_bootstrap_metrics(
    bootstrap_summary: dict,
) -> dict[str, float]:
    """Flatten finite bootstrap statistics for MLflow logging."""

    # Metric Flattening
    metrics: dict[str, float] = {}

    for metric, statistics in bootstrap_summary.items():
        for statistic, value in statistics.items():
            numeric_value = float(value)

            if math.isfinite(numeric_value):
                metrics[f"combined_{metric}_{statistic}"] = numeric_value

    return metrics


def print_combined_metrics(
    metrics: dict,
    n_cases: int,
) -> None:
    """Print the combined evaluation point estimates."""

    # Console Summary
    print()
    print("-" * 68)
    print(f"COMBINED {n_cases}-CASE RESULTS")
    print("-" * 68)

    print(f"Accuracy            : {metrics['accuracy']:.4f}")

    print(f"Balanced accuracy   : {metrics['balanced_accuracy']:.4f}")

    print(f"Macro F1            : {metrics['macro_f1']:.4f}")

    print(f"NOT_ELIGIBLE recall : {metrics['not_eligible_recall']:.4f}")

    print(f"BORDERLINE recall   : {metrics['borderline_recall']:.4f}")

    print(f"ELIGIBLE precision  : {metrics['eligible_precision']:.4f}")

    print(f"ELIGIBLE recall     : {metrics['eligible_recall']:.4f}")

    print(
        "Unsafe eligible     : "
        f"{metrics['unsafe_eligible_count']} "
        f"({metrics['unsafe_eligible_rate']:.4f})"
    )

    print(f"Confusion matrix    : {metrics['confusion_matrix']}")


def print_bootstrap_summary(
    metrics: dict,
    bootstrap_summary: dict,
) -> None:
    """Print selected 95% stratified-bootstrap intervals."""

    # Selected Dissertation Metrics
    display_metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "not_eligible_recall",
        "borderline_recall",
        "eligible_precision",
        "eligible_recall",
    )

    print()
    print("95% STRATIFIED BOOTSTRAP INTERVALS")
    print("-" * 68)

    for metric in display_metrics:
        values = bootstrap_summary.get(metric)

        if values is None:
            print(f"{metric:24s}: unavailable")
            continue

        print(
            f"{metric:24s}: "
            f"{metrics[metric]:.4f} "
            "["
            f"{values['ci_95_lower']:.4f}, "
            f"{values['ci_95_upper']:.4f}"
            "]"
        )


def export_dissertation_results(
    *,
    artifact_paths: dict[str, Path],
    metrics: dict,
    bootstrap_summary: dict,
    bootstrap_path: Path,
    labels: np.ndarray,
    n_external: int,
    n_challenge: int,
) -> dict[str, Path]:
    """
    Export safe combined-analysis results into the frozen
    dissertation-results structure.

    Raw clinic-letter text is deliberately not exported.
    """

    # Create Result Directories
    COMBINED_ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DISSERTATION_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATISTICAL_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    exported_paths: dict[str, Path] = {}

    # Archive Main Evaluation Artifacts
    for key in (
        "metrics_path",
        "predictions_path",
        "report_path",
        "confusion_path",
    ):
        source = Path(artifact_paths[key])

        destination = COMBINED_ARTIFACTS_DIR / source.name

        shutil.copy2(
            source,
            destination,
        )

        exported_paths[key] = destination

    # Bootstrap Result
    bootstrap_archive_path = COMBINED_ARTIFACTS_DIR / bootstrap_path.name

    shutil.copy2(
        bootstrap_path,
        bootstrap_archive_path,
    )

    exported_paths["bootstrap_archive"] = bootstrap_archive_path

    # Safe Combined Summary
    summary_path = COMBINED_ARTIFACTS_DIR / "combined_summary.json"

    safe_summary = {
        "analysis": ("combined_external_challenge"),
        "evaluation_only": True,
        "training_used": False,
        "threshold_tuning_used": False,
        "n_external": n_external,
        "n_challenge": n_challenge,
        "n_combined": int(len(labels)),
        "class_counts": (class_counts(labels)),
        "point_estimates": (metric_subset(metrics)),
        "bootstrap": {
            "method": ("class-stratified bootstrap"),
            "replicates": (BOOTSTRAP_REPLICATES),
            "seed": (BOOTSTRAP_SEED),
            "metrics": (bootstrap_summary),
        },
        "interpretation": (
            "Supplementary pooled robustness analysis. "
            "The 174 cases reuse the existing external-test "
            "and challenge observations and therefore do not "
            "constitute a third independent test set."
        ),
    }

    summary_path.write_text(
        json.dumps(
            safe_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exported_paths["summary"] = summary_path

    # Dissertation Table Inputs
    table_exports = {
        Path(artifact_paths["metrics_path"]): (
            "finetuned_combined_external_challenge_metrics.json"
        ),
        Path(artifact_paths["confusion_path"]): (
            "finetuned_combined_external_challenge_confusion_matrix.csv"
        ),
        Path(artifact_paths["report_path"]): (
            "finetuned_combined_external_challenge_classification_report.txt"
        ),
    }

    for source, filename in table_exports.items():
        destination = DISSERTATION_TABLES_DIR / filename

        shutil.copy2(
            source,
            destination,
        )

    # Statistical Analysis Outputs
    statistical_bootstrap_path = (
        STATISTICAL_RESULTS_DIR / "combined_stratified_bootstrap.json"
    )

    shutil.copy2(
        bootstrap_path,
        statistical_bootstrap_path,
    )

    exported_paths["statistical_bootstrap"] = statistical_bootstrap_path

    # Dissertation-Ready Point Estimate + CI Table
    ci_table_path = STATISTICAL_RESULTS_DIR / "combined_metrics_with_ci.csv"

    with ci_table_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "point_estimate",
                "bootstrap_mean",
                "ci_95_lower",
                "ci_95_upper",
            ],
        )

        writer.writeheader()

        for metric_name, values in bootstrap_summary.items():
            point_estimate = metrics.get(metric_name)

            if point_estimate is None:
                continue

            writer.writerow(
                {
                    "metric": (metric_name),
                    "point_estimate": (float(point_estimate)),
                    "bootstrap_mean": (values["bootstrap_mean"]),
                    "ci_95_lower": (values["ci_95_lower"]),
                    "ci_95_upper": (values["ci_95_upper"]),
                }
            )

    exported_paths["ci_table"] = ci_table_path

    return exported_paths


def main() -> None:
    """Run the supplementary pooled FastPIFU evaluation."""

    # Held-Out Evaluation Sets
    external_samples = annotate_source(
        load_samples(PIFU_TEST_PATH),
        "external_test",
    )

    challenge_samples = annotate_source(
        load_samples(PIFU_CHALLENGE_PATH),
        "challenge",
    )

    combined_samples = external_samples + challenge_samples

    if not combined_samples:
        raise ValueError("The combined evaluation set contains no samples.")

    # Text and Label Extraction
    texts = [str(sample["text"]) for sample in combined_samples]

    labels = np.asarray(
        [int(sample["label"]) for sample in combined_samples],
        dtype=int,
    )

    # Console Header
    print("=" * 68)
    print("COMBINED FASTPIFU EXTERNAL + CHALLENGE EVALUATION")
    print("=" * 68)

    print(f"External test : {len(external_samples)}")

    print(f"Challenge     : {len(challenge_samples)}")

    print(f"Combined      : {len(combined_samples)}")

    print(f"Class counts  : {class_counts(labels)}")

    print()
    print("Evaluation only: no training or threshold selection is performed.")

    # Fine-Tuned Model Loading
    model, tokenizer = load_finetuned_model()

    try:
        # Exact Label Probability Scoring
        probabilities = score_probabilities(
            model,
            tokenizer,
            texts,
        )

        predicted_indices = probabilities.argmax(axis=1)

        predictions = np.asarray(
            [PIFU_LABEL_IDS[int(index)] for index in predicted_indices],
            dtype=int,
        )

        # Combined Point Estimates
        metrics = compute_metrics(
            labels,
            predictions,
        )

        print_combined_metrics(
            metrics,
            n_cases=len(combined_samples),
        )

        # Standard Evaluation Artifacts
        artifact_paths = write_split_artifacts(
            samples=combined_samples,
            probabilities=probabilities,
            predictions=predictions,
            metrics=metrics,
            output_prefix=OUTPUT_PREFIX,
        )

    finally:
        # Model Memory Cleanup
        unload_model(
            model,
            tokenizer,
        )

    # Evaluation Output Directory
    PIFU_EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Combined Dataset Audit File
    combined_dataset_path = PIFU_EVALUATION_DIR / (
        f"combined_external_challenge_{len(combined_samples)}.json"
    )

    combined_dataset_path.write_text(
        json.dumps(
            build_combined_dataset_payload(
                combined_samples,
                labels,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Stratified Bootstrap
    print()
    print("=" * 68)
    print(f"STRATIFIED BOOTSTRAP ({BOOTSTRAP_REPLICATES:,} replicates)")
    print("=" * 68)

    bootstrap_summary = calculate_stratified_bootstrap(
        labels,
        predictions,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )

    # Bootstrap Output
    bootstrap_path = PIFU_EVALUATION_DIR / (
        "finetuned_combined_stratified_bootstrap.json"
    )

    bootstrap_payload = {
        "method": ("class-stratified non-parametric bootstrap"),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "n_samples": len(labels),
        "class_counts": class_counts(labels),
        "point_estimates": metric_subset(metrics),
        "bootstrap": bootstrap_summary,
    }

    bootstrap_path.write_text(
        json.dumps(
            bootstrap_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Export Dissertation Results
    export_dissertation_results(
        artifact_paths=artifact_paths,
        metrics=metrics,
        bootstrap_summary=bootstrap_summary,
        bootstrap_path=bootstrap_path,
        labels=labels,
        n_external=len(external_samples),
        n_challenge=len(challenge_samples),
    )

    # Bootstrap Console Summary
    print_bootstrap_summary(
        metrics,
        bootstrap_summary,
    )

    # MLflow Setup
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_FINETUNE)

    # MLflow Run
    with mlflow.start_run(
        run_name=("evaluation_pifu_combined_external_challenge_stratified"),
        tags={
            "task": "pifu_eligibility_3class",
            "analysis": "combined_external_challenge",
            "evaluation_only": "true",
            "stratified_bootstrap": "true",
            "human_review_required": "true",
        },
    ):
        # Parameters
        mlflow.log_params(
            {
                "adapter": str(PIFU_OUTPUT_DIR),
                "n_external": len(external_samples),
                "n_challenge": len(challenge_samples),
                "n_combined": len(combined_samples),
                "n_not_eligible": int(np.sum(labels == 0)),
                "n_borderline": int(np.sum(labels == 1)),
                "n_eligible": int(np.sum(labels == 2)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )

        # Point Metrics
        point_metrics = {
            f"combined_{key}": float(value)
            for key, value in metric_subset(metrics).items()
            if (
                isinstance(
                    value,
                    (int, float),
                )
                and math.isfinite(float(value))
            )
        }

        mlflow.log_metrics(point_metrics)

        # Bootstrap Metrics
        mlflow.log_metrics(finite_bootstrap_metrics(bootstrap_summary))

        # Standard Evaluation Artifacts
        for path in artifact_paths.values():
            mlflow.log_artifact(
                str(path),
                artifact_path="combined_evaluation",
            )

        # Combined Dataset Artifact
        mlflow.log_artifact(
            str(combined_dataset_path),
            artifact_path="combined_evaluation",
        )

        # Bootstrap Artifact
        mlflow.log_artifact(
            str(bootstrap_path),
            artifact_path="combined_evaluation",
        )

    # Final Summary
    print()
    print("=" * 68)
    print("COMBINED EVALUATION COMPLETE")
    print("=" * 68)

    print(f"Combined dataset : {combined_dataset_path}")

    print(f"Metrics          : {artifact_paths['metrics_path']}")

    print(f"Predictions      : {artifact_paths['predictions_path']}")

    print(f"Confusion matrix : {artifact_paths['confusion_path']}")

    print(f"Bootstrap        : {bootstrap_path}")


if __name__ == "__main__":
    main()
