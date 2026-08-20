"""
FastPIFU Model Evaluation

Evaluates the base model and fine-tuned PIFU LoRA adapter on the
external test and challenge splits using exact label-sequence
log probabilities.

Run:
uv run --extra finetune python src/scripts/evaluate_pifu.py
"""

import argparse
import csv
import gc
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Experiment Tracking
import mlflow

# Data and Numerical Libraries
import numpy as np
import torch

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project Configuration
from src.config.pifu_settings import (  # noqa: E402
    MLFLOW_EXPERIMENT_FINETUNE,
    PIFU_BASE_MODEL,
    PIFU_CHALLENGE_PATH,
    PIFU_DATASET_SUMMARY_PATH,
    PIFU_EVALUATION_DIR,
    PIFU_EVALUATION_RUN_NAME,
    PIFU_ID_TO_LABEL,
    PIFU_LABEL_IDS,
    PIFU_MAX_LENGTH,
    PIFU_OUTPUT_DIR,
    PIFU_PROMPT_TEMPLATE,
    PIFU_TEST_PATH,
)
from src.config.settings import settings  # noqa: E402

# Project-Specific Imports
from src.inference.pifu_classifier import (  # noqa: E402
    load_base_model,
    load_finetuned_model,
    score_probabilities,
)

# Evaluation Configuration
PIFU_EVAL_BATCH_SIZE = 2


def load_samples(path: Path) -> list[dict]:
    """Load and validate one prepared PIFU split."""

    # Split Existence Check
    if not path.exists():
        raise FileNotFoundError(f"PIFU split not found: {path}")

    # Split Loading
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)

    samples = payload.get("samples")

    if not isinstance(samples, list):
        raise ValueError(f"{path} must contain a list named 'samples'.")

    if not samples:
        raise ValueError(f"{path} contains no samples.")

    # Sample Validation
    for index, sample in enumerate(samples):
        if "text" not in sample or "label" not in sample:
            raise ValueError(f"Sample {index} in {path} must contain text and label.")

        label = int(sample["label"])

        if label not in PIFU_LABEL_IDS:
            raise ValueError(f"Unexpected label {label} in sample {index} from {path}.")

    return samples


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    """Calculate general and PIFU-specific safety metrics."""

    # Confusion Matrix
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(PIFU_LABEL_IDS),
    )

    # Classification Report
    target_names = [PIFU_ID_TO_LABEL[label] for label in PIFU_LABEL_IDS]

    report_dict = classification_report(
        labels,
        predictions,
        labels=list(PIFU_LABEL_IDS),
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        labels,
        predictions,
        labels=list(PIFU_LABEL_IDS),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    # Safety Metrics
    true_not_eligible = labels == 0
    unsafe_eligible = true_not_eligible & (predictions == 2)

    unsafe_rate = (
        float(unsafe_eligible.sum()) / float(true_not_eligible.sum())
        if true_not_eligible.sum()
        else float("nan")
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "not_eligible_precision": report_dict["NOT_ELIGIBLE"]["precision"],
        "not_eligible_recall": report_dict["NOT_ELIGIBLE"]["recall"],
        "borderline_precision": report_dict["BORDERLINE"]["precision"],
        "borderline_recall": report_dict["BORDERLINE"]["recall"],
        "eligible_precision": report_dict["ELIGIBLE"]["precision"],
        "eligible_recall": report_dict["ELIGIBLE"]["recall"],
        "unsafe_eligible_count": int(unsafe_eligible.sum()),
        "unsafe_eligible_rate": unsafe_rate,
        "manual_review_rate": float(np.mean(predictions == 1)),
        "prediction_counts": dict(Counter(int(value) for value in predictions)),
        "confusion_matrix": matrix.tolist(),
        "report": report_dict,
        "report_text": report_text,
    }


def write_split_artifacts(
    samples: list[dict],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    metrics: dict,
    output_prefix: str,
) -> dict[str, Path]:
    """Write evaluation artifacts for one split."""

    # Output Paths
    PIFU_EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = PIFU_EVALUATION_DIR / f"{output_prefix}_metrics.json"
    predictions_path = PIFU_EVALUATION_DIR / f"{output_prefix}_predictions.csv"
    report_path = PIFU_EVALUATION_DIR / f"{output_prefix}_classification_report.txt"
    confusion_path = PIFU_EVALUATION_DIR / f"{output_prefix}_confusion_matrix.csv"

    # Metrics JSON
    serialisable = {
        key: value for key, value in metrics.items() if key != "report_text"
    }

    metrics_path.write_text(
        json.dumps(
            serialisable,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Classification Report
    report_path.write_text(
        metrics["report_text"],
        encoding="utf-8",
    )

    # Predictions CSV
    with open(
        predictions_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "source",
                "true_label",
                "true_label_name",
                "predicted_label",
                "predicted_label_name",
                "prob_not_eligible",
                "prob_borderline",
                "prob_eligible",
            ],
        )

        writer.writeheader()

        for sample, prediction, row_probabilities in zip(
            samples,
            predictions,
            probabilities,
            strict=True,
        ):
            true_label = int(sample["label"])

            writer.writerow(
                {
                    "sample_id": sample.get("sample_id"),
                    "source": sample.get("source"),
                    "true_label": true_label,
                    "true_label_name": (
                        sample.get("label_name") or PIFU_ID_TO_LABEL[true_label]
                    ),
                    "predicted_label": int(prediction),
                    "predicted_label_name": PIFU_ID_TO_LABEL[int(prediction)],
                    "prob_not_eligible": float(row_probabilities[0]),
                    "prob_borderline": float(row_probabilities[1]),
                    "prob_eligible": float(row_probabilities[2]),
                }
            )

    # Confusion Matrix CSV
    with open(
        confusion_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "true/predicted",
                *[PIFU_ID_TO_LABEL[label] for label in PIFU_LABEL_IDS],
            ]
        )

        for label, row in zip(
            PIFU_LABEL_IDS,
            metrics["confusion_matrix"],
            strict=True,
        ):
            writer.writerow(
                [
                    PIFU_ID_TO_LABEL[label],
                    *row,
                ]
            )

    return {
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
        "report_path": report_path,
        "confusion_path": confusion_path,
    }


def evaluate_split(
    model,
    tokenizer,
    split_name: str,
    samples: list[dict],
    output_prefix: str,
) -> dict:
    """Evaluate one model on one PIFU split and save artifacts."""

    # Split Data Extraction
    texts = [str(sample["text"]) for sample in samples]

    labels = np.asarray(
        [int(sample["label"]) for sample in samples],
        dtype=int,
    )

    # Probability Scoring
    probabilities = score_probabilities(
        model,
        tokenizer,
        texts,
    )

    predictions = probabilities.argmax(axis=1)

    # Metric Calculation
    metrics = compute_metrics(
        labels,
        predictions,
    )

    # Console Summary
    print("\n" + "=" * 68)
    print(split_name)
    print("=" * 68)
    print(f"Accuracy            : {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy   : {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1            : {metrics['macro_f1']:.4f}")
    print(f"NOT_ELIGIBLE recall : {metrics['not_eligible_recall']:.4f}")
    print(f"ELIGIBLE precision  : {metrics['eligible_precision']:.4f}")
    print(
        f"Unsafe eligible     : "
        f"{metrics['unsafe_eligible_count']} "
        f"({metrics['unsafe_eligible_rate']:.4f})"
    )
    print(f"Manual review rate  : {metrics['manual_review_rate']:.4f}")
    print(f"Confusion matrix    : {metrics['confusion_matrix']}")
    print(metrics["report_text"])

    # Artifact Writing
    artifact_paths = write_split_artifacts(
        samples=samples,
        probabilities=probabilities,
        predictions=predictions,
        metrics=metrics,
        output_prefix=output_prefix,
    )

    print(f"Metrics saved     : {artifact_paths['metrics_path']}")
    print(f"Predictions saved : {artifact_paths['predictions_path']}")

    return {
        **metrics,
        **artifact_paths,
    }


def unload_model(
    model,
    tokenizer,
) -> None:
    """Release GPU memory."""

    # Memory Cleanup
    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()


def metric_subset(metrics: dict) -> dict:
    """Return only finite numeric metrics for MLflow."""

    # Metric Filtering
    excluded = {
        "report",
        "report_text",
        "prediction_counts",
        "confusion_matrix",
        "metrics_path",
        "predictions_path",
        "report_path",
        "confusion_path",
    }

    result = {}

    for key, value in metrics.items():
        if key in excluded:
            continue

        if isinstance(value, (int, float)):
            number = float(value)

            if math.isfinite(number):
                result[key] = number

    return result


def comparison_summary(
    base_external: dict | None,
    fine_external: dict,
    base_challenge: dict | None,
    fine_challenge: dict,
) -> str:
    """Build a readable base-versus-fine-tuned summary."""

    # Summary Header
    lines = [
        f"Base Model vs Fine-Tuned PIFU Model - {PIFU_BASE_MODEL}",
        "=" * 68,
        f"Adapter: {PIFU_OUTPUT_DIR}",
        f"External test: {PIFU_TEST_PATH}",
        f"Challenge set: {PIFU_CHALLENGE_PATH}",
        "",
    ]

    # Fine-Tuned Only Evaluation Case
    if base_external is None or base_challenge is None:
        lines.extend(
            [
                "Base-model evaluation was skipped.",
                "",
                "Fine-tuned external-test metrics:",
                json.dumps(
                    metric_subset(fine_external),
                    indent=2,
                ),
                "",
                "Fine-tuned challenge-set metrics:",
                json.dumps(
                    metric_subset(fine_challenge),
                    indent=2,
                ),
            ]
        )

        return "\n".join(lines)

    # Metric Deltas
    external_macro_f1_delta = fine_external["macro_f1"] - base_external["macro_f1"]

    external_balanced_accuracy_delta = (
        fine_external["balanced_accuracy"] - base_external["balanced_accuracy"]
    )

    external_not_eligible_recall_delta = (
        fine_external["not_eligible_recall"] - base_external["not_eligible_recall"]
    )

    external_eligible_precision_delta = (
        fine_external["eligible_precision"] - base_external["eligible_precision"]
    )

    challenge_macro_f1_delta = fine_challenge["macro_f1"] - base_challenge["macro_f1"]

    challenge_not_eligible_recall_delta = (
        fine_challenge["not_eligible_recall"] - base_challenge["not_eligible_recall"]
    )

    # External Test Comparison
    lines.extend(
        [
            "External test",
            "-" * 68,
            (
                "Macro F1: "
                f"{base_external['macro_f1']:.4f} -> "
                f"{fine_external['macro_f1']:.4f} "
                f"({external_macro_f1_delta:+.4f})"
            ),
            (
                "Balanced accuracy: "
                f"{base_external['balanced_accuracy']:.4f} -> "
                f"{fine_external['balanced_accuracy']:.4f} "
                f"({external_balanced_accuracy_delta:+.4f})"
            ),
            (
                "NOT_ELIGIBLE recall: "
                f"{base_external['not_eligible_recall']:.4f} -> "
                f"{fine_external['not_eligible_recall']:.4f} "
                f"({external_not_eligible_recall_delta:+.4f})"
            ),
            (
                "ELIGIBLE precision: "
                f"{base_external['eligible_precision']:.4f} -> "
                f"{fine_external['eligible_precision']:.4f} "
                f"({external_eligible_precision_delta:+.4f})"
            ),
            (
                "Unsafe eligible count: "
                f"{base_external['unsafe_eligible_count']} -> "
                f"{fine_external['unsafe_eligible_count']}"
            ),
            "",
        ]
    )

    # Challenge Set Comparison
    lines.extend(
        [
            "Challenge set",
            "-" * 68,
            (
                "Macro F1: "
                f"{base_challenge['macro_f1']:.4f} -> "
                f"{fine_challenge['macro_f1']:.4f} "
                f"({challenge_macro_f1_delta:+.4f})"
            ),
            (
                "NOT_ELIGIBLE recall: "
                f"{base_challenge['not_eligible_recall']:.4f} -> "
                f"{fine_challenge['not_eligible_recall']:.4f} "
                f"({challenge_not_eligible_recall_delta:+.4f})"
            ),
            (
                "Unsafe eligible count: "
                f"{base_challenge['unsafe_eligible_count']} -> "
                f"{fine_challenge['unsafe_eligible_count']}"
            ),
        ]
    )

    return "\n".join(lines)


def log_result_artifacts(
    prefix: str,
    metrics: dict,
) -> None:
    """Log saved evaluation artifacts into MLflow."""

    # Artifact Logging
    for key in (
        "metrics_path",
        "predictions_path",
        "report_path",
        "confusion_path",
    ):
        mlflow.log_artifact(
            str(metrics[key]),
            artifact_path=prefix,
        )


def log_metric_block(
    prefix: str,
    metrics: dict,
) -> None:
    """Log finite numeric metrics to MLflow with a prefix."""

    # Metric Logging
    mlflow.log_metrics(
        {f"{prefix}_{key}": value for key, value in metric_subset(metrics).items()}
    )


def run_evaluation(
    include_base: bool,
) -> None:
    """Evaluate the base and PIFU-adapted models."""

    # GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for PIFU evaluation.")

    # Dataset Loading
    test_samples = load_samples(PIFU_TEST_PATH)
    challenge_samples = load_samples(PIFU_CHALLENGE_PATH)

    # Run Header
    print("=" * 68)
    print("FASTPIFU MODEL EVALUATION")
    print("=" * 68)
    print(f"Base model     : {PIFU_BASE_MODEL}")
    print(f"Adapter        : {PIFU_OUTPUT_DIR}")
    print(f"External test  : {len(test_samples):,}")
    print(f"Challenge test : {len(challenge_samples):,}")
    print()

    base_external = None
    base_challenge = None

    # Optional Base Model Evaluation
    if include_base:
        print("=" * 40)
        print("Evaluating BASE model...")
        print("=" * 40)

        base_model, base_tokenizer = load_base_model()

        base_external = evaluate_split(
            base_model,
            base_tokenizer,
            "BASE MODEL - EXTERNAL TEST",
            test_samples,
            "base_external_test",
        )

        base_challenge = evaluate_split(
            base_model,
            base_tokenizer,
            "BASE MODEL - CHALLENGE TEST",
            challenge_samples,
            "base_challenge",
        )

        unload_model(
            base_model,
            base_tokenizer,
        )

        print("Base model unloaded.\n")

    # Fine-Tuned Model Evaluation
    print("=" * 40)
    print("Evaluating FINE-TUNED PIFU model...")
    print("=" * 40)

    fine_model, fine_tokenizer = load_finetuned_model()

    fine_external = evaluate_split(
        fine_model,
        fine_tokenizer,
        "FINE-TUNED MODEL - EXTERNAL TEST",
        test_samples,
        "finetuned_external_test",
    )

    fine_challenge = evaluate_split(
        fine_model,
        fine_tokenizer,
        "FINE-TUNED MODEL - CHALLENGE TEST",
        challenge_samples,
        "finetuned_challenge",
    )

    unload_model(
        fine_model,
        fine_tokenizer,
    )

    # Comparison Summary
    summary = comparison_summary(
        base_external,
        fine_external,
        base_challenge,
        fine_challenge,
    )

    print("\n" + "=" * 68)
    print("COMPARISON: BASE VS FINE-TUNED")
    print("=" * 68)
    print(summary)

    # MLflow Setup
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_FINETUNE)

    # MLflow Run
    with mlflow.start_run(
        run_name=PIFU_EVALUATION_RUN_NAME,
        tags={
            "task": "pifu_eligibility_3class",
            "dataset": "fastpifu_cardiology",
            "base_model": PIFU_BASE_MODEL,
            "adapter": str(PIFU_OUTPUT_DIR),
            "comparison_protocol": "same_as_original_letters",
        },
    ):
        # Parameter Logging
        mlflow.log_params(
            {
                "base_model": PIFU_BASE_MODEL,
                "adapter_path": str(PIFU_OUTPUT_DIR),
                "max_length": PIFU_MAX_LENGTH,
                "n_external_test": len(test_samples),
                "n_challenge": len(challenge_samples),
                "include_base": include_base,
            }
        )

        # Metric Logging
        log_metric_block("ft_external", fine_external)
        log_metric_block("ft_challenge", fine_challenge)

        if base_external is not None:
            log_metric_block("base_external", base_external)

        if base_challenge is not None:
            log_metric_block("base_challenge", base_challenge)

        if base_external is not None:
            mlflow.log_metrics(
                {
                    "external_macro_f1_delta": (
                        fine_external["macro_f1"] - base_external["macro_f1"]
                    ),
                    "external_balanced_accuracy_delta": (
                        fine_external["balanced_accuracy"]
                        - base_external["balanced_accuracy"]
                    ),
                    "external_not_eligible_recall_delta": (
                        fine_external["not_eligible_recall"]
                        - base_external["not_eligible_recall"]
                    ),
                    "external_eligible_precision_delta": (
                        fine_external["eligible_precision"]
                        - base_external["eligible_precision"]
                    ),
                }
            )

        # Text Artifact Logging
        mlflow.log_text(
            PIFU_PROMPT_TEMPLATE,
            "prompt_template.txt",
        )

        mlflow.log_text(
            summary,
            "comparison_summary.txt",
        )

        # Evaluation Artifact Logging
        log_result_artifacts(
            "finetuned_external_test",
            fine_external,
        )

        log_result_artifacts(
            "finetuned_challenge",
            fine_challenge,
        )

        if base_external is not None:
            log_result_artifacts(
                "base_external_test",
                base_external,
            )

        if base_challenge is not None:
            log_result_artifacts(
                "base_challenge",
                base_challenge,
            )

        # Dataset Summary Logging
        if PIFU_DATASET_SUMMARY_PATH.exists():
            mlflow.log_artifact(
                str(PIFU_DATASET_SUMMARY_PATH),
                artifact_path="data",
            )

        # Adapter Configuration Logging
        adapter_config_path = PIFU_OUTPUT_DIR / "adapter_config.json"

        if adapter_config_path.exists():
            mlflow.log_artifact(
                str(adapter_config_path),
                artifact_path="adapter",
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    # CLI Argument Setup
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--finetuned-only",
        action="store_true",
        help=(
            "Skip the base-model evaluation. By default, both base and "
            "fine-tuned models are evaluated."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the FastPIFU evaluation CLI."""

    # CLI Setup
    args = parse_args()

    # Evaluation Run
    run_evaluation(
        include_base=not args.finetuned_only,
    )


if __name__ == "__main__":
    main()
