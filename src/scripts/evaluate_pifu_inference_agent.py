"""
FastPIFU Inference-System Batch Evaluation

Evaluates the fine-tuned FastPIFU classifier plus deterministic safety
review layer on the external and challenge datasets.

Run:
uv run --extra finetune python src/scripts/evaluate_pifu_inference_agent.py

Example:
uv run --extra finetune python src/scripts/evaluate_pifu_inference_agent.py \
 --gpu 0 --batch-size 2
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Experiment Tracking
import mlflow

# Data and Numerical Libraries
import numpy as np

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    # CLI Parser Setup
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the PIFU inference system on the external and challenge datasets."
        )
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU ID used for inference.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional evaluation run identifier.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    return args


def configure_runtime_environment(
    gpu_id: int,
) -> None:
    """Configure GPU visibility before model imports."""

    # GPU Selection
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Tokeniser Runtime Setting
    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )


def load_samples(
    path: Path,
) -> list[dict]:
    """Load and validate one prepared FastPIFU split."""

    # File Validation
    if not path.exists():
        raise FileNotFoundError(f"PIFU split not found: {path}")

    # JSON Loading
    with open(
        path,
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    samples = payload.get("samples")

    if not isinstance(samples, list):
        raise ValueError(f"{path} does not contain a samples list.")

    if not samples:
        raise ValueError(f"{path} contains no samples.")

    # Sample Validation
    for index, sample in enumerate(samples):
        if "text" not in sample or "label" not in sample:
            raise ValueError(f"Sample {index} in {path} must contain text and label.")

    return samples


def text_hash(
    text: str,
) -> str:
    """Return SHA-256 without saving raw clinic text."""

    # Text Hashing
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def safe_mean(
    values: list[float],
) -> float | None:
    """Return the mean or None for an empty collection."""

    # Empty Collection Handling
    if not values:
        return None

    return float(np.mean(values))


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    margins: np.ndarray,
    safety_flags: list[list[str]],
    human_review: list[bool],
    confidence_threshold: float,
    margin_threshold: float,
) -> dict:
    """Calculate classifier and inference-system metrics."""

    # Fixed Label Order
    label_ids = [0, 1, 2]

    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_ids,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=label_ids,
    )

    # Classification Correctness
    correct = predictions == labels

    true_not_eligible = labels == 0
    unsafe_eligible = true_not_eligible & (predictions == 2)

    unsafe_rate = (
        float(unsafe_eligible.sum()) / float(true_not_eligible.sum())
        if true_not_eligible.sum()
        else None
    )

    correct_confidences = confidences[correct].tolist()
    incorrect_confidences = confidences[~correct].tolist()

    return {
        "n": int(len(labels)),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "not_eligible_precision": float(precision[0]),
        "not_eligible_recall": float(recall[0]),
        "not_eligible_f1": float(class_f1[0]),
        "not_eligible_support": int(support[0]),
        "borderline_precision": float(precision[1]),
        "borderline_recall": float(recall[1]),
        "borderline_f1": float(class_f1[1]),
        "borderline_support": int(support[1]),
        "eligible_precision": float(precision[2]),
        "eligible_recall": float(recall[2]),
        "eligible_f1": float(class_f1[2]),
        "eligible_support": int(support[2]),
        "unsafe_eligible_count": int(unsafe_eligible.sum()),
        "unsafe_eligible_rate": unsafe_rate,
        "borderline_prediction_rate": float(np.mean(predictions == 1)),
        "mean_confidence": float(confidences.mean()),
        "mean_confidence_correct": safe_mean(correct_confidences),
        "mean_confidence_incorrect": safe_mean(incorrect_confidences),
        "mean_top_two_margin": float(margins.mean()),
        "low_confidence_rate": float(np.mean(confidences < confidence_threshold)),
        "close_margin_rate": float(np.mean(margins < margin_threshold)),
        "safety_flag_rate": float(np.mean([bool(flags) for flags in safety_flags])),
        "human_review_required_rate": float(np.mean(human_review)),
        "misclassified_count": int((~correct).sum()),
        "confusion_matrix": matrix.tolist(),
    }


def write_predictions(
    path: Path,
    rows: list[dict],
) -> None:
    """Save case-level inference results."""

    # Empty Output Handling
    if not rows:
        return

    # Prediction CSV Saving
    with open(
        path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(
    summaries: dict[str, dict],
) -> str:
    """Create a dissertation-friendly Markdown summary."""

    # Report Header
    lines = [
        "# PIFU Inference-System Batch Evaluation",
        "",
    ]

    # Split Summaries
    for split_name, metrics in summaries.items():
        lines.extend(
            [
                f"## {split_name}",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Cases | {metrics['n']} |",
                f"| Accuracy | {metrics['accuracy']:.4f} |",
                f"| Balanced accuracy | {metrics['balanced_accuracy']:.4f} |",
                f"| Macro F1 | {metrics['macro_f1']:.4f} |",
                f"| NOT_ELIGIBLE recall | {metrics['not_eligible_recall']:.4f} |",
                f"| BORDERLINE recall | {metrics['borderline_recall']:.4f} |",
                f"| ELIGIBLE precision | {metrics['eligible_precision']:.4f} |",
                f"| ELIGIBLE recall | {metrics['eligible_recall']:.4f} |",
                f"| Unsafe eligible count | {metrics['unsafe_eligible_count']} |",
                f"| Safety flag rate | {metrics['safety_flag_rate']:.4f} |",
                f"| Low-confidence rate | {metrics['low_confidence_rate']:.4f} |",
                f"| Close-margin rate | {metrics['close_margin_rate']:.4f} |",
                (
                    "| Human review required | "
                    f"{metrics['human_review_required_rate']:.4f} |"
                ),
                f"| Mean confidence | {metrics['mean_confidence']:.4f} |",
                "",
            ]
        )

    # Safety Note
    lines.extend(
        [
            "---",
            "",
            (
                "Human review is mandatory for all cases. Confidence and margin "
                "thresholds are research review triggers and are not clinically "
                "validated thresholds."
            ),
        ]
    )

    return "\n".join(lines)


def probability_by_label(
    row_probabilities: np.ndarray,
    label_ids: tuple[int, ...],
) -> dict[int, float]:
    """Map one probability row to numeric PIFU labels."""

    # Probability Mapping
    return {
        int(label): float(row_probabilities[index])
        for index, label in enumerate(label_ids)
    }


def build_case_prediction_row(
    sample: dict,
    split_name: str,
    true_label: int,
    predicted_label: int,
    probabilities: dict[int, float],
    confidence: float,
    margin: float,
    safety,
    label_map: dict[int, str],
) -> dict:
    """Create one case-level prediction row."""

    # Prediction Row
    return {
        "sample_id": sample.get("sample_id"),
        "source": sample.get("source"),
        "split": split_name,
        "text_sha256": text_hash(str(sample["text"])),
        "true_label": true_label,
        "true_class": label_map[true_label],
        "predicted_label": predicted_label,
        "predicted_class": label_map[predicted_label],
        "correct": true_label == predicted_label,
        "confidence": confidence,
        "prob_not_eligible": probabilities[0],
        "prob_borderline": probabilities[1],
        "prob_eligible": probabilities[2],
        "top_two_margin": margin,
        "requires_human_review": safety.requires_human_review,
        "safety_flag_count": len(safety.flags),
        "safety_flags": " | ".join(safety.flags),
    }


def finite_numeric_metrics(
    summaries: dict[str, dict],
) -> dict[str, float]:
    """Return finite scalar metrics for MLflow logging."""

    # Metric Filtering
    numeric_metrics = {}

    for split_name, summary in summaries.items():
        for key, value in summary.items():
            if isinstance(value, (int, float)) and value is not None:
                number = float(value)

                if math.isfinite(number):
                    numeric_metrics[f"{split_name}_{key}"] = number

    return numeric_metrics


def write_summary_outputs(
    output_dir: Path,
    summaries: dict[str, dict],
) -> tuple[Path, Path]:
    """Write combined JSON and Markdown summaries."""

    # Combined Output Paths
    combined_json = output_dir / "combined_summary.json"
    combined_md = output_dir / "combined_summary.md"

    # Combined JSON Saving
    combined_json.write_text(
        json.dumps(
            summaries,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Combined Markdown Saving
    combined_md.write_text(
        markdown_summary(summaries),
        encoding="utf-8",
    )

    return combined_json, combined_md


def main() -> None:
    """Run the batch inference-system evaluation."""

    # CLI and Runtime Setup
    args = parse_args()
    configure_runtime_environment(args.gpu)

    # Imports occur after GPU selection.
    import torch

    from src.agents.pifu_decision_models import PIFUModelPrediction
    from src.agents.pifu_safety import (
        CONFIDENCE_THRESHOLD,
        MARGIN_THRESHOLD,
        assess_pifu_safety,
    )
    from src.config.pifu_settings import (
        PIFU_BASE_MODEL,
        PIFU_CHALLENGE_PATH,
        PIFU_ID_TO_LABEL,
        PIFU_LABEL_IDS,
        PIFU_OUTPUT_DIR,
        PIFU_TEST_PATH,
    )
    from src.config.settings import settings
    from src.inference.pifu_classifier import (
        load_finetuned_model,
        score_probabilities,
    )

    # Run Directory Setup
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    output_dir = PROJECT_ROOT / "data" / "evaluations" / "pifu_inference_agent" / run_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    splits = {
        "external_test": PIFU_TEST_PATH,
        "challenge": PIFU_CHALLENGE_PATH,
    }

    # Console Header
    print("=" * 68)
    print("PIFU INFERENCE-SYSTEM BATCH EVALUATION")
    print("=" * 68)
    print(f"Model   : {PIFU_BASE_MODEL}")
    print(f"Adapter : {PIFU_OUTPUT_DIR}")
    print(f"GPU     : {args.gpu}")
    print()

    # Model Loading
    model, tokenizer = load_finetuned_model()

    all_summaries: dict[str, dict] = {}

    try:
        for split_name, split_path in splits.items():
            samples = load_samples(split_path)

            texts = [str(sample["text"]) for sample in samples]

            labels = np.asarray(
                [int(sample["label"]) for sample in samples],
                dtype=int,
            )

            print()
            print("-" * 68)
            print(f"{split_name}: {len(samples)} cases")
            print("-" * 68)

            # Probability Scoring
            probabilities = score_probabilities(
                model,
                tokenizer,
                texts,
                batch_size=args.batch_size,
                verbose=True,
            )

            predicted_indices = probabilities.argmax(axis=1)

            rows: list[dict] = []
            confidences: list[float] = []
            margins: list[float] = []
            safety_flags: list[list[str]] = []
            human_review: list[bool] = []
            predictions: list[int] = []

            # Case-Level Safety Assessment
            for sample, true_label, predicted_index, row_probs in zip(
                samples,
                labels,
                predicted_indices,
                probabilities,
                strict=True,
            ):
                true_label_int = int(true_label)

                predicted_label = int(PIFU_LABEL_IDS[int(predicted_index)])

                class_probabilities = probability_by_label(
                    row_probs,
                    PIFU_LABEL_IDS,
                )

                confidence = class_probabilities[predicted_label]

                ranked = sorted(
                    class_probabilities.values(),
                    reverse=True,
                )

                margin = float(ranked[0] - ranked[1])

                model_prediction = PIFUModelPrediction(
                    predicted_label=predicted_label,
                    predicted_class=PIFU_ID_TO_LABEL[predicted_label],
                    confidence=confidence,
                    probabilities={
                        "not_eligible": class_probabilities[0],
                        "borderline": class_probabilities[1],
                        "eligible": class_probabilities[2],
                    },
                )

                safety = assess_pifu_safety(model_prediction)

                predictions.append(predicted_label)
                confidences.append(confidence)
                margins.append(margin)
                safety_flags.append(safety.flags)
                human_review.append(safety.requires_human_review)

                rows.append(
                    build_case_prediction_row(
                        sample=sample,
                        split_name=split_name,
                        true_label=true_label_int,
                        predicted_label=predicted_label,
                        probabilities=class_probabilities,
                        confidence=confidence,
                        margin=margin,
                        safety=safety,
                        label_map=PIFU_ID_TO_LABEL,
                    )
                )

            prediction_array = np.asarray(
                predictions,
                dtype=int,
            )

            confidence_array = np.asarray(
                confidences,
                dtype=float,
            )

            margin_array = np.asarray(
                margins,
                dtype=float,
            )

            # Split-Level Metrics
            summary = calculate_metrics(
                labels=labels,
                predictions=prediction_array,
                confidences=confidence_array,
                margins=margin_array,
                safety_flags=safety_flags,
                human_review=human_review,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                margin_threshold=MARGIN_THRESHOLD,
            )

            all_summaries[split_name] = summary

            # Split Output Paths
            predictions_path = output_dir / f"{split_name}_predictions.csv"

            summary_path = output_dir / f"{split_name}_summary.json"

            # Split Output Saving
            write_predictions(
                predictions_path,
                rows,
            )

            summary_path.write_text(
                json.dumps(
                    summary,
                    indent=2,
                ),
                encoding="utf-8",
            )

            # Console Split Summary
            print()
            print(f"Accuracy            : {summary['accuracy']:.4f}")
            print(f"Macro F1            : {summary['macro_f1']:.4f}")
            print(f"NOT_ELIGIBLE recall : {summary['not_eligible_recall']:.4f}")
            print(f"ELIGIBLE precision  : {summary['eligible_precision']:.4f}")
            print(f"Unsafe eligible     : {summary['unsafe_eligible_count']}")
            print(f"Safety flag rate    : {summary['safety_flag_rate']:.4f}")

    finally:
        # Model Memory Cleanup
        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()

    # Combined Summary Outputs
    combined_json, combined_md = write_summary_outputs(
        output_dir,
        all_summaries,
    )

    # MLflow Logging
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("nhs_omop_pifu_inference")

    with mlflow.start_run(
        run_name=f"pifu_batch_evaluation_{run_id}",
        tags={
            "task": "pifu_inference_batch",
            "run_type": "quantitative_evaluation",
            "synthetic_data_only": "true",
            "human_review_required": "true",
        },
    ):
        mlflow.log_params(
            {
                "run_id": run_id,
                "model": PIFU_BASE_MODEL,
                "adapter": str(PIFU_OUTPUT_DIR),
                "gpu": args.gpu,
                "batch_size": args.batch_size,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "margin_threshold": MARGIN_THRESHOLD,
            }
        )

        mlflow.log_metrics(finite_numeric_metrics(all_summaries))

        mlflow.log_artifacts(
            str(output_dir),
            artifact_path="batch_evaluation",
        )

    # Final Console Summary
    print()
    print("=" * 68)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 68)
    print(f"Output directory: {output_dir}")
    print(f"Summary JSON    : {combined_json}")
    print(f"Summary Markdown: {combined_md}")


if __name__ == "__main__":
    main()
