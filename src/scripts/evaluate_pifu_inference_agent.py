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
import asyncio
import csv
import gc
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        "--judge-sample-size",
        type=int,
        default=24,
        help=(
            "Number of cases selected across external and challenge splits "
            "for explanation and LLM-as-a-Judge evaluation. "
            "Use 0 to skip judge evaluation."
        ),
    )

    parser.add_argument(
        "--no-llm-explanation",
        action="store_true",
        help="Disable LLM explanation generation for the judge subset.",
    )

    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM-as-a-Judge evaluation for the selected subset.",
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional evaluation run identifier.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if args.judge_sample_size < 0:
        raise ValueError("--judge-sample-size must be zero or greater.")

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


def judge_case_sort_key(
    record: dict[str, Any],
) -> tuple:
    """Return deterministic ordering for judge-case selection."""

    row = record["row"]

    return (
        float(row["confidence"]),
        float(row["top_two_margin"]),
        str(row["split"]),
        str(row.get("sample_id") or ""),
    )


def select_judge_cases(
    records: list[dict[str, Any]],
    sample_size: int,
) -> list[dict[str, Any]]:
    """Select representative difficult cases for explanation judging."""

    if sample_size <= 0 or not records:
        return []

    target = min(
        sample_size,
        len(records),
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def add_case(
        record: dict[str, Any],
        reason: str,
    ) -> bool:
        """Add one case once."""

        if len(selected) >= target:
            return False

        record_id = id(record)

        if record_id in selected_ids:
            return False

        selected_ids.add(record_id)

        selected.append(
            {
                **record,
                "selection_reason": reason,
            }
        )

        return True

    # Reserve around half of the subset for difficult cases.
    priority_limit = max(
        1,
        target // 2,
    )

    priority_cases = [
        record
        for record in records
        if (
            not record["row"]["correct"]
            or bool(record["safety"].flags)
            or int(record["row"]["predicted_label"]) == 1
        )
    ]

    priority_cases.sort(
        key=lambda record: (
            bool(record["row"]["correct"]),
            -int(record["row"]["safety_flag_count"]),
            *judge_case_sort_key(record),
        )
    )

    for record in priority_cases:
        if len(selected) >= priority_limit:
            break

        if not record["row"]["correct"]:
            reason = "misclassified_priority"

        elif record["safety"].flags:
            reason = "safety_flag_priority"

        else:
            reason = "borderline_priority"

        add_case(
            record,
            reason,
        )

    # Fill remaining slots with a balanced selection across splits and classes
    buckets: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = {}

    for record in records:
        key = (
            str(record["row"]["split"]),
            int(record["row"]["predicted_label"]),
        )

        buckets.setdefault(
            key,
            [],
        ).append(record)

    for bucket in buckets.values():
        bucket.sort(key=judge_case_sort_key)

    bucket_keys = sorted(buckets)

    positions = {key: 0 for key in bucket_keys}

    while len(selected) < target:
        made_progress = False

        for key in bucket_keys:
            bucket = buckets[key]

            position = positions[key]

            while position < len(bucket) and id(bucket[position]) in selected_ids:
                position += 1

            positions[key] = position

            if position >= len(bucket):
                continue

            record = bucket[position]

            positions[key] += 1

            if add_case(
                record,
                "split_class_representative",
            ):
                made_progress = True

            if len(selected) >= target:
                break

        if not made_progress:
            break

    # Fill remaining slots with low-confidence cases if needed
    if len(selected) < target:
        remaining = sorted(
            records,
            key=judge_case_sort_key,
        )

        for record in remaining:
            add_case(
                record,
                "low_confidence_fill",
            )

            if len(selected) >= target:
                break

    return selected


def explanation_available(
    explanation,
) -> bool:
    """Return whether real explanation generation succeeded."""

    return (
        explanation.clinical_summary
        != "Automated narrative explanation was unavailable."
    )


def judge_available(
    judge,
) -> bool:
    """Return whether the real LLM judge ran successfully."""

    return not judge.judge_summary.startswith(
        "LLM-as-a-Judge evaluation was unavailable:"
    )


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


def finite_judge_metrics(
    summary: dict | None,
) -> dict[str, float]:
    """Return finite judge summary metrics for MLflow."""

    if not summary:
        return {}

    metrics: dict[str, float] = {}

    for key, value in summary.items():
        if isinstance(value, (int, float)) and value is not None:
            number = float(value)

            if math.isfinite(number):
                metrics[f"judge_{key}"] = number

    return metrics


def write_summary_outputs(
    output_dir: Path,
    summaries: dict[str, dict],
    judge_summary: dict | None,
) -> tuple[Path, Path]:
    """Write combined JSON and Markdown summaries."""

    # Combined Output Paths
    combined_json = output_dir / "combined_summary.json"
    combined_md = output_dir / "combined_summary.md"

    # Combined JSON Payload
    combined_payload = {
        "classification_and_safety": (summaries),
        "explanation_and_llm_judge": (judge_summary),
    }

    # Combined JSON Saving
    combined_json.write_text(
        json.dumps(
            combined_payload,
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


async def evaluate_judge_subset(
    selected_cases: list[dict[str, Any]],
    *,
    use_llm_explanation: bool,
    use_llm_judge: bool,
) -> tuple[list[dict], dict]:
    """Generate explanations and judge them for selected batch cases."""

    from src.agents.pifu_inference_agent import (
        create_pifu_explanation,
    )
    from src.agents.pifu_llm_judge import (
        fallback_judge,
        judge_pifu_explanation,
    )

    judge_rows: list[dict] = []

    print()
    print("=" * 68)
    print("PIFU EXPLANATION + LLM-AS-A-JUDGE")
    print("=" * 68)

    print(f"Selected cases: {len(selected_cases)}")

    for index, record in enumerate(
        selected_cases,
        start=1,
    ):
        sample = record["sample"]
        prediction = record["prediction"]
        safety = record["safety"]
        row = record["row"]

        print(
            f"[{index:02d}/{len(selected_cases):02d}] "
            f"{row['split']} | "
            f"{row['sample_id']} | "
            f"{row['predicted_class']}"
        )

        # Explanation
        explanation = await create_pifu_explanation(
            text=str(sample["text"]),
            prediction=prediction,
            safety=safety,
            use_llm=use_llm_explanation,
        )

        explanation_ok = explanation_available(explanation)

        # Judge
        if explanation_ok:
            judge = await judge_pifu_explanation(
                text=str(sample["text"]),
                prediction=prediction,
                safety=safety,
                explanation=explanation,
                use_llm=use_llm_judge,
            )

        else:
            judge = fallback_judge(
                "Explanation generation was unavailable, "
                "so explanation quality was not submitted "
                "to the LLM judge."
            )

        judge_ok = judge_available(judge)

        # Save case-level output
        judge_rows.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "text_sha256": row["text_sha256"],
                "selection_reason": record["selection_reason"],
                "true_label": row["true_label"],
                "true_class": row["true_class"],
                "predicted_label": row["predicted_label"],
                "predicted_class": row["predicted_class"],
                "correct": row["correct"],
                "confidence": row["confidence"],
                "top_two_margin": row["top_two_margin"],
                "safety_flags": row["safety_flags"],
                "explanation_available": (explanation_ok),
                "clinical_summary": (explanation.clinical_summary),
                "evidence_summary": " | ".join(explanation.evidence_summary),
                "limitations": " | ".join(explanation.limitations),
                "judge_available": judge_ok,
                "judge_faithfulness": (judge.explanation_faithfulness),
                "judge_grounding": (judge.evidence_grounding),
                "judge_prediction_consistency": (judge.prediction_consistency),
                "judge_safety": (judge.safety_compliance),
                "judge_hallucination": (judge.hallucination_detected),
                "unsupported_claim_count": len(judge.unsupported_claims),
                "unsupported_claims": " | ".join(judge.unsupported_claims),
                "judge_pass": (judge.judge_pass),
                "judge_summary": (judge.judge_summary),
            }
        )

    # Only successful judge calls contribute to judge-quality metrics.
    valid_judge_rows = [row for row in judge_rows if row["judge_available"]]

    explanation_count = sum(bool(row["explanation_available"]) for row in judge_rows)

    judge_count = len(valid_judge_rows)

    total = len(judge_rows)

    judge_summary = {
        "selected_case_count": total,
        "explanation_available_count": explanation_count,
        "explanation_coverage_rate": (
            float(explanation_count / total) if total else None
        ),
        "judge_available_count": judge_count,
        "judge_unavailable_count": (total - judge_count),
        "judge_coverage_rate": (float(judge_count / total) if total else None),
        "judge_pass_rate": (
            float(np.mean([row["judge_pass"] for row in valid_judge_rows]))
            if valid_judge_rows
            else None
        ),
        "judge_hallucination_rate": (
            float(np.mean([row["judge_hallucination"] for row in valid_judge_rows]))
            if valid_judge_rows
            else None
        ),
        "unsupported_claim_rate": (
            float(
                np.mean(
                    [row["unsupported_claim_count"] > 0 for row in valid_judge_rows]
                )
            )
            if valid_judge_rows
            else None
        ),
        "mean_judge_faithfulness": safe_mean(
            [float(row["judge_faithfulness"]) for row in valid_judge_rows]
        ),
        "mean_judge_grounding": safe_mean(
            [float(row["judge_grounding"]) for row in valid_judge_rows]
        ),
        "mean_judge_prediction_consistency": (
            safe_mean(
                [float(row["judge_prediction_consistency"]) for row in valid_judge_rows]
            )
        ),
        "mean_judge_safety": safe_mean(
            [float(row["judge_safety"]) for row in valid_judge_rows]
        ),
    }

    return (
        judge_rows,
        judge_summary,
    )


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

    all_case_records: list[dict[str, Any]] = []

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

                case_row = build_case_prediction_row(
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

                rows.append(case_row)

                all_case_records.append(
                    {
                        "sample": sample,
                        "prediction": model_prediction,
                        "safety": safety,
                        "row": case_row,
                    }
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

    # Judge Subset Selection and Evaluation
    selected_judge_cases = select_judge_cases(
        all_case_records,
        args.judge_sample_size,
    )

    judge_rows: list[dict] = []
    judge_summary: dict | None = None

    if selected_judge_cases:
        judge_rows, judge_summary = asyncio.run(
            evaluate_judge_subset(
                selected_judge_cases,
                use_llm_explanation=(not args.no_llm_explanation),
                use_llm_judge=(not args.no_llm_judge),
            )
        )

        judge_cases_path = output_dir / "llm_judge_cases.csv"

        judge_summary_path = output_dir / "llm_judge_summary.json"

        write_predictions(
            judge_cases_path,
            judge_rows,
        )

        judge_summary_path.write_text(
            json.dumps(
                judge_summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("LLM-AS-A-JUDGE SUMMARY")
        print("-" * 68)

        print(f"Selected cases      : {judge_summary['selected_case_count']}")

        print(f"Judge available     : {judge_summary['judge_available_count']}")

        if judge_summary["judge_pass_rate"] is not None:
            print(f"Judge pass rate     : {judge_summary['judge_pass_rate']:.4f}")

            print(
                f"Hallucination rate  : {judge_summary['judge_hallucination_rate']:.4f}"
            )

            print(
                f"Mean faithfulness   : {judge_summary['mean_judge_faithfulness']:.4f}"
            )

            print(f"Mean grounding      : {judge_summary['mean_judge_grounding']:.4f}")

            print(
                "Mean consistency    : "
                f"{judge_summary['mean_judge_prediction_consistency']:.4f}"
            )

            print(f"Mean safety         : {judge_summary['mean_judge_safety']:.4f}")

    # Combined Summary Outputs
    combined_json, combined_md = write_summary_outputs(
        output_dir,
        all_summaries,
        judge_summary,
    )

    # MLflow Logging
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("nhs_omop_pifu_inference")

    with mlflow.start_run(
        run_name=f"pifu_batch_evaluation_{run_id}",
        tags={
            "task": "pifu_inference_batch",
            "run_type": "quantitative_plus_judge_evaluation",
            "synthetic_data_only": "true",
            "human_review_required": "true",
            "llm_explanation_enabled": "explanation_quality",
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
                "judge_sample_size_requested": args.judge_sample_size,
                "judge_sample_size_selected": len(selected_judge_cases),
                "use_llm_explanation": not args.no_llm_explanation,
                "use_llm_judge": not args.no_llm_judge,
            }
        )

        mlflow_metrics = finite_numeric_metrics(all_summaries)
        mlflow_metrics.update(finite_judge_metrics(judge_summary))
        mlflow.log_metrics(mlflow_metrics)

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

    if selected_judge_cases:
        print(f"Judge cases CSV : {output_dir / 'llm_judge_cases.csv'}")
        print(f"Judge summary   : {output_dir / 'llm_judge_summary.json'}")


if __name__ == "__main__":
    main()
