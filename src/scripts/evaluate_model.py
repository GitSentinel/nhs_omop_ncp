"""
Validation-Calibrated Model Evaluation

Compares the base model and fine-tuned adapter using exact label-sequence log probabilities. Thresholds are selected on validation data and applied once to the held-out test split.

Run:
uv run --extra finetune python src/scripts/evaluate_model.py
"""

import gc
import json
import sys
from collections import Counter
from pathlib import Path

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Experiment Tracking
import mlflow

# Data and Numerical Libraries
import numpy as np
import torch

# PEFT Model Loading
from peft import PeftModel

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Model Loading Components
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Project Configuration
from config import (
    BASE_MODEL,
    BNB_CONFIG_PARAMS,
    DATASET_PATH,
    FINETUNE_OUTPUT_DIR,
    FINETUNE_PROMPT_TEMPLATE as PROMPT_TEMPLATE,
    MAX_LENGTH,
    MLFLOW_EXPERIMENT_FINETUNE,
    PIFU_MIN_RECALL,
)

from src.config.settings import settings

# Quantisation Configuration
BNB_CONFIG = BitsAndBytesConfig(**BNB_CONFIG_PARAMS)

# Evaluation Configuration
DEFAULT_THRESHOLD = 0.50
EVAL_BATCH_SIZE = 4


def setup_tokenizer(model_path: str | Path):
    """Load a tokenizer and ensure valid padding settings."""

    # Tokeniser Loading
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Padding Token Setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    return tokenizer


def get_model_device(model) -> torch.device:
    """Return the first real device used by the model."""

    # Device Detection
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device

    return torch.device("cuda:0")


def get_label_token_ids(tokenizer) -> dict[int, list[int]]:
    """Return token IDs for the two class-label candidates."""

    # Label Token Construction
    label_token_ids = {
        label: tokenizer.encode(
            f" {label}",
            add_special_tokens=False,
        )
        for label in (0, 1)
    }

    # Label Token Validation
    for label, token_ids in label_token_ids.items():
        if not token_ids:
            raise ValueError(f"Label {label} produced no token IDs.")

    if label_token_ids[0] == label_token_ids[1]:
        raise ValueError(
            "Labels 0 and 1 produced identical token sequences: "
            f"{label_token_ids[0]}"
        )

    return label_token_ids


def build_prompt_ids(
    tokenizer,
    text: str,
    max_label_length: int,
) -> list[int]:
    """Build prompt token IDs while preserving space for label tokens."""

    # Prompt Formatting
    prompt = PROMPT_TEMPLATE.format(text=str(text))

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    # Prompt Length Control
    max_prompt_length = MAX_LENGTH - max_label_length

    if max_prompt_length < 1:
        raise ValueError(
            f"MAX_LENGTH={MAX_LENGTH} is too small for the label tokens."
        )

    prompt_ids = prompt_ids[:max_prompt_length]

    if not prompt_ids:
        raise ValueError("The formatted prompt produced no token IDs.")

    return prompt_ids


def score_label_candidates(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = EVAL_BATCH_SIZE,
) -> list[float]:
    """Score labels 0 and 1 using exact candidate log probabilities."""

    # Evaluation Setup
    model.eval()

    device = get_model_device(model)
    label_token_ids = get_label_token_ids(tokenizer)
    max_label_length = max(
        len(token_ids)
        for token_ids in label_token_ids.values()
    )

    probabilities = []

    print(
        "Label token sequences: "
        f"0={label_token_ids[0]}, 1={label_token_ids[1]}"
    )

    # Batch Scoring Loop
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        sequences = []
        metadata = []

        # Candidate Sequence Construction
        for text in batch_texts:
            prompt_ids = build_prompt_ids(
                tokenizer,
                text,
                max_label_length,
            )

            for label in (0, 1):
                candidate_ids = label_token_ids[label]
                sequences.append(prompt_ids + candidate_ids)
                metadata.append((len(prompt_ids), candidate_ids))

        # Tensor Construction
        max_sequence_length = max(
            len(sequence)
            for sequence in sequences
        )

        input_ids = torch.full(
            (len(sequences), max_sequence_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.zeros_like(input_ids)

        for row_index, sequence in enumerate(sequences):
            sequence_length = len(sequence)

            input_ids[row_index, :sequence_length] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )

            attention_mask[row_index, :sequence_length] = 1

        # Model Forward Pass
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        candidate_scores = []

        # Candidate Log-Probability Calculation
        for row_index, (prompt_length, candidate_ids) in enumerate(metadata):
            score = torch.zeros(
                (),
                dtype=torch.float32,
                device=device,
            )

            for offset, token_id in enumerate(candidate_ids):
                logits_position = prompt_length + offset - 1

                token_logits = outputs.logits[
                    row_index,
                    logits_position,
                    :,
                ].float()

                score = score + torch.log_softmax(
                    token_logits,
                    dim=-1,
                )[token_id]

            candidate_scores.append(score)

        # Treatment Probability Calculation
        paired_scores = torch.stack(candidate_scores).reshape(
            len(batch_texts),
            2,
        )

        batch_probabilities = torch.softmax(
            paired_scores,
            dim=-1,
        )[:, 1]

        probabilities.extend(batch_probabilities.cpu().tolist())

        completed = min(batch_start + len(batch_texts), len(texts))

        if completed % 50 == 0 or completed == len(texts):
            print(f"  {completed:,}/{len(texts):,} evaluated...")

        # Batch Memory Cleanup
        del outputs
        del input_ids
        del attention_mask
        del paired_scores
        del batch_probabilities

    return probabilities


def compute_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: list[float],
) -> dict:
    """Compute classification metrics and probability diagnostics."""

    # Array Conversion
    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)

    # Classification Report
    report = classification_report(
        labels_array,
        predictions_array,
        labels=[0, 1],
        target_names=[
            "routine_followup",
            "treatment_event",
        ],
        digits=4,
        zero_division=0,
    )

    # Confusion Matrix
    matrix = confusion_matrix(
        labels_array,
        predictions_array,
        labels=[0, 1],
    )

    # Ranking Metrics
    roc_auc = float("nan")
    pr_auc = float("nan")

    if len(np.unique(labels_array)) == 2:
        roc_auc = roc_auc_score(labels_array, probabilities_array)
        pr_auc = average_precision_score(labels_array, probabilities_array)

    return {
        "f1": f1_score(
            labels_array,
            predictions_array,
            average="binary",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            labels_array,
            predictions_array,
            average="macro",
            zero_division=0,
        ),
        "accuracy": accuracy_score(labels_array, predictions_array),
        "balanced_accuracy": balanced_accuracy_score(
            labels_array,
            predictions_array,
        ),
        "precision": precision_score(
            labels_array,
            predictions_array,
            zero_division=0,
        ),
        "recall": recall_score(
            labels_array,
            predictions_array,
            zero_division=0,
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": matrix.tolist(),
        "prediction_counts": dict(Counter(predictions)),
        "probability_min": float(probabilities_array.min()),
        "probability_median": float(np.median(probabilities_array)),
        "probability_max": float(probabilities_array.max()),
        "report": report,
    }


def unpack_split(
    data: dict,
    split_name: str,
) -> tuple[list[str], list[int], list[dict]]:
    """Extract and validate one dataset split."""

    # Split Retrieval
    samples = data.get(split_name)

    if not isinstance(samples, list):
        raise ValueError(
            f"The dataset must contain a list named {split_name!r}."
        )

    if not samples:
        raise ValueError(f"The split {split_name!r} is empty.")

    # Text and Label Extraction
    texts = [str(example["text"]) for example in samples]
    labels = [int(example["label"]) for example in samples]

    # Label Validation
    invalid_labels = sorted(set(labels) - {0, 1})

    if invalid_labels:
        raise ValueError(
            f"Unexpected labels in {split_name}: {invalid_labels}"
        )

    if len(set(labels)) < 2:
        raise ValueError(
            f"The split {split_name!r} must contain both labels 0 and 1."
        )

    return texts, labels, samples


def load_evaluation_splits() -> tuple[
    list[str],
    list[int],
    list[dict],
    list[str],
    list[int],
    list[dict],
]:
    """Load validation and test splits from the labelled dataset."""

    # Dataset Existence Check
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}"
        )

    # Dataset Loading
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Split Extraction
    validation = unpack_split(data, "validation")
    test = unpack_split(data, "test")

    return (*validation, *test)


def select_threshold(
    labels: list[int],
    probabilities: list[float],
    minimum_recall: float = PIFU_MIN_RECALL,
) -> tuple[float, float, float]:
    """Select a threshold on validation data."""

    # Array Conversion
    labels_array = np.asarray(labels, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)

    # Precision-Recall Curve
    precision_values, recall_values, thresholds = precision_recall_curve(
        labels_array,
        probabilities_array,
    )

    if len(thresholds) == 0:
        return DEFAULT_THRESHOLD, 0.0, 0.0

    precision_values = precision_values[:-1]
    recall_values = recall_values[:-1]

    # Minimum Precision Operating Point
    # valid_indices = np.flatnonzero(
    #     precision_values >= minimum_precision
    # )

    valid_indices = np.flatnonzero(
        recall_values >= minimum_recall
    )

    if len(valid_indices) > 0:
        # best_index = valid_indices[
        #     np.argmax(recall_values[valid_indices])
        # ]

        best_index = valid_indices[
            np.argmax(precision_values[valid_indices])
        ]

    else:
        # F2 Fallback
        denominator = (
            4 * precision_values
            + recall_values
            + 1e-12
        )

        f2_values = (
            5 * precision_values * recall_values
            / denominator
        )

        best_index = int(np.nanargmax(f2_values))

    return (
        float(thresholds[best_index]),
        float(precision_values[best_index]),
        float(recall_values[best_index]),
    )


def apply_threshold(
    probabilities: list[float],
    threshold: float,
) -> list[int]:
    """Convert probabilities into binary predictions."""

    # Threshold Application
    return [
        int(probability >= threshold)
        for probability in probabilities
    ]


def load_base_model():
    """Load the base model without a LoRA adapter."""

    # Base Model Loading
    print(f"Loading base model: {BASE_MODEL}")

    tokenizer = setup_tokenizer(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    return model, tokenizer


def load_finetuned_model():
    """Load the base model with the saved LoRA adapter."""

    # Adapter Configuration Check
    adapter_config_path = FINETUNE_OUTPUT_DIR / "adapter_config.json"

    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"Adapter config not found: {adapter_config_path}. "
            "Run finetune.py first."
        )

    # Adapter Base Model Validation
    with open(adapter_config_path, "r", encoding="utf-8") as file:
        adapter_config = json.load(file)

    adapter_base_model = adapter_config.get(
        "base_model_name_or_path",
        BASE_MODEL,
    )

    if adapter_base_model != BASE_MODEL:
        raise ValueError(
            "The configured base model does not match the saved adapter. "
            f"config.py={BASE_MODEL!r}, adapter={adapter_base_model!r}"
        )

    print(f"Loading fine-tuned model from: {FINETUNE_OUTPUT_DIR}")

    # Tokeniser Loading
    try:
        tokenizer = setup_tokenizer(FINETUNE_OUTPUT_DIR)
    except OSError:
        tokenizer = setup_tokenizer(BASE_MODEL)

    # Base Model Loading
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    # Adapter Loading
    model = PeftModel.from_pretrained(
        base_model,
        str(FINETUNE_OUTPUT_DIR),
        is_trainable=False,
    )

    model.eval()

    return model, tokenizer


def clear_cuda_memory() -> None:
    """Clear unused CPU and GPU memory."""

    # Memory Cleanup
    gc.collect()
    torch.cuda.empty_cache()


def print_metric_summary(title: str, metrics: dict) -> None:
    """Print classification metrics and diagnostics."""

    # Confusion Matrix Values
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]

    # Metric Summary
    print(f"\n{title} Results:")
    print(f"  F1                : {metrics['f1']:.4f}")
    print(f"  Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  ROC AUC           : {metrics['roc_auc']:.4f}")
    print(f"  PR AUC            : {metrics['pr_auc']:.4f}")
    print(f"  Predictions       : {metrics['prediction_counts']}")
    print(
        "  Probability range : "
        f"min={metrics['probability_min']:.4f}, "
        f"median={metrics['probability_median']:.4f}, "
        f"max={metrics['probability_max']:.4f}"
    )
    print(f"  Confusion matrix  : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(metrics["report"])


def build_comparison_summary(
    base_metrics: dict,
    finetuned_metrics: dict,
    base_threshold: float,
    finetuned_threshold: float,
) -> str:
    """Create a text summary comparing base and fine-tuned performance."""

    # Comparison Metric Setup
    metric_names = [
        "f1",
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
    ]

    lines = [
        f"Base Model vs Fine-Tuned - {BASE_MODEL}",
        "=" * 60,
    ]

    # Comparison Lines
    for metric_name in metric_names:
        base_value = base_metrics[metric_name]
        finetuned_value = finetuned_metrics[metric_name]

        lines.append(
            f"{metric_name:<18} "
            f"{base_value:.4f} -> {finetuned_value:.4f} "
            f"({finetuned_value - base_value:+.4f})"
        )

    lines.extend([
        "",
        f"Dataset: {DATASET_PATH}",
        f"Adapter: {FINETUNE_OUTPUT_DIR}",
        f"Minimum validation recall: {PIFU_MIN_RECALL}"
        f"Base threshold: {base_threshold:.6f}",
        f"Fine-tuned threshold: {finetuned_threshold:.6f}",
        f"Evaluation batch size: {EVAL_BATCH_SIZE}",
        "Text input: full stored clinic letter with token truncation",
    ])

    return "\n".join(lines)


def log_metrics(prefix: str, metrics: dict) -> None:
    """Log finite scalar metrics to MLflow with a prefix."""

    # Metric Selection
    metric_names = [
        "f1",
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
    ]

    # MLflow Metric Logging
    mlflow.log_metrics({
        f"{prefix}_{metric_name}": metrics[metric_name]
        for metric_name in metric_names
        if np.isfinite(metrics[metric_name])
    })


def run_evaluation() -> None:
    """Evaluate base and fine-tuned models using validation thresholds."""

    # GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation run.")

    # Evaluation Split Loading
    (
        validation_texts,
        validation_labels,
        validation_samples,
        test_texts,
        test_labels,
        test_samples,
    ) = load_evaluation_splits()

    # Run Header
    print("=" * 65)
    print("NHS OMOP MCP Model Evaluation")
    print("=" * 65)
    print(f"Base model          : {BASE_MODEL}")
    print(f"Adapter             : {FINETUNE_OUTPUT_DIR}")
    print(f"Dataset             : {DATASET_PATH}")
    print(f"Validation samples  : {len(validation_samples):,}")
    print(f"Test samples        : {len(test_samples):,}")
    print(f"Minimum recall      : {PIFU_MIN_RECALL:.2f}")
    print(f"Evaluation batch    : {EVAL_BATCH_SIZE}")
    print("Text input          : full stored letter")
    print()

    # MLflow Setup
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_FINETUNE)

    # Test Distribution Summary
    print("Test set distribution:")
    print(f"  routine_followup : {test_labels.count(0):,}")
    print(f"  treatment_event  : {test_labels.count(1):,}")
    print()

    # Base Model Evaluation
    print("=" * 40)
    print("Evaluating BASE model without adapter...")
    print("=" * 40)

    base_model, base_tokenizer = load_base_model()

    base_validation_probabilities = score_label_candidates(
        base_model,
        base_tokenizer,
        validation_texts,
    )

    (
        base_threshold,
        base_validation_precision,
        base_validation_recall,
    ) = select_threshold(
        validation_labels,
        base_validation_probabilities,
    )

    print(
        f"Base validation operating point: "
        f"threshold={base_threshold:.6f}, "
        f"precision={base_validation_precision:.4f}, "
        f"recall={base_validation_recall:.4f}"
    )

    base_test_probabilities = score_label_candidates(
        base_model,
        base_tokenizer,
        test_texts,
    )

    base_predictions = apply_threshold(
        base_test_probabilities,
        base_threshold,
    )

    base_metrics = compute_metrics(
        test_labels,
        base_predictions,
        base_test_probabilities,
    )

    print_metric_summary("BASE MODEL", base_metrics)

    del base_model
    del base_tokenizer
    clear_cuda_memory()

    print("Base model unloaded.\n")

    # Fine-Tuned Model Evaluation
    print("=" * 40)
    print("Evaluating FINE-TUNED model with adapter...")
    print("=" * 40)

    finetuned_model, finetuned_tokenizer = load_finetuned_model()

    finetuned_validation_probabilities = score_label_candidates(
        finetuned_model,
        finetuned_tokenizer,
        validation_texts,
    )

    (
        finetuned_threshold,
        finetuned_validation_precision,
        finetuned_validation_recall,
    ) = select_threshold(
        validation_labels,
        finetuned_validation_probabilities,
    )

    print(
        f"Fine-tuned validation operating point: "
        f"threshold={finetuned_threshold:.6f}, "
        f"precision={finetuned_validation_precision:.4f}, "
        f"recall={finetuned_validation_recall:.4f}"
    )

    finetuned_test_probabilities = score_label_candidates(
        finetuned_model,
        finetuned_tokenizer,
        test_texts,
    )

    finetuned_predictions = apply_threshold(
        finetuned_test_probabilities,
        finetuned_threshold,
    )

    finetuned_metrics = compute_metrics(
        test_labels,
        finetuned_predictions,
        finetuned_test_probabilities,
    )

    print_metric_summary("FINE-TUNED MODEL", finetuned_metrics)

    del finetuned_model
    del finetuned_tokenizer
    clear_cuda_memory()

    print("Fine-tuned model unloaded.\n")

    # Comparison Summary
    print("=" * 65)
    print("COMPARISON: Base vs Fine-Tuned")
    print("=" * 65)
    print(
        f"{'Metric':<20} "
        f"{'Base':>10} "
        f"{'Fine-Tuned':>12} "
        f"{'Delta':>10}"
    )
    print("-" * 56)

    comparison_metrics = [
        "f1",
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
    ]

    for metric_name in comparison_metrics:
        base_value = base_metrics[metric_name]
        finetuned_value = finetuned_metrics[metric_name]

        print(
            f"{metric_name:<20} "
            f"{base_value:>10.4f} "
            f"{finetuned_value:>12.4f} "
            f"{finetuned_value - base_value:>+10.4f}"
        )

    # MLflow Logging
    with mlflow.start_run(run_name="evaluation_base_vs_finetuned"):
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "adapter_path": str(FINETUNE_OUTPUT_DIR),
            "n_validation_samples": len(validation_samples),
            "n_test_samples": len(test_samples),
            "dataset": str(DATASET_PATH),
            "minimum_validation_recall": PIFU_MIN_RECALL,
            "base_threshold": base_threshold,
            "finetuned_threshold": finetuned_threshold,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "scoring_method": "exact_label_sequence_log_probability",
            # "threshold_selection": "max_recall_at_minimum_precision",
            "threshold_selection": "max_precision_at_minimum_recall",
        })

        mlflow.log_metrics({
            "base_validation_precision": base_validation_precision,
            "base_validation_recall": base_validation_recall,
            "finetuned_validation_precision": (
                finetuned_validation_precision
            ),
            "finetuned_validation_recall": (
                finetuned_validation_recall
            ),
        })

        log_metrics("base", base_metrics)
        log_metrics("ft", finetuned_metrics)

        for metric_name in comparison_metrics:
            delta = finetuned_metrics[metric_name] - base_metrics[metric_name]

            if np.isfinite(delta):
                mlflow.log_metric(f"{metric_name}_delta", delta)

        mlflow.log_text(
            base_metrics["report"],
            "base_model_report.txt",
        )

        mlflow.log_text(
            finetuned_metrics["report"],
            "finetuned_model_report.txt",
        )

        mlflow.log_text(
            PROMPT_TEMPLATE,
            "prompt_template.txt",
        )

        mlflow.log_text(
            build_comparison_summary(
                base_metrics,
                finetuned_metrics,
                base_threshold,
                finetuned_threshold,
            ),
            "comparison_summary.txt",
        )


if __name__ == "__main__":
    run_evaluation()