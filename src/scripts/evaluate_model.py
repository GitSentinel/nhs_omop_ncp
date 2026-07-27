import gc
import json
import sys
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
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Model Loading Components
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Project Configuration
from config import (
    DATASET_PATH,
    FINETUNE_OUTPUT_DIR,
    BASE_MODEL,
    MAX_LENGTH,
    BNB_CONFIG_PARAMS,
    FINETUNE_PROMPT_TEMPLATE as PROMPT_TEMPLATE,
    MLFLOW_EXPERIMENT_FINETUNE,
)

from src.config.settings import settings

# Quantisation Configuration
BNB_CONFIG = BitsAndBytesConfig(**BNB_CONFIG_PARAMS)

# Prediction Configuration
DEFAULT_THRESHOLD = 0.30


def setup_tokenizer(model_path: str | Path):
    # Tokeniser Loading
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Padding Token Setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_label_token_ids(tokenizer, label: str) -> list[int]:
    # Candidate Label Forms
    candidates = [
        label,
        f" {label}",
        f"\n{label}",
    ]

    token_ids = []

    for candidate in candidates:
        encoded = tokenizer.encode(
            candidate,
            add_special_tokens=False,
        )

        if encoded:
            token_ids.append(encoded[0])

    return sorted(set(token_ids))


def get_label_probability(
    first_token_scores: torch.Tensor,
    tokenizer,
) -> float:
    # Token Probability Calculation
    probs = torch.softmax(first_token_scores, dim=-1)

    token_0_ids = get_label_token_ids(tokenizer, "0")
    token_1_ids = get_label_token_ids(tokenizer, "1")

    prob_0 = sum(probs[token_id].item() for token_id in token_0_ids)
    prob_1 = sum(probs[token_id].item() for token_id in token_1_ids)

    return prob_1 / (prob_0 + prob_1 + 1e-9)


def parse_generated_label(generated_text: str) -> int:
    """Convert generated model text into a binary label."""

    # Label Parsing
    generated_text = str(generated_text).strip()

    if generated_text.startswith("1"):
        return 1

    return 0


def predict_with_threshold(
    model,
    tokenizer,
    texts: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[int], list[float]]:
    # Prediction Setup
    model.eval()

    predictions = []
    probabilities = []

    # Prediction Loop
    for index, text in enumerate(texts):
        prompt = PROMPT_TEMPLATE.format(text=text[:600])

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH - 10,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated = tokenizer.decode(
            output.sequences[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Threshold-Based Prediction
        if getattr(output, "scores", None):
            prob_1 = get_label_probability(
                output.scores[0][0],
                tokenizer,
            )

            prediction = 1 if prob_1 > threshold else 0

        else:
            prob_1 = float(parse_generated_label(generated))
            prediction = parse_generated_label(generated)

        predictions.append(prediction)
        probabilities.append(prob_1)

        if (index + 1) % 50 == 0:
            print(f"  {index + 1:,}/{len(texts):,} evaluated...")

    return predictions, probabilities


def compute_metrics(labels: list[int], predictions: list[int]) -> dict:
    # Metric Calculation
    labels_arr = np.array(labels)
    preds_arr = np.array(predictions)

    report = classification_report(
        labels_arr,
        preds_arr,
        target_names=[
            "routine_followup",
            "treatment_event",
        ],
        digits=4,
        zero_division=0,
    )

    return {
        "f1": f1_score(labels_arr, preds_arr, average="binary", zero_division=0),
        "accuracy": accuracy_score(labels_arr, preds_arr),
        "precision": precision_score(labels_arr, preds_arr, zero_division=0),
        "recall": recall_score(labels_arr, preds_arr, zero_division=0),
        "report": report,
    }


def load_test_split() -> tuple[list[str], list[int], list[dict]]:
    # Dataset Existence Check
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}"
        )

    # Dataset Loading
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Test Split Extraction
    test_samples = data["test"]

    texts = [example["text"] for example in test_samples]
    labels = [int(example["label"]) for example in test_samples]

    return texts, labels, test_samples


def load_base_model():
    # Base Model Loading
    print(f"Loading base model: {BASE_MODEL}")

    tokenizer = setup_tokenizer(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BNB_CONFIG,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    return model, tokenizer


def load_finetuned_model():
    # Adapter Path Check
    adapter_config = FINETUNE_OUTPUT_DIR / "adapter_config.json"

    if not adapter_config.exists():
        raise FileNotFoundError(
            f"Adapter config not found: {adapter_config}. "
            "Run finetune.py first."
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
        trust_remote_code=True,
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    # Adapter Loading
    model = PeftModel.from_pretrained(
        base_model,
        str(FINETUNE_OUTPUT_DIR),
    )

    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    # Memory Cleanup
    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()


def print_metric_summary(title: str, metrics: dict) -> None:
    # Metric Summary
    print(f"\n{title} Results:")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(metrics["report"])


def build_comparison_summary(
    base_metrics: dict,
    ft_metrics: dict,
) -> str:
    # Comparison Text
    return "\n".join([
        f"Base Model vs Fine-Tuned - {BASE_MODEL}",
        "=" * 50,
        f"F1:        {base_metrics['f1']:.4f} -> {ft_metrics['f1']:.4f} "
        f"({ft_metrics['f1'] - base_metrics['f1']:+.4f})",
        f"Accuracy:  {base_metrics['accuracy']:.4f} -> {ft_metrics['accuracy']:.4f} "
        f"({ft_metrics['accuracy'] - base_metrics['accuracy']:+.4f})",
        f"Precision: {base_metrics['precision']:.4f} -> {ft_metrics['precision']:.4f} "
        f"({ft_metrics['precision'] - base_metrics['precision']:+.4f})",
        f"Recall:    {base_metrics['recall']:.4f} -> {ft_metrics['recall']:.4f} "
        f"({ft_metrics['recall'] - base_metrics['recall']:+.4f})",
        "",
        f"Dataset: {DATASET_PATH}",
        f"Adapter: {FINETUNE_OUTPUT_DIR}",
        f"Threshold: {DEFAULT_THRESHOLD}",
    ])


def run_evaluation() -> None:
    # GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this 4-bit evaluation run.")

    # Test Dataset Loading
    texts, labels, test_samples = load_test_split()

    # Run Header
    print("=" * 65)
    print("NHS OMOP MCP Model Evaluation")
    print("=" * 65)
    print(f"Base model  : {BASE_MODEL}")
    print(f"Adapter     : {FINETUNE_OUTPUT_DIR}")
    print(f"Dataset     : {DATASET_PATH}")
    print(f"Test samples: {len(test_samples):,}")
    print(f"Threshold   : {DEFAULT_THRESHOLD}")
    print()

    # MLflow Setup
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_FINETUNE)

    # Test Distribution Summary
    print("Test set distribution:")
    print(f"  routine_followup : {labels.count(0):,}")
    print(f"  treatment_event  : {labels.count(1):,}")
    print()

    # Base Model Evaluation
    print("=" * 40)
    print("Evaluating BASE model without adapter...")
    print("=" * 40)

    base_model, base_tokenizer = load_base_model()

    base_predictions, base_probabilities = predict_with_threshold(
        base_model,
        base_tokenizer,
        texts,
    )

    base_metrics = compute_metrics(labels, base_predictions)

    print_metric_summary("BASE MODEL", base_metrics)

    unload_model(base_model, base_tokenizer)
    print("Base model unloaded.\n")

    # Fine-Tuned Model Evaluation
    print("=" * 40)
    print("Evaluating FINE-TUNED model with adapter...")
    print("=" * 40)

    ft_model, ft_tokenizer = load_finetuned_model()

    ft_predictions, ft_probabilities = predict_with_threshold(
        ft_model,
        ft_tokenizer,
        texts,
    )

    ft_metrics = compute_metrics(labels, ft_predictions)

    print_metric_summary("FINE-TUNED MODEL", ft_metrics)

    unload_model(ft_model, ft_tokenizer)
    print("Fine-tuned model unloaded.\n")

    # Comparison Summary
    print("=" * 65)
    print("COMPARISON: Base vs Fine-Tuned")
    print("=" * 65)
    print(f"{'Metric':<15} {'Base':>10} {'Fine-Tuned':>12} {'Delta':>10}")
    print("-" * 55)

    for metric in ["f1", "accuracy", "precision", "recall"]:
        base_value = base_metrics[metric]
        ft_value = ft_metrics[metric]
        delta = ft_value - base_value

        print(
            f"{metric:<15} "
            f"{base_value:>10.4f} "
            f"{ft_value:>12.4f} "
            f"{delta:>+10.4f}"
        )

    # MLflow Logging
    with mlflow.start_run(run_name="evaluation_base_vs_finetuned"):
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "adapter_path": str(FINETUNE_OUTPUT_DIR),
            "n_eval_samples": len(test_samples),
            "dataset": str(DATASET_PATH),
            "threshold": DEFAULT_THRESHOLD,
        })

        mlflow.log_metrics({
            "base_f1": base_metrics["f1"],
            "base_accuracy": base_metrics["accuracy"],
            "base_precision": base_metrics["precision"],
            "base_recall": base_metrics["recall"],
            "ft_f1": ft_metrics["f1"],
            "ft_accuracy": ft_metrics["accuracy"],
            "ft_precision": ft_metrics["precision"],
            "ft_recall": ft_metrics["recall"],
            "f1_delta": ft_metrics["f1"] - base_metrics["f1"],
            "accuracy_delta": ft_metrics["accuracy"] - base_metrics["accuracy"],
            "precision_delta": ft_metrics["precision"] - base_metrics["precision"],
            "recall_delta": ft_metrics["recall"] - base_metrics["recall"],
        })

        mlflow.log_text(
            base_metrics["report"],
            "base_model_report.txt",
        )

        mlflow.log_text(
            ft_metrics["report"],
            "finetuned_model_report.txt",
        )

        mlflow.log_text(
            PROMPT_TEMPLATE,
            "prompt_template.txt",
        )

        mlflow.log_text(
            build_comparison_summary(base_metrics, ft_metrics),
            "comparison_summary.txt",
        )


if __name__ == "__main__":
    run_evaluation()