"""
FastPIFU Fine-Tuning

Fine-tunes Qwen3.5 for three-class PIFU eligibility classification using the same supervised generative-classification protocol as the earlier clinic-letter model.

Run:
uv run --extra finetune python src/scripts/finetune_pifu.py
"""

import csv
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Launch Configuration
from src.config.pifu_settings import PIFU_GPU_CONFIG


def detect_gpu_indices() -> list[int]:
    """Return physical NVIDIA GPU indices."""

    # GPU Detection
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except FileNotFoundError as error:
        raise RuntimeError(
            "nvidia-smi was not found. CUDA GPUs are required for "
            "PIFU fine-tuning."
        ) from error

    return [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def configured_or_visible_gpu_ids() -> list[int]:
    """Resolve GPUs for an automatic distributed launch."""

    # Explicit GPU Configuration
    configured = PIFU_GPU_CONFIG.get("gpu_ids")

    if configured:
        return [
            int(value)
            for value in configured
        ]

    # Existing CUDA Visibility
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if visible:
        return [
            int(value.strip())
            for value in visible.split(",")
            if value.strip()
        ]

    # Automatic GPU Selection
    available_gpu_ids = detect_gpu_indices()

    return available_gpu_ids[
        :PIFU_GPU_CONFIG["default_gpu_count"]
    ]


def validate_gpu_ids(gpu_ids: list[int]) -> None:
    """Validate selected physical GPU IDs before torchrun launch."""

    # GPU Selection Check
    if not gpu_ids:
        raise RuntimeError(
            "No GPUs were selected for PIFU fine-tuning."
        )

    # Physical GPU Availability Check
    available_gpu_ids = detect_gpu_indices()

    missing_gpu_ids = [
        gpu_id
        for gpu_id in gpu_ids
        if gpu_id not in available_gpu_ids
    ]

    if missing_gpu_ids:
        raise RuntimeError(
            f"Configured GPUs are unavailable: {missing_gpu_ids}. "
            f"Available GPUs: {available_gpu_ids}"
        )


def launch_distributed_if_needed() -> None:
    """Relaunch with torch distributed when the script is run directly."""

    # Existing Distributed Run Check
    if os.environ.get("LOCAL_RANK") is not None:
        return

    # Multi-GPU Policy Check
    if not PIFU_GPU_CONFIG["enabled"]:
        return

    gpu_ids = configured_or_visible_gpu_ids()
    validate_gpu_ids(gpu_ids)

    # Distributed Environment Setup
    environment = os.environ.copy()

    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu_id)
        for gpu_id in gpu_ids
    )

    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    environment.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        f"--master_port={PIFU_GPU_CONFIG['master_port']}",
        str(Path(__file__).resolve()),
    ]

    print(
        f"Launching PIFU fine-tuning on "
        f"{len(gpu_ids)} GPU(s): {gpu_ids}"
    )

    completed = subprocess.run(
        command,
        env=environment,
        check=False,
    )

    raise SystemExit(completed.returncode)


# Distributed Relaunch
launch_distributed_if_needed()

# Distributed Runtime Context
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN_PROCESS = RANK == 0

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

if not IS_MAIN_PROCESS:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["DATASETS_VERBOSITY"] = "error"

# Experiment Tracking
import mlflow

# Data and Numerical Libraries
import numpy as np
import torch
from datasets import Dataset

# PEFT Libraries
from peft import (
    LoraConfig,
    get_peft_model,
)

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Training Components
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from transformers.utils import logging as transformers_logging

# Project Configuration
from src.config.pifu_settings import (
    MLFLOW_EXPERIMENT_FINETUNE,
    PIFU_BASE_MODEL,
    PIFU_CHALLENGE_PATH,
    PIFU_DATASET_SUMMARY_PATH,
    PIFU_ID_TO_LABEL,
    PIFU_LABEL_IDS,
    PIFU_LORA_CONFIG,
    PIFU_MAX_LENGTH,
    PIFU_OUTPUT_DIR,
    PIFU_PROMPT_TEMPLATE,
    PIFU_TEST_PATH,
    PIFU_TRAIN_PATH,
    PIFU_TRAIN_RUN_NAME,
    PIFU_VALIDATION_PATH,
    build_pifu_training_config,
)

from src.config.settings import settings

# Logging Configuration
if not IS_MAIN_PROCESS:
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

# Device Setup
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for BF16 LoRA fine-tuning."
    )

torch.cuda.set_device(LOCAL_RANK)

DEVICE = torch.device(
    "cuda",
    LOCAL_RANK,
)


def rank_zero_print(*args, **kwargs) -> None:
    """Print only from rank zero."""

    # Main Process Printing
    if IS_MAIN_PROCESS:
        print(*args, **kwargs)


def load_samples(path: Path) -> list[dict]:
    """Load and validate one prepared FastPIFU split."""

    # Split Existence Check
    if not path.exists():
        raise FileNotFoundError(
            f"PIFU dataset split not found: {path}. "
            "Run prepare_fastpifu_dataset.py first."
        )

    # Split Loading
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    samples = payload.get("samples")

    if not isinstance(samples, list):
        raise ValueError(
            f"{path} does not contain a valid 'samples' list."
        )

    if not samples:
        raise ValueError(f"{path} contains no samples.")

    # Sample Validation
    for index, sample in enumerate(samples):
        if "text" not in sample or "label" not in sample:
            raise ValueError(
                f"Sample {index} in {path} must contain text and label."
            )

        label = int(sample["label"])

        if label not in PIFU_LABEL_IDS:
            raise ValueError(
                f"Unexpected label {label} in sample {index} from {path}."
            )

    return samples


def format_example(
    text: str,
    label: int,
    tokenizer,
) -> dict:
    """Format one PIFU letter as a supervised generative example."""

    # Prompt and Response Construction
    prompt = PIFU_PROMPT_TEMPLATE.format(
        text=str(text),
    )

    response = f" {int(label)}{tokenizer.eos_token}"

    # Token ID Construction
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
    )["input_ids"]

    # Prompt Truncation
    max_prompt_len = PIFU_MAX_LENGTH - len(response_ids)

    if max_prompt_len < 1:
        raise ValueError(
            f"PIFU_MAX_LENGTH={PIFU_MAX_LENGTH} is too small "
            "for the label response tokens."
        )

    prompt_ids = prompt_ids[:max_prompt_len]

    # Input and Label Construction
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataset(
    samples: list[dict],
    tokenizer,
) -> Dataset:
    """Convert prepared samples into a HuggingFace Dataset."""

    # Dataset Formatting
    return Dataset.from_list([
        format_example(
            sample["text"],
            sample["label"],
            tokenizer,
        )
        for sample in samples
    ])


def load_splits(
    tokenizer,
) -> tuple[
    Dataset,
    Dataset,
    Dataset,
    dict,
]:
    """Load train, validation, test and challenge data."""

    # Raw Split Loading
    raw_data = {
        "train": load_samples(PIFU_TRAIN_PATH),
        "validation": load_samples(PIFU_VALIDATION_PATH),
        "test": load_samples(PIFU_TEST_PATH),
        "challenge": load_samples(PIFU_CHALLENGE_PATH),
    }

    # Dataset Formatting
    train_ds = build_dataset(
        raw_data["train"],
        tokenizer,
    )

    validation_ds = build_dataset(
        raw_data["validation"],
        tokenizer,
    )

    test_ds = build_dataset(
        raw_data["test"],
        tokenizer,
    )

    return (
        train_ds,
        validation_ds,
        test_ds,
        raw_data,
    )


def parse_generated_label(
    generated_text: str,
) -> tuple[int, bool]:
    """Parse a generated PIFU label conservatively."""

    # Strict Leading Label Parsing
    value = str(generated_text).strip()

    if value and value[0] in {"0", "1", "2"}:
        return int(value[0]), False

    # Invalid Output Safety Routing
    return 1, True


def evaluate_generative(
    model,
    tokenizer,
    samples: list[dict],
) -> dict:
    """Evaluate the final adapter using generative decoding."""

    # Evaluation Setup
    model.eval()

    predictions = []
    labels = []
    invalid_output_count = 0
    prediction_rows = []

    rank_zero_print(
        f"\nRunning generative evaluation on "
        f"{len(samples):,} PIFU examples..."
    )

    # Prediction Loop
    for index, sample in enumerate(samples):
        prompt = PIFU_PROMPT_TEMPLATE.format(
            text=str(sample["text"]),
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=PIFU_MAX_LENGTH - 10,
        ).to(DEVICE)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        prediction, invalid = parse_generated_label(generated)
        label = int(sample["label"])

        invalid_output_count += int(invalid)
        predictions.append(prediction)
        labels.append(label)

        prediction_rows.append({
            "sample_id": sample.get("sample_id"),
            "source": sample.get("source"),
            "true_label": label,
            "true_label_name": PIFU_ID_TO_LABEL[label],
            "predicted_label": prediction,
            "predicted_label_name": PIFU_ID_TO_LABEL[prediction],
            "generated_text": generated,
            "invalid_output": invalid,
        })

        if (index + 1) % 25 == 0:
            rank_zero_print(
                f"  {index + 1:,}/{len(samples):,} evaluated..."
            )

    # Metric Arrays
    labels_arr = np.asarray(
        labels,
        dtype=int,
    )

    predictions_arr = np.asarray(
        predictions,
        dtype=int,
    )

    # Classification Report
    target_names = [
        PIFU_ID_TO_LABEL[label]
        for label in PIFU_LABEL_IDS
    ]

    report_text = classification_report(
        labels_arr,
        predictions_arr,
        labels=list(PIFU_LABEL_IDS),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        labels_arr,
        predictions_arr,
        labels=list(PIFU_LABEL_IDS),
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    # Confusion Matrix
    matrix = confusion_matrix(
        labels_arr,
        predictions_arr,
        labels=list(PIFU_LABEL_IDS),
    )

    # Safety Metrics
    true_not_eligible = labels_arr == 0
    unsafe_eligible = true_not_eligible & (predictions_arr == 2)

    unsafe_eligible_rate = (
        float(unsafe_eligible.sum()) / float(true_not_eligible.sum())
        if true_not_eligible.sum()
        else float("nan")
    )

    return {
        "accuracy": accuracy_score(
            labels_arr,
            predictions_arr,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            labels_arr,
            predictions_arr,
        ),
        "macro_f1": f1_score(
            labels_arr,
            predictions_arr,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            labels_arr,
            predictions_arr,
            average="weighted",
            zero_division=0,
        ),
        "not_eligible_recall": report_dict["NOT_ELIGIBLE"]["recall"],
        "borderline_recall": report_dict["BORDERLINE"]["recall"],
        "eligible_precision": report_dict["ELIGIBLE"]["precision"],
        "eligible_recall": report_dict["ELIGIBLE"]["recall"],
        "unsafe_eligible_count": int(unsafe_eligible.sum()),
        "unsafe_eligible_rate": unsafe_eligible_rate,
        "manual_review_rate": float(np.mean(predictions_arr == 1)),
        "invalid_output_count": invalid_output_count,
        "n_evaluated": len(samples),
        "confusion_matrix": matrix.tolist(),
        "report": report_text,
        "predictions": prediction_rows,
    }


def print_run_header(
    training_config: dict,
) -> None:
    """Print the same run summary style as the earlier model."""

    # Run Header
    rank_zero_print("=" * 65)
    rank_zero_print("BF16 LoRA Fine-Tuning - PIFU")
    rank_zero_print("=" * 65)
    rank_zero_print(f"Base model             : {PIFU_BASE_MODEL}")
    rank_zero_print(f"Train dataset          : {PIFU_TRAIN_PATH}")
    rank_zero_print(f"Visible CUDA devices   : {torch.cuda.device_count()}")
    rank_zero_print(f"Distributed world size : {WORLD_SIZE}")
    rank_zero_print(f"Per-device batch       : {training_config['per_device_train_batch_size']}")
    rank_zero_print(f"Gradient accumulation  : {training_config['gradient_accumulation_steps']}")
    rank_zero_print(f"Global batch size      : {training_config['global_batch_size']}")
    rank_zero_print(f"Epochs                 : {training_config['num_train_epochs']}")
    rank_zero_print(f"Steps per epoch        : {training_config['optimiser_steps_per_epoch']}")
    rank_zero_print(f"Total optimiser steps  : {training_config['total_optimiser_steps']}")

    # GPU Summary
    for gpu_index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(gpu_index)

        rank_zero_print(
            f"GPU {gpu_index}                 : "
            f"{properties.name}, "
            f"{properties.total_memory / 1024**3:.1f} GiB"
        )

    rank_zero_print()


def trainer_argument_config(
    training_config: dict,
) -> dict:
    """Remove non-Trainer summary keys."""

    # Trainer Argument Filtering
    excluded_keys = {
        "global_batch_size",
        "optimiser_steps_per_epoch",
        "total_optimiser_steps",
    }

    return {
        key: value
        for key, value in training_config.items()
        if key not in excluded_keys
    }


def write_evaluation_artifacts(
    metrics: dict,
) -> dict[str, Path]:
    """Write report, metrics, predictions and confusion matrix."""

    # Evaluation Directory Setup
    evaluation_dir = PIFU_OUTPUT_DIR / "evaluation"

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = evaluation_dir / "classification_report.txt"
    metrics_path = evaluation_dir / "test_metrics.json"
    predictions_path = evaluation_dir / "test_predictions.csv"
    matrix_path = evaluation_dir / "confusion_matrix.csv"

    # Report Artifact
    report_path.write_text(
        metrics["report"],
        encoding="utf-8",
    )

    # Metrics Artifact
    serialisable = {
        key: value
        for key, value in metrics.items()
        if key not in {
            "report",
            "predictions",
        }
    }

    metrics_path.write_text(
        json.dumps(
            serialisable,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Predictions Artifact
    with open(
        predictions_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "sample_id",
            "source",
            "true_label",
            "true_label_name",
            "predicted_label",
            "predicted_label_name",
            "generated_text",
            "invalid_output",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metrics["predictions"])

    # Confusion Matrix Artifact
    with open(
        matrix_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "true/predicted",
            *[
                PIFU_ID_TO_LABEL[label]
                for label in PIFU_LABEL_IDS
            ],
        ])

        for label, row in zip(
            PIFU_LABEL_IDS,
            metrics["confusion_matrix"],
            strict=True,
        ):
            writer.writerow([
                PIFU_ID_TO_LABEL[label],
                *row,
            ])

    return {
        "report": report_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "confusion_matrix": matrix_path,
    }


def finite_metrics(metrics: dict) -> dict:
    """Return only finite scalar metrics for MLflow logging."""

    # Finite Metric Filtering
    result = {}

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            number = float(value)

            if math.isfinite(number):
                result[key] = number

    return result


def log_adapter_artifacts_to_mlflow() -> None:
    """Log saved adapter and tokenizer files."""

    # Adapter Artifact Selection
    artifact_names = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "trainer_state.json",
    ]

    # Adapter Artifact Logging
    for artifact_name in artifact_names:
        artifact_path = PIFU_OUTPUT_DIR / artifact_name

        if artifact_path.exists():
            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="adapter",
            )


def run_finetune() -> None:
    """Run distributed BF16 LoRA PIFU fine-tuning."""

    # Output Directory Setup
    PIFU_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Tokeniser Loading
    rank_zero_print("Loading tokeniser...")

    tokenizer = AutoTokenizer.from_pretrained(
        PIFU_BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Dataset Formatting
    rank_zero_print("Loading and formatting FastPIFU dataset...")

    train_ds, validation_ds, test_ds, raw_data = load_splits(tokenizer)

    training_config = build_pifu_training_config(
        len(train_ds)
    )

    print_run_header(training_config)

    rank_zero_print(
        f"Train: {len(train_ds):,}  |  "
        f"Validation: {len(validation_ds):,}  |  "
        f"External test: {len(test_ds):,}  |  "
        f"Challenge: {len(raw_data['challenge']):,}"
    )

    # Model Loading
    rank_zero_print(
        f"\nLoading model in BF16: {PIFU_BASE_MODEL}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        PIFU_BASE_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.enable_input_require_grads()

    # LoRA Adapter Setup
    lora_config = LoraConfig(**PIFU_LORA_CONFIG)

    model = get_peft_model(
        model,
        lora_config,
    )

    model.to(DEVICE)

    if IS_MAIN_PROCESS:
        model.print_trainable_parameters()

        rank_zero_print(
            f"Model memory footprint: "
            f"{model.get_memory_footprint() / 1024**3:.2f} GiB"
        )

    # Trainer Setup
    training_args = TrainingArguments(
        output_dir=str(PIFU_OUTPUT_DIR),
        **trainer_argument_config(training_config),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
        ),
    )

    # MLflow Setup
    if IS_MAIN_PROCESS:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_FINETUNE)

        run_context = mlflow.start_run(
            run_name=PIFU_TRAIN_RUN_NAME,
            tags={
                "sprint": "sprint_3",
                "task": "pifu_eligibility_3class",
                "dataset": "fastpifu_cardiology",
                "base_model": PIFU_BASE_MODEL,
                "adapter": "bf16_lora",
                "world_size": str(WORLD_SIZE),
                "comparison_protocol": "same_as_original_letters",
            },
        )

    else:
        run_context = nullcontext()

    with run_context:
        # MLflow Parameter Logging
        if IS_MAIN_PROCESS:
            mlflow.log_params({
                "base_model": PIFU_BASE_MODEL,
                "approach": "generative_classification",
                "task": "pifu_eligibility_3class",
                "dataset": "fastpifu_cardiology",
                "lora_r": PIFU_LORA_CONFIG["r"],
                "lora_alpha": PIFU_LORA_CONFIG["lora_alpha"],
                "target_modules": str(PIFU_LORA_CONFIG["target_modules"]),
                "epochs": training_config["num_train_epochs"],
                "per_device_batch_size": (
                    training_config["per_device_train_batch_size"]
                ),
                "gradient_accumulation_steps": (
                    training_config["gradient_accumulation_steps"]
                ),
                "global_batch_size": training_config["global_batch_size"],
                "world_size": WORLD_SIZE,
                "learning_rate": training_config["learning_rate"],
                "precision": "bf16",
                "max_length": PIFU_MAX_LENGTH,
                "n_train": len(train_ds),
                "n_validation": len(validation_ds),
                "n_test": len(test_ds),
                "n_challenge": len(raw_data["challenge"]),
            })

        # Model Training
        rank_zero_print(
            "\nStarting BF16 LoRA PIFU fine-tuning..."
        )

        trainer.train()
        trainer.accelerator.wait_for_everyone()

        if not IS_MAIN_PROCESS:
            return

        # Model and State Saving
        trainer.save_state()

        inference_model = trainer.accelerator.unwrap_model(
            trainer.model_wrapped
        )

        inference_model.eval()

        inference_model.save_pretrained(
            str(PIFU_OUTPUT_DIR)
        )

        tokenizer.save_pretrained(
            str(PIFU_OUTPUT_DIR)
        )

        # Full External-Test Evaluation
        rank_zero_print(
            f"Automatically evaluating external test split: "
            f"{len(raw_data['test']):,} examples"
        )

        test_metrics = evaluate_generative(
            inference_model,
            tokenizer,
            raw_data["test"],
        )

        mlflow.log_metrics({
            f"test_{key}": value
            for key, value in finite_metrics(test_metrics).items()
        })

        rank_zero_print("\nClassification Report:")
        rank_zero_print(test_metrics["report"])

        # Artifact Writing
        artifact_paths = write_evaluation_artifacts(
            test_metrics
        )

        # MLflow Artifact Logging
        mlflow.log_text(
            test_metrics["report"],
            "classification_report.txt",
        )

        mlflow.log_text(
            PIFU_PROMPT_TEMPLATE,
            "prompt_template.txt",
        )

        mlflow.log_dict(
            PIFU_LORA_CONFIG,
            "lora_config.json",
        )

        mlflow.log_dict(
            training_config,
            "training_config.json",
        )

        for artifact_path in artifact_paths.values():
            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="evaluation",
            )

        if PIFU_DATASET_SUMMARY_PATH.exists():
            mlflow.log_artifact(
                str(PIFU_DATASET_SUMMARY_PATH),
                artifact_path="data",
            )

        log_adapter_artifacts_to_mlflow()

        # Final Summary
        rank_zero_print("\n" + "=" * 65)
        rank_zero_print("PIFU FINE-TUNING COMPLETE")
        rank_zero_print("=" * 65)
        rank_zero_print(f"  Model            : {PIFU_BASE_MODEL}")
        rank_zero_print(f"  Macro F1         : {test_metrics['macro_f1']:.4f}")
        rank_zero_print(f"  Accuracy         : {test_metrics['accuracy']:.4f}")
        rank_zero_print(
            f"  NOT_EL recall    : "
            f"{test_metrics['not_eligible_recall']:.4f}"
        )
        rank_zero_print(
            f"  EL precision     : "
            f"{test_metrics['eligible_precision']:.4f}"
        )
        rank_zero_print(
            f"  Unsafe eligible  : "
            f"{test_metrics['unsafe_eligible_count']}"
        )
        rank_zero_print(f"  Adapter          : {PIFU_OUTPUT_DIR}")
        rank_zero_print("  MLflow           : http://127.0.0.1:5000")
        rank_zero_print("=" * 65)


if __name__ == "__main__":
    run_finetune()