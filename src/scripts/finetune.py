"""
Distributed BF16 LoRA Fine-Tuning

Launches multi-GPU training when configured, fine-tunes a BF16 base model with LoRA adapters, and evaluates the final adapter on the full test split.

Run:
uv run --extra finetune python src/scripts/finetune.py
"""

import json
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
from config import GPU_CONFIG


def detect_gpu_indices() -> list[int]:
    """Return available NVIDIA GPU indices from nvidia-smi."""

    # GPU Detection
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

    return [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def launch_distributed_if_needed() -> None:
    """Relaunch the script with torch distributed if required."""

    # Existing Distributed Run Check
    if os.environ.get("LOCAL_RANK") is not None:
        return

    # Multi-GPU Policy Check
    if not GPU_CONFIG["enabled"]:
        return

    available_gpu_ids = detect_gpu_indices()
    configured_gpu_ids = list(
        GPU_CONFIG.get("gpu_ids") or available_gpu_ids
    )

    # GPU Availability Validation
    missing_gpu_ids = [
        gpu_id
        for gpu_id in configured_gpu_ids
        if gpu_id not in available_gpu_ids
    ]

    if missing_gpu_ids:
        raise RuntimeError(
            f"Configured GPUs are unavailable: {missing_gpu_ids}. "
            f"Available GPUs: {available_gpu_ids}"
        )

    if not configured_gpu_ids:
        raise RuntimeError("No NVIDIA GPUs were selected for fine-tuning.")

    # Distributed Environment Setup
    environment = os.environ.copy()

    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu_id)
        for gpu_id in configured_gpu_ids
    )

    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    environment.setdefault("TOKENIZERS_PARALLELISM", "false")

    process_count = len(configured_gpu_ids)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={process_count}",
        f"--master_port={GPU_CONFIG['master_port']}",
        str(Path(__file__).resolve()),
    ]

    print(
        f"Launching fine-tuning on {process_count} GPU(s): "
        f"{configured_gpu_ids}"
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    classification_report,
    f1_score,
    precision_score,
    recall_score,
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
from config import (
    BASE_MODEL,
    DATASET_PATH,
    FINETUNE_OUTPUT_DIR as OUTPUT_DIR,
    FINETUNE_PROMPT_TEMPLATE as PROMPT_TEMPLATE,
    MAX_LENGTH,
    MLFLOW_EXPERIMENT_FINETUNE,
    QLORA_CONFIG,
    build_training_config,
)

from src.config.settings import settings

# Logging Configuration
if not IS_MAIN_PROCESS:
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

# Device Setup
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for BF16 LoRA fine-tuning.")

torch.cuda.set_device(LOCAL_RANK)

DEVICE = torch.device("cuda", LOCAL_RANK)


def rank_zero_print(*args, **kwargs) -> None:
    """Print only from the main distributed process."""

    # Main Process Printing
    if IS_MAIN_PROCESS:
        print(*args, **kwargs)


def format_example(text: str, label: int, tokenizer) -> dict:
    """Format one clinic letter as a supervised generative example."""

    # Prompt and Response Construction
    prompt = PROMPT_TEMPLATE.format(text=str(text))
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
    max_prompt_len = max(0, MAX_LENGTH - len(response_ids))
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


def load_splits(
    tokenizer,
) -> tuple[Dataset, Dataset, Dataset, dict]:
    """Load train, validation and test splits."""

    # Dataset Loading
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Split Validation
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(data)

    if missing_splits:
        raise ValueError(
            f"Dataset is missing required splits: {sorted(missing_splits)}. "
            "Run relabel_dataset.py again."
        )

    # Training Split Formatting
    train_ds = Dataset.from_list([
        format_example(example["text"], example["label"], tokenizer)
        for example in data["train"]
    ])

    # Validation Split Formatting
    validation_ds = Dataset.from_list([
        format_example(example["text"], example["label"], tokenizer)
        for example in data["validation"]
    ])

    # Test Split Formatting
    test_ds = Dataset.from_list([
        format_example(example["text"], example["label"], tokenizer)
        for example in data["test"]
    ])

    return train_ds, validation_ds, test_ds, data


def parse_generated_label(generated_text: str) -> int:
    """Convert generated model text into a binary label."""

    # Label Parsing
    generated_text = str(generated_text).strip()

    return int(generated_text.startswith("1"))


def evaluate_generative(
    model,
    tokenizer,
    test_raw: list,
) -> dict:
    """Evaluate the final model on the full test split."""

    # Evaluation Setup
    model.eval()

    predictions = []
    labels = []

    rank_zero_print(
        f"\nRunning generative evaluation on "
        f"{len(test_raw):,} test examples..."
    )

    # Prediction Loop
    for index, example in enumerate(test_raw):
        prompt = PROMPT_TEMPLATE.format(
            text=str(example["text"]),
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH - 10,
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

        predictions.append(parse_generated_label(generated))
        labels.append(int(example["label"]))

        if (index + 1) % 50 == 0:
            rank_zero_print(
                f"  {index + 1:,}/{len(test_raw):,} evaluated..."
            )

    # Metric Calculation
    labels_arr = np.asarray(labels)
    predictions_arr = np.asarray(predictions)

    report = classification_report(
        labels_arr,
        predictions_arr,
        target_names=[
            "routine_followup",
            "treatment_event",
        ],
        digits=4,
        zero_division=0,
    )

    return {
        "f1": f1_score(
            labels_arr,
            predictions_arr,
            average="binary",
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            labels_arr,
            predictions_arr,
        ),
        "precision": precision_score(
            labels_arr,
            predictions_arr,
            zero_division=0,
        ),
        "recall": recall_score(
            labels_arr,
            predictions_arr,
            zero_division=0,
        ),
        "report": report,
        "n_evaluated": len(test_raw),
    }


def print_run_header(training_config: dict) -> None:
    """Print the distributed fine-tuning configuration."""

    # Run Header
    rank_zero_print("=" * 65)
    rank_zero_print("BF16 LoRA Fine-Tuning")
    rank_zero_print("=" * 65)
    rank_zero_print(f"Base model             : {BASE_MODEL}")
    rank_zero_print(f"Dataset                : {DATASET_PATH}")
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


def trainer_argument_config(training_config: dict) -> dict:
    """Remove logging-only values before constructing TrainingArguments."""

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


def run_finetune() -> None:
    """Run distributed BF16 LoRA fine-tuning."""

    # Output Directory Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tokeniser Loading
    rank_zero_print("Loading tokeniser...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Dataset Formatting
    rank_zero_print("Loading and formatting dataset...")

    train_ds, validation_ds, test_ds, raw_data = load_splits(tokenizer)
    training_config = build_training_config(len(train_ds))

    print_run_header(training_config)

    rank_zero_print(
        f"Train: {len(train_ds):,}  |  "
        f"Validation: {len(validation_ds):,}  |  "
        f"Test: {len(test_ds):,}"
    )

    # Model Loading
    rank_zero_print(f"\nLoading model in BF16: {BASE_MODEL}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.enable_input_require_grads()

    # LoRA Adapter Setup
    lora_config = LoraConfig(**QLORA_CONFIG)

    model = get_peft_model(model, lora_config)
    model.to(DEVICE)

    if IS_MAIN_PROCESS:
        model.print_trainable_parameters()

        rank_zero_print(
            f"Model memory footprint: "
            f"{model.get_memory_footprint() / 1024**3:.2f} GiB"
        )

    # Training Argument Setup
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        **trainer_argument_config(training_config),
    )

    # Trainer Setup
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
            run_name="lora_qwen35_9b_bf16_3gpu_classifier",
            tags={
                "sprint": "sprint_3",
                "task": "generative_classification",
                "base_model": BASE_MODEL,
                "adapter": "bf16_lora",
                "world_size": str(WORLD_SIZE),
            },
        )
    else:
        run_context = nullcontext()

    with run_context:
        # MLflow Parameter Logging
        if IS_MAIN_PROCESS:
            mlflow.log_params({
                "base_model": BASE_MODEL,
                "approach": "generative_classification",
                "lora_r": QLORA_CONFIG["r"],
                "lora_alpha": QLORA_CONFIG["lora_alpha"],
                "target_modules": str(
                    QLORA_CONFIG["target_modules"]
                ),
                "epochs": training_config["num_train_epochs"],
                "per_device_batch_size": (
                    training_config[
                        "per_device_train_batch_size"
                    ]
                ),
                "gradient_accumulation_steps": (
                    training_config[
                        "gradient_accumulation_steps"
                    ]
                ),
                "global_batch_size": training_config["global_batch_size"],
                "world_size": WORLD_SIZE,
                "learning_rate": training_config["learning_rate"],
                "precision": "bf16",
                "max_length": MAX_LENGTH,
                "n_train": len(train_ds),
                "n_validation": len(validation_ds),
                "n_test": len(test_ds),
            })

        # Model Training
        rank_zero_print("\nStarting BF16 LoRA fine-tuning...")

        trainer.train()
        trainer.accelerator.wait_for_everyone()

        if not IS_MAIN_PROCESS:
            return

        # Model Unwrapping
        inference_model = trainer.accelerator.unwrap_model(
            trainer.model_wrapped
        )

        inference_model.eval()

        # Full Test-Set Evaluation
        rank_zero_print(
            f"Automatically evaluating full test split: "
            f"{len(raw_data['test']):,} examples"
        )

        eval_metrics = evaluate_generative(
            inference_model,
            tokenizer,
            raw_data["test"],
        )

        # MLflow Metric Logging
        mlflow.log_metrics({
            "test_f1": eval_metrics["f1"],
            "test_accuracy": eval_metrics["accuracy"],
            "test_precision": eval_metrics["precision"],
            "test_recall": eval_metrics["recall"],
            "n_evaluated": eval_metrics["n_evaluated"],
        })

        # Report Logging
        rank_zero_print("\nClassification Report:")
        rank_zero_print(eval_metrics["report"])

        mlflow.log_text(
            eval_metrics["report"],
            "classification_report.txt",
        )

        mlflow.log_text(
            PROMPT_TEMPLATE,
            "prompt_template.txt",
        )

        # Adapter Saving
        inference_model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        # Final Summary
        rank_zero_print("\n" + "=" * 65)
        rank_zero_print("FINE-TUNING COMPLETE")
        rank_zero_print("=" * 65)
        rank_zero_print(f"  Model      : {BASE_MODEL}")
        rank_zero_print(f"  F1 Score   : {eval_metrics['f1']:.4f}")
        rank_zero_print(f"  Accuracy   : {eval_metrics['accuracy']:.4f}")
        rank_zero_print(f"  Precision  : {eval_metrics['precision']:.4f}")
        rank_zero_print(f"  Recall     : {eval_metrics['recall']:.4f}")
        rank_zero_print(f"  Adapter    : {OUTPUT_DIR}")
        rank_zero_print("  MLflow     : http://127.0.0.1:5000")
        rank_zero_print("=" * 65)


if __name__ == "__main__":
    run_finetune()