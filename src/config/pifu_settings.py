"""
FastPIFU same-protocol configuration.

Stores dataset paths, output paths, label mappings, LoRA settings, MLflow names, prompt template, and distributed training parameters for the three-class FastPIFU cardiology task.
"""

import math
import os
import sys
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = str(Path(sys.executable))

# FastPIFU Data Root
PIFU_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "fastpifu_cardiology"
)

# FastPIFU Data Directories
PIFU_RAW_DIR = PIFU_DATA_ROOT / "raw"
PIFU_EXTRACTED_DIR = PIFU_DATA_ROOT / "extracted"
PIFU_PROCESSED_DIR = PIFU_DATA_ROOT / "processed"

# FastPIFU Source Data
PIFU_ZIP_PATH = PIFU_RAW_DIR / "fastpifu-cardiology.zip"

# FastPIFU Processed Dataset Paths
PIFU_TRAIN_PATH = PIFU_PROCESSED_DIR / "pifu_train.json"
PIFU_VALIDATION_PATH = PIFU_PROCESSED_DIR / "pifu_validation.json"
PIFU_TEST_PATH = PIFU_PROCESSED_DIR / "pifu_external_test_150.json"
PIFU_CHALLENGE_PATH = PIFU_PROCESSED_DIR / "pifu_challenge_24.json"
PIFU_DATASET_SUMMARY_PATH = PIFU_PROCESSED_DIR / "dataset_summary.json"

# Backward-Compatible Summary Alias
PIFU_SUMMARY_PATH = PIFU_DATASET_SUMMARY_PATH

# FastPIFU Split Path Map
PIFU_SPLIT_PATHS = {
    "train": PIFU_TRAIN_PATH,
    "validation": PIFU_VALIDATION_PATH,
    "test": PIFU_TEST_PATH,
    "challenge": PIFU_CHALLENGE_PATH,
}

# Fine-Tuning Output Paths
PIFU_FINETUNE_DIR = (
    PROJECT_ROOT
    / "data"
    / "finetune"
)

PIFU_OUTPUT_DIR = (
    PIFU_FINETUNE_DIR
    / "qwen35_9b_pifu_lora_3epochs_same_protocol"
)

PIFU_EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluations"
    / "pifu_cardiology_same_protocol"
)

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow_runs/mlflow.db"
MLFLOW_EXPERIMENT_FINETUNE = "nhs_omop_pifu_finetune"

PIFU_TRAIN_RUN_NAME = "lora_qwen35_9b_bf16_pifu_3class"
PIFU_EVALUATION_RUN_NAME = "evaluation_pifu_base_vs_finetuned"

# Backward-Compatible MLflow Aliases
PIFU_MLFLOW_EXPERIMENT = MLFLOW_EXPERIMENT_FINETUNE
PIFU_MLFLOW_TRAIN_RUN_NAME = PIFU_TRAIN_RUN_NAME
PIFU_MLFLOW_EVAL_RUN_NAME = PIFU_EVALUATION_RUN_NAME

# Model Configuration
PIFU_BASE_MODEL = "Qwen/Qwen3.5-9B"
PIFU_MAX_LENGTH = 512
PIFU_EVAL_BATCH_SIZE = 2

# Three-Class PIFU Label Space
PIFU_LABEL_TO_ID = {
    "NOT_ELIGIBLE": 0,
    "BORDERLINE": 1,
    "ELIGIBLE": 2,
}

PIFU_ID_TO_LABEL = {
    label_id: label_name
    for label_name, label_id in PIFU_LABEL_TO_ID.items()
}

PIFU_LABEL_IDS = tuple(
    sorted(PIFU_ID_TO_LABEL)
)

PIFU_CLASS_NAMES = [
    PIFU_ID_TO_LABEL[label_id]
    for label_id in PIFU_LABEL_IDS
]

# Multi-GPU Policy
PIFU_GPU_CONFIG = {
    "enabled": True,
    "gpu_ids": None,
    "target_global_batch_size": 16,
    "master_port": 29510,
    "default_gpu_count": 2,
    "max_used_memory_mib": 1500,
    "max_utilisation_percent": 10,
}

# Backward-Compatible GPU Aliases
PIFU_GPU_IDS = PIFU_GPU_CONFIG["gpu_ids"]
PIFU_MASTER_PORT = PIFU_GPU_CONFIG["master_port"]
PIFU_TARGET_GLOBAL_BATCH_SIZE = PIFU_GPU_CONFIG[
    "target_global_batch_size"
]
PIFU_EPOCHS = 3

# LoRA Configuration
PIFU_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


def distributed_context() -> dict:
    """Return distributed training metadata from environment variables."""

    # Distributed Environment Detection
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_distributed": world_size > 1,
        "is_main_process": rank == 0,
    }


def build_pifu_training_config(
    n_train: int,
) -> dict:
    """Build Trainer arguments using the same protocol as the earlier model."""

    # Distributed Context
    context = distributed_context()
    world_size = context["world_size"]

    # Core Training Settings
    epochs = PIFU_EPOCHS
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1

    # Batch Size Calculation
    gradient_accumulation_steps = max(
        1,
        round(
            PIFU_GPU_CONFIG["target_global_batch_size"]
            / (
                per_device_train_batch_size
                * world_size
            )
        ),
    )

    global_batch_size = (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * world_size
    )

    # Step Calculation
    samples_per_rank = math.ceil(
        n_train / world_size
    )

    micro_batches_per_epoch = math.ceil(
        samples_per_rank
        / per_device_train_batch_size
    )

    optimiser_steps_per_epoch = math.ceil(
        micro_batches_per_epoch
        / gradient_accumulation_steps
    )

    total_optimiser_steps = (
        optimiser_steps_per_epoch
        * epochs
    )

    warmup_steps = max(
        1,
        round(total_optimiser_steps * 0.10),
    )

    return {
        "num_train_epochs": epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": 2e-4,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": 10,
        "bf16": True,
        "fp16": False,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        "report_to": "none",
        "optim": "adamw_torch_fused",
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
        },
        "ddp_find_unused_parameters": False,
        "remove_unused_columns": False,
        "seed": 42,
        "data_seed": 42,
        "global_batch_size": global_batch_size,
        "optimiser_steps_per_epoch": optimiser_steps_per_epoch,
        "total_optimiser_steps": total_optimiser_steps,
    }


# Backward-Compatible Training Config Alias
def build_training_config(n_train: int) -> dict:
    """Return PIFU Trainer arguments for scripts expecting build_training_config."""

    # Training Config Compatibility
    return build_pifu_training_config(n_train)


# PIFU Prompt Template
PIFU_PROMPT_TEMPLATE = """You are a safety-focused clinical text classifier for NHS cardiology outpatient letters.

Classify the patient's current suitability for Patient-Initiated Follow-Up (PIFU):

- 0 = NOT_ELIGIBLE: an exclusion criterion is present, timed specialist follow-up is required, active management is required, or the patient is discharged to a different pathway rather than placed on PIFU.
- 1 = BORDERLINE: the evidence is incomplete, conflicting, or uncertain and requires clinician review.
- 2 = ELIGIBLE: the patient is stable, appropriately informed, low risk, and suitable for PIFU.

Clinic letter:
{text}

Return only 0, 1, or 2:"""


def create_pifu_directories() -> None:
    """Create local directories required by the PIFU workflow."""

    # Directory Creation
    for directory in [
        PIFU_RAW_DIR,
        PIFU_EXTRACTED_DIR,
        PIFU_PROCESSED_DIR,
        PIFU_FINETUNE_DIR,
        PIFU_OUTPUT_DIR,
        PIFU_EVALUATION_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)