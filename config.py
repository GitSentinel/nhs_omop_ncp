"""
Project Configuration

Stores shared paths, model settings, fine-tuning parameters, MCP server paths, MLflow experiment names, and agent prompts.
"""

import math
import os
import sys
from pathlib import Path

import torch

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_BIN = str(Path(sys.executable))

# Data Paths
DUCKDB_PATH = PROJECT_ROOT / "data" / "processed" / "omop_v54.duckdb"
FINETUNE_DIR = PROJECT_ROOT / "data" / "finetune"
COHORT_OUTPUT_DIR = PROJECT_ROOT / "data" / "cohort_run_outputs"

SOURCE_DATASET_PATH = FINETUNE_DIR / "clinic_letters_labelled.json"
AZURE_DATASET_PATH = FINETUNE_DIR / "clinic_letters_azure_labelled.json"
DATASET_PATH = AZURE_DATASET_PATH

FINETUNE_OUTPUT_DIR = (
    FINETUNE_DIR / "qwen35_9b_qlora_adapter_3epochs_recall_v2"
)

# MCP Server Paths
OMOP_SERVER_PATH = PROJECT_ROOT / "src" / "mcp_server" / "server.py"
SKILLS_SERVER_PATH = PROJECT_ROOT / "src" / "mcp_server" / "skills_server.py"
SKILLS_DIR = PROJECT_ROOT / "src" / "mcp_server" / "resources" / "skills"

# Azure OpenAI Configuration
AZURE_ENDPOINT = "https://openai-omop-dev-01.services.ai.azure.com/openai/v1"
AZURE_DEPLOYMENT = "gpt-5-nano"

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow_runs/mlflow.db"
MLFLOW_EXPERIMENT_AGENT = "nhs_omop_agent"
MLFLOW_EXPERIMENT_FINETUNE = "nhs_omop_finetune"

# Fine-Tuning Configuration
BASE_MODEL = "Qwen/Qwen3.5-9B"
MAX_LENGTH = 512
N_DATASET_SAMPLES = 3000
PIFU_MIN_RECALL = 0.95

# Multi-GPU Policy
GPU_CONFIG = {
    "enabled": True,
    "gpu_ids": (0, 1, 2),
    "target_global_batch_size": 16,
    "master_port": 29500,
}

# LoRA Configuration
QLORA_CONFIG = {
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


def build_training_config(n_train: int) -> dict:
    """Build Trainer arguments using dataset size and GPU context."""

    # Distributed Context
    context = distributed_context()
    world_size = context["world_size"]

    # Core Training Settings
    epochs = 3
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1

    # Batch Size Calculation
    gradient_accumulation_steps = max(
        1,
        round(
            GPU_CONFIG["target_global_batch_size"]
            / (per_device_train_batch_size * world_size)
        ),
    )

    global_batch_size = (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * world_size
    )

    # Step Calculation
    samples_per_rank = math.ceil(n_train / world_size)

    micro_batches_per_epoch = math.ceil(
        samples_per_rank / per_device_train_batch_size
    )

    optimiser_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / gradient_accumulation_steps
    )

    total_optimiser_steps = optimiser_steps_per_epoch * epochs
    warmup_steps = max(1, round(total_optimiser_steps * 0.10))

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


# Training Configuration Preview
TRAINING_CONFIG = build_training_config(
    n_train=int(N_DATASET_SAMPLES * 0.8)
)

# BitsAndBytes Configuration
BNB_CONFIG_PARAMS = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": True,
}

# Fine-Tuning Prompt Template
FINETUNE_PROMPT_TEMPLATE = """You are a safety-focused clinical text classifier for NHS outpatient letters.

Classify the letter as:
- 0 = routine_followup: stable monitoring or review only, with no new treatment, treatment change, escalation, procedure or active clinical action.
- 1 = treatment_event: any new or changed medication or treatment, procedure, referral, escalation, urgent review, abnormal result requiring action, intervention, or discharge following treatment.

Follow-up wording alone does not make the letter routine.

Clinic letter:
{text}

Return only 0 or 1:"""

# Treatment Keywords
TREATMENT_KEYWORDS = [
    "procedure",
    "operation",
    "surgery",
    "theatre",
    "treatment",
    "intervention",
    "injection",
    "biopsy",
    "excision",
    "repair",
    "removal",
    "transplant",
    "admitted",
    "admission",
    "inpatient",
    "referred for",
    "referral to",
    "discharged",
    "consent obtained",
    "consent signed",
    "scheduled for",
    "booked for",
    "listed for",
    "clock stop",
    "rtt",
    "18 week",
    "commenced treatment",
    "started treatment",
    "prescribed",
    "initiated therapy",
]

# Routine Keywords
ROUTINE_KEYWORDS = [
    "routine follow-up",
    "follow up in",
    "review in",
    "stable",
    "no change",
    "unchanged",
    "monitoring",
    "surveillance",
    "watchful waiting",
    "continue current",
    "no new concerns",
    "appointment in",
    "clinic in",
    "doing well",
    "no intervention required",
    "conservative management",
    "reassured",
]

# Cohort Queries
COHORT_QUERIES = [
    "Summarise this patient's medical history including conditions, medications and recent visits.",
    "What conditions does this patient have and what medications have been prescribed?",
    "What recent measurements and laboratory values are recorded for this patient?",
    "Provide a clinical summary of this patient's visit history and procedures.",
    "Based on this patient's conditions, which FastPIFU specialty skill is most relevant and what does it say about PIFU suitability?",
]

# FastPIFU Skill Map
SPECIALTY_MAP = {
    "cardiology": "cardiology.md",
    "dermatology": "dermatology.md",
    "ent": "ent.md",
    "gastroenterology": "gastroenterology.md",
    "general_surgery": "general_surgery.md",
    "gynaecology": "gynaecology.md",
    "omfs": "omfs.md",
    "ophthalmology": "ophthalmology.md",
    "orthopaedics": "orthopaedics.md",
    "spinal": "spinal.md",
    "urology": "urology.md",
    "omop_clinical_reasoning": "omop_clinical_reasoning.md",
}

# Condition Specialty Map
CONDITION_SPECIALTY_MAP = {
    "cardiology": [
        "heart",
        "cardiac",
        "arrhythmia",
        "atrial",
        "valve",
        "failure",
        "angina",
        "myocardial",
        "coronary",
        "pots",
        "lbbb",
        "bundle branch",
        "pericarditis",
        "brugada syndrome",
    ],
    "gastroenterology": [
        "liver",
        "cirrhosis",
        "hepatitis",
        "bowel",
        "crohn",
        "colitis",
        "gastro",
        "coeliac",
        "iron deficiency",
        "anaemia",
        "masld",
        "nafld",
        "ulcerative colitis",
    ],
    "dermatology": [
        "skin",
        "dermat",
        "melanoma",
        "acne",
        "psoriasis",
        "eczema",
        "keratosis",
        "carcinoma",
        "mole",
    ],
    "orthopaedics": [
        "joint",
        "knee",
        "hip",
        "shoulder",
        "fracture",
        "arthritis",
        "carpal",
        "tendon",
        "ligament",
        "bone",
        "spondylolisthesis",
    ],
    "ophthalmology": [
        "eye",
        "ocular",
        "retina",
        "glaucoma",
        "cataract",
        "macular",
        "visual",
        "diabetic eye",
    ],
    "urology": [
        "kidney",
        "bladder",
        "prostate",
        "urolog",
        "renal",
        "urinary",
        "incontinence",
    ],
    "gynaecology": [
        "gynaecolog",
        "uterine",
        "ovarian",
        "pelvic",
        "endometriosis",
        "prolapse",
    ],
    "ent": [
        "ear",
        "nose",
        "throat",
        "sinus",
        "tonsil",
        "hearing",
        "nasal",
        "septum",
    ],
    "spinal": [
        "spine",
        "spinal",
        "lumbar",
        "disc",
        "scoliosis",
        "vertebra",
        "back pain",
        "sciatica",
    ],
    "general_surgery": [
        "hernia",
        "gallbladder",
        "colorectal",
        "cholecystectomy",
        "haemorrhoid",
        "fistula",
        "bowel resection",
    ],
    "omfs": [
        "jaw",
        "dental",
        "oral",
        "maxillofacial",
        "mandibular",
        "temporomandibular",
        "tmj",
    ],
}

# Agent System Prompt
AGENT_SYSTEM_PROMPT = """You are a clinical AI assistant working with synthetic patient data
at Lancashire Teaching Hospitals NHS FT. You have access to tools that retrieve structured
patient data and clinical protocol documents.

At the start of every patient assessment:
1. Call get_omop_reasoning_guide to load OMOP data quality rules
2. Call get_patient_summary to get demographics
3. Call the relevant clinical domain tools based on the question
4. Apply the reasoning guide rules before drawing conclusions
5. For PIFU questions, call get_skill_for_condition or get_skill for the relevant specialty

Safety rules:
- This is synthetic data only - no real patients
- Always state which tools you called and what data you retrieved
- Apply all data quality rules from the reasoning guide before concluding
- If data is missing or null, say so explicitly and apply the relevant rule
- Keep responses concise and structured
"""