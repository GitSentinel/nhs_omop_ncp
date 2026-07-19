import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_BIN   = str(Path(sys.executable))

# Data paths 
DUCKDB_PATH         = PROJECT_ROOT / "data" / "processed" / "omop_v54.duckdb"
FINETUNE_DIR        = PROJECT_ROOT / "data" / "finetune"
COHORT_OUTPUT_DIR   = PROJECT_ROOT / "data" / "cohort_run_outputs"
# DATASET_PATH        = FINETUNE_DIR / "clinic_letters_labelled.json"
DATASET_PATH = FINETUNE_DIR / "clinic_letters_azure_labelled.json"
AZURE_DATASET_PATH  = FINETUNE_DIR / "clinic_letters_azure_labelled.json"
FINETUNE_OUTPUT_DIR = FINETUNE_DIR / "qwen35_4b_lora_adapter"

# MCP server paths 
OMOP_SERVER_PATH   = PROJECT_ROOT / "src" / "mcp_server" / "server.py"
SKILLS_SERVER_PATH = PROJECT_ROOT / "src" / "mcp_server" / "skills_server.py"
SKILLS_DIR         = PROJECT_ROOT / "src" / "mcp_server" / "resources" / "skills"

# Azure OpenAI 
AZURE_ENDPOINT   = "https://openai-omop-dev-01.services.ai.azure.com/openai/v1"
AZURE_DEPLOYMENT = "gpt-5-nano"

# MLflow 
MLFLOW_TRACKING_URI        = "sqlite:///mlflow_runs/mlflow.db"
MLFLOW_EXPERIMENT_AGENT    = "nhs_omop_agent"
MLFLOW_EXPERIMENT_FINETUNE = "nhs_omop_finetune"

# Fine-tuning 
BASE_MODEL = "Qwen/Qwen3.5-4B"
MAX_LENGTH = 384

# QLoRA config
QLORA_CONFIG = dict(
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    target_modules  = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Training config
TRAINING_CONFIG = dict(
    num_train_epochs             = 3,
    per_device_train_batch_size  = 1,
    per_device_eval_batch_size   = 1,
    gradient_accumulation_steps  = 16,
    learning_rate                = 2e-4,
    warmup_ratio                 = 0.1,
    weight_decay                 = 0.01,
    eval_strategy                = "epoch",
    save_strategy                = "epoch",
    load_best_model_at_end       = True,
    metric_for_best_model        = "eval_loss",
    greater_is_better            = False,
    logging_steps                = 10,
    bf16                         = torch.cuda.is_bf16_supported(),
    fp16                         = not torch.cuda.is_bf16_supported(),
    dataloader_num_workers       = 0,
    report_to                    = "none",
    optim                        = "paged_adamw_8bit",
    max_grad_norm                = 1.0,
    gradient_checkpointing       = True,
    remove_unused_columns        = False,
)

# BNB config params
BNB_CONFIG_PARAMS = dict(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
)

# Finetune prompt template
FINETUNE_PROMPT_TEMPLATE = """You are a clinical text classifier for NHS outpatient letters.
Classify the following clinic letter as either a treatment event or routine follow-up.

Clinic letter:
{text}

Classification (output 0 for routine_followup or 1 for treatment_event):"""

# Dataset generation 
N_DATASET_SAMPLES = 6000

# Treatment keywords
TREATMENT_KEYWORDS = [
    "procedure", "operation", "surgery", "theatre",
    "treatment", "intervention", "injection", "biopsy",
    "excision", "repair", "removal", "transplant",
    "admitted", "admission", "inpatient",
    "referred for", "referral to", "discharged",
    "consent obtained", "consent signed",
    "scheduled for", "booked for", "listed for",
    "clock stop", "rtt", "18 week",
    "commenced treatment", "started treatment",
    "prescribed", "initiated therapy",
]

# Routine keywords
ROUTINE_KEYWORDS = [
    "routine follow-up", "follow up in", "review in",
    "stable", "no change", "unchanged",
    "monitoring", "surveillance", "watchful waiting",
    "continue current", "no new concerns",
    "appointment in", "clinic in",
    "doing well", "no intervention required",
    "conservative management", "reassured",
]

# Cohort run 
COHORT_QUERIES = [
    "Summarise this patient's medical history including conditions, medications and recent visits.",
    "What conditions does this patient have and what medications have been prescribed?",
    "What recent measurements and laboratory values are recorded for this patient?",
    "Provide a clinical summary of this patient's visit history and procedures.",
    "Based on this patient's conditions, which FastPIFU specialty skill is most relevant and what does it say about PIFU suitability?",
]

# FastPIFU skills
SPECIALTY_MAP = {
    "cardiology"              : "cardiology.md",
    "dermatology"             : "dermatology.md",
    "ent"                     : "ent.md",
    "gastroenterology"        : "gastroenterology.md",
    "general_surgery"         : "general_surgery.md",
    "gynaecology"             : "gynaecology.md",
    "omfs"                    : "omfs.md",
    "ophthalmology"           : "ophthalmology.md",
    "orthopaedics"            : "orthopaedics.md",
    "spinal"                  : "spinal.md",
    "urology"                 : "urology.md",
    "omop_clinical_reasoning" : "omop_clinical_reasoning.md",
}

# Condition specialty map
CONDITION_SPECIALTY_MAP = {
    "cardiology"       : ["heart", "cardiac", "arrhythmia", "atrial", "valve", "failure", "angina", "myocardial", "coronary", "pots", "lbbb", "bundle branch", "pericarditis", "brugada syndrome"],
    "gastroenterology" : ["liver", "cirrhosis", "hepatitis", "bowel", "crohn", "colitis", "gastro", "coeliac", "iron deficiency", "anaemia", "masld", "nafld", "ulcerative colitis"],
    "dermatology"      : ["skin", "dermat", "melanoma", "acne", "psoriasis", "eczema", "keratosis", "carcinoma", "mole"],
    "orthopaedics"     : ["joint", "knee", "hip", "shoulder", "fracture", "arthritis", "carpal", "tendon", "ligament", "bone", "spondylolisthesis"],
    "ophthalmology"    : ["eye", "ocular", "retina", "glaucoma", "cataract", "macular", "visual", "diabetic eye"],
    "urology"          : ["kidney", "bladder", "prostate", "urolog", "renal", "urinary", "incontinence"],
    "gynaecology"      : ["gynaecolog", "uterine", "ovarian", "pelvic", "endometriosis", "prolapse"],
    "ent"              : ["ear", "nose", "throat", "sinus", "tonsil", "hearing", "nasal", "septum"],
    "spinal"           : ["spine", "spinal", "lumbar", "disc", "scoliosis", "vertebra", "back pain", "sciatica"],
    "general_surgery"  : ["hernia", "gallbladder", "colorectal", "cholecystectomy", "haemorrhoid", "fistula", "bowel resection"],
    "omfs"             : ["jaw", "dental", "oral", "maxillofacial", "mandibular", "temporomandibular", "tmj"],
}

# Agent system prompt
AGENT_SYSTEM_PROMPT = """You are a clinical AI assistant working with synthetic OMOP CDM v5.4 patient data
at Lancashire Teaching Hospitals NHS FT. You have access to tools that retrieve structured
patient data and clinical protocol documents.

At the start of every patient assessment:
1. Call get_omop_reasoning_guide to load OMOP data quality rules
2. Call get_patient_summary to get demographics
3. Call the relevant clinical domain tools based on the question
4. Apply the reasoning guide rules before drawing conclusions
5. For PIFU questions, call get_skill_for_condition or get_skill for the relevant specialty

Important:
- This is synthetic data only — no real patients
- Always state which tools you called and what data you retrieved
- Apply all data quality rules from the reasoning guide before concluding
- If data is missing or null, say so explicitly and apply the relevant rule
- Keep responses concise and structured
"""