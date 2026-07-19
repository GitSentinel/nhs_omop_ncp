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
from datasets import Dataset

# PEFT Libraries
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
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
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Project Configuration
from config import (
    DATASET_PATH,
    FINETUNE_OUTPUT_DIR as OUTPUT_DIR,
    BASE_MODEL,
    MAX_LENGTH,
    QLORA_CONFIG,
    TRAINING_CONFIG,
    BNB_CONFIG_PARAMS,
    FINETUNE_PROMPT_TEMPLATE as PROMPT_TEMPLATE,
    MLFLOW_EXPERIMENT_FINETUNE,
)

from src.config.settings import settings

# Quantisation Configuration
BNB_CONFIG = BitsAndBytesConfig(**BNB_CONFIG_PARAMS)


def format_example(text: str, label: int, tokenizer) -> dict:
    # Prompt and Response Construction
    prompt = PROMPT_TEMPLATE.format(text=text[:600])
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
    max_prompt_len = MAX_LENGTH - len(response_ids)
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


def load_splits(tokenizer) -> tuple[Dataset, Dataset, dict]:
    # Dataset Loading
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Training Split Formatting
    train_ds = Dataset.from_list([
        format_example(example["text"], example["label"], tokenizer)
        for example in data["train"]
    ])

    # Test Split Formatting
    test_ds = Dataset.from_list([
        format_example(example["text"], example["label"], tokenizer)
        for example in data["test"]
    ])

    return train_ds, test_ds, data


def parse_generated_label(generated_text: str) -> int:
    # Label Parsing
    generated_text = generated_text.strip()

    if generated_text.startswith("1"):
        return 1

    return 0


def evaluate_generative(
    model,
    tokenizer,
    test_raw: list,
    max_examples: int = 200,
) -> dict:  
    # Evaluation Setup
    model.eval()

    predictions = []
    labels = []
    sample = test_raw[:max_examples]

    print(f"\nRunning generative evaluation on {len(sample):,} test examples...")

    # Prediction Loop
    for index, example in enumerate(sample):
        prompt = PROMPT_TEMPLATE.format(text=example["text"][:600])

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
            )

        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        predictions.append(parse_generated_label(generated))
        labels.append(int(example["label"]))

        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(sample)} evaluated...")

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
        "n_evaluated": len(sample),
    }


def print_run_header() -> None:
    # Device Summary
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        device_name = "CPU"
        vram_gb = 0.0

    # Run Header
    print("=" * 65)
    print("NHS OMOP MCP - Sprint 3 QLoRA Fine-Tuning")
    print("=" * 65)
    print(f"Base model : {BASE_MODEL}")
    print(f"Device     : {device_name}")
    print(f"VRAM       : {vram_gb:.1f} GB")
    print(f"Dataset    : {DATASET_PATH}")
    print()


def run_finetune() -> None:
    # GPU Check
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this 4-bit QLoRA run.")

    # Output Directory Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # MLflow Setup
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_FINETUNE)

    # Run Header
    print_run_header()

    # Tokeniser Loading
    print("Loading tokeniser...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Dataset Formatting
    print("Loading and formatting dataset...")

    train_ds, test_ds, raw_data = load_splits(tokenizer)

    print(f"Train: {len(train_ds):,}  |  Test: {len(test_ds):,}")

    # Model Loading
    print(f"\nLoading model in 4-bit: {BASE_MODEL}")

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

    # K-Bit Training Preparation
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # LoRA Adapter Setup
    lora_config = LoraConfig(**QLORA_CONFIG)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training Argument Setup
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        **TRAINING_CONFIG,
    )

    # Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
        ),
    )

    # MLflow Run
    with mlflow.start_run(
        run_name="qlora_qwen35_4b_generative_classifier",
        tags={
            "sprint": "sprint_3",
            "task": "generative_classification",
            "base_model": BASE_MODEL,
            "adapter": "qlora",
            "dataset": "delphi-100k clinic letters",
        },
    ):
        # Parameter Logging
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "approach": "generative_classification",
            "lora_r": QLORA_CONFIG["r"],
            "lora_alpha": QLORA_CONFIG["lora_alpha"],
            "target_modules": str(QLORA_CONFIG["target_modules"]),
            "epochs": TRAINING_CONFIG["num_train_epochs"],
            "effective_batch_size": (
                TRAINING_CONFIG["per_device_train_batch_size"]
                * TRAINING_CONFIG["gradient_accumulation_steps"]
            ),
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "quantisation": "4-bit NF4 double quant",
            "max_length": MAX_LENGTH,
            "n_train": len(train_ds),
            "n_test": len(test_ds),
        })

        # Model Training
        print("\nStarting QLoRA fine-tuning...")
        trainer.train()

        # Generative Evaluation
        eval_metrics = evaluate_generative(
            model,
            tokenizer,
            raw_data["test"],
        )

        # Metric Logging
        mlflow.log_metrics({
            "test_f1": eval_metrics["f1"],
            "test_accuracy": eval_metrics["accuracy"],
            "test_precision": eval_metrics["precision"],
            "test_recall": eval_metrics["recall"],
            "n_evaluated": eval_metrics["n_evaluated"],
        })

        # Report Logging
        print("\nClassification Report:")
        print(eval_metrics["report"])

        mlflow.log_text(
            eval_metrics["report"],
            "classification_report.txt",
        )

        mlflow.log_text(
            PROMPT_TEMPLATE,
            "prompt_template.txt",
        )

        # Adapter Saving
        model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        # Final Summary
        print("\n" + "=" * 65)
        print("FINE-TUNING COMPLETE")
        print("=" * 65)
        print(f"  Model      : {BASE_MODEL}")
        print(f"  F1 Score   : {eval_metrics['f1']:.4f}")
        print(f"  Accuracy   : {eval_metrics['accuracy']:.4f}")
        print(f"  Precision  : {eval_metrics['precision']:.4f}")
        print(f"  Recall     : {eval_metrics['recall']:.4f}")
        print(f"  Adapter    : {OUTPUT_DIR}")
        print("  MLflow     : http://127.0.0.1:5000")
        print("=" * 65)


if __name__ == "__main__":
    run_finetune()