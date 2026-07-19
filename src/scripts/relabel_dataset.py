import json
import sys
import time
from pathlib import Path

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# LLM Client
from langchain_openai import ChatOpenAI

# Project Configuration
from config import (
    AZURE_DATASET_PATH,
    DATASET_PATH,
)

from src.config.settings import settings

# Label Prompt
LABEL_PROMPT = """You are a clinical coding expert for NHS outpatient services.

Read this clinic letter carefully and classify it as ONE of:
- 0 = routine_followup (monitoring only, surveillance, review appointment, no treatment decision, watchful waiting, stable condition review)
- 1 = treatment_event (procedure booked or performed, treatment started or changed, referral for treatment, discharge after treatment, surgery scheduled, injection given, biopsy performed)

Rules:
- A patient "presenting for follow-up" is routine (0) unless a new treatment decision is made
- A prescription being CONTINUED is routine (0); a NEW prescription is treatment (1)
- Vaccination alone is routine (0)
- If genuinely unclear, choose 0

Clinic letter:
{text}

Reply with ONLY the digit 0 or 1. Nothing else."""


def make_llm() -> ChatOpenAI:
    # Azure OpenAI Client Setup
    return ChatOpenAI(
        model=settings.azure_openai_deployment,
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0,
    )


def parse_label(content: str) -> int:
    # Label Parsing
    content = content.strip()

    if content.startswith("1"):
        return 1

    return 0


def label_with_azure(
    text: str,
    llm: ChatOpenAI,
    max_retries: int = 3,
) -> tuple[int, bool]:
    # Prompt Construction
    prompt = LABEL_PROMPT.format(text=text[:800])

    # Retry Loop
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return parse_label(response.content), False

        except Exception as error:
            wait_time = 1 + attempt

            print(
                f"  Labelling error on attempt {attempt + 1}: "
                f"{type(error).__name__}. Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    return 0, True


def load_source_samples(n_samples: int) -> list[dict]:
    # Dataset Loading
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Sample Selection
    all_samples = data["train"] + data["test"]

    return all_samples[:min(n_samples, len(all_samples))]


def build_relabelled_example(
    example: dict,
    new_label: int,
    label_failed: bool,
) -> dict:
    # Label Metadata
    original_label = int(example["label"])
    label_changed = new_label != original_label

    return {
        "note_id": example["note_id"],
        "person_id": example["person_id"],
        "note_date": example.get("note_date"),
        "text": example["text"],
        "label": new_label,
        "label_name": (
            "treatment_event"
            if new_label == 1
            else "routine_followup"
        ),
        "original_label": original_label,
        "label_changed": label_changed,
        "label_failed": label_failed,
    }


def relabel(n_samples: int = 1200) -> None:
    # Run Setup
    print(
        "Re-labelling "
        f"{n_samples:,} samples with Azure OpenAI "
        f"{settings.azure_openai_deployment}..."
    )

    samples = load_source_samples(n_samples)
    llm = make_llm()

    relabelled = []
    label_counts = {0: 0, 1: 0}
    n_changed = 0
    n_failed = 0

    # Re-Labelling Loop
    for index, example in enumerate(samples):
        new_label, label_failed = label_with_azure(example["text"], llm)

        relabelled_example = build_relabelled_example(
            example,
            new_label,
            label_failed,
        )

        relabelled.append(relabelled_example)

        label_counts[new_label] += 1
        n_changed += int(relabelled_example["label_changed"])
        n_failed += int(label_failed)

        if (index + 1) % 100 == 0:
            print(
                f"  {index + 1:,}/{len(samples):,}  |  "
                f"0: {label_counts[0]:,}  "
                f"1: {label_counts[1]:,}  "
                f"changed: {n_changed:,}  "
                f"failed: {n_failed:,}"
            )

        time.sleep(0.05)

    # Train and Test Split
    n_train = int(len(relabelled) * 0.8)

    dataset = {
        "task": "binary_classification",
        "description": (
            f"Azure OpenAI {settings.azure_openai_deployment} "
            "labelled clinic letters"
        ),
        "labeller": settings.azure_openai_deployment,
        "label_map": {
            "0": "routine_followup",
            "1": "treatment_event",
        },
        "n_changed": n_changed,
        "n_failed": n_failed,
        "n_train": n_train,
        "n_test": len(relabelled) - n_train,
        "train": relabelled[:n_train],
        "test": relabelled[n_train:],
    }

    # Dataset Saving
    AZURE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(AZURE_DATASET_PATH, "w", encoding="utf-8") as file:
        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Final Summary
    print(f"\n{'=' * 60}")
    print("RE-LABELLING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total samples       : {len(relabelled):,}")
    print(f"  Labels changed      : {n_changed:,} ({n_changed / len(relabelled) * 100:.1f}%)")
    print(f"  Failed labels       : {n_failed:,}")
    print(f"  Label 0 (routine)   : {label_counts[0]:,}")
    print(f"  Label 1 (treatment) : {label_counts[1]:,}")
    print(f"  Output              : {AZURE_DATASET_PATH}")


if __name__ == "__main__":
    relabel(n_samples=1200)