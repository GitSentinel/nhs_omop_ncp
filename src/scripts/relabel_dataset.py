"""
Asynchronous Azure OpenAI Re-Labelling

Re-labels clinic letters using Azure OpenAI, stores unresolved cases, and creates patient-level grouped train, validation and test splits.

Run:
uv run python src/scripts/relabel_dataset.py
"""

import asyncio
import json
import random
import sys
from pathlib import Path

# Data Processing Libraries
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# LLM Client
from langchain_openai import ChatOpenAI

# Project Configuration
from config import (
    AZURE_DATASET_PATH,
    SOURCE_DATASET_PATH,
    N_DATASET_SAMPLES,
)

from src.config.settings import settings

# Re-Labelling Configuration
MAX_CONCURRENCY = 8
MAX_RETRIES = 6
CHECKPOINT_EVERY = 50
BASE_RETRY_SECONDS = 1.0
TEXT_CHAR_LIMIT = 1000

# Label Prompt
LABEL_PROMPT = """You are a clinical coding expert for NHS outpatient services.

Classify the clinic letter as exactly one of:
- 0 = routine_followup: stable monitoring, surveillance or review only, with no new treatment, treatment change, escalation, procedure or action required.
- 1 = treatment_event: any new or changed medication or treatment, procedure booked or performed, referral for treatment, escalation, urgent review, abnormal result requiring action, intervention, or discharge following treatment.
- U = uncertain: the evidence is incomplete, conflicting or genuinely ambiguous.

Safety rules:
- Follow-up wording alone does not make the letter routine.
- Continued unchanged medication with stable monitoring can be 0.
- Any new medication, dose change, procedure, referral, escalation or active clinical action is 1.
- Do not force an unclear case into class 0; use U.

Clinic letter:
{text}

Reply with ONLY 0, 1 or U."""


def resolve_dataset_path(path: Path, default_filename: str) -> Path:
    """Resolve a dataset path that may point to a file or directory."""

    # Path Resolution
    if path.suffix:
        return path

    return path / default_filename


def make_llm() -> ChatOpenAI:
    """Create the Azure OpenAI labelling model."""

    # Azure OpenAI Client Setup
    return ChatOpenAI(
        model=settings.azure_openai_deployment,
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0,
        max_retries=0,
    )


def parse_label(content: str) -> int | None:
    """Parse a strict Azure OpenAI label response."""

    # Strict Label Parsing
    value = str(content).strip().upper()

    if value == "0":
        return 0

    if value == "1":
        return 1

    if value == "U":
        return None

    raise ValueError(f"Unexpected label response: {value!r}")


def parse_original_label(value) -> int | None:
    """Parse the original heuristic label when available."""

    # Original Label Parsing
    try:
        label = int(value)
    except (TypeError, ValueError):
        return None

    if label in (0, 1):
        return label

    return None


async def label_with_azure(
    text: str,
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
) -> tuple[int | None, bool, str | None]:
    """Label one clinic letter using Azure OpenAI."""

    # Prompt Construction
    prompt = LABEL_PROMPT.format(text=str(text)[:TEXT_CHAR_LIMIT])

    # Retry Loop
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await llm.ainvoke(prompt)

            return parse_label(response.content), False, None

        except Exception as error:
            if attempt == MAX_RETRIES - 1:
                return None, True, type(error).__name__

            wait_time = min(
                60.0,
                BASE_RETRY_SECONDS * (2**attempt) + random.uniform(0.0, 1.0),
            )

            await asyncio.sleep(wait_time)

    return None, True, "UnknownError"


def load_source_samples(n_samples: int) -> list[dict]:
    """Load source examples from the heuristic-labelled dataset."""

    # Source Path Setup
    source_path = resolve_dataset_path(
        SOURCE_DATASET_PATH,
        "clinic_letters_labelled.json",
    )

    # Dataset Loading
    with open(source_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Sample Selection
    all_samples = data["train"] + data["test"]

    return all_samples[: min(n_samples, len(all_samples))]


def build_relabelled_example(
    example: dict,
    new_label: int,
) -> dict:
    """Build one resolved re-labelled example."""

    # Original Label Metadata
    original_label = parse_original_label(example.get("label"))

    label_changed = original_label is not None and new_label != original_label

    return {
        "note_id": example["note_id"],
        "person_id": example["person_id"],
        "note_date": example.get("note_date"),
        "text": example["text"],
        "label": new_label,
        "label_name": ("treatment_event" if new_label == 1 else "routine_followup"),
        "original_label": original_label,
        "label_changed": label_changed,
        "label_failed": False,
    }


def validate_group_split_input(frame: pd.DataFrame) -> None:
    """Validate data before grouped stratified splitting."""

    # Split Input Validation
    required_columns = {"person_id", "label"}

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(f"Missing required split columns: {missing_columns}")

    if frame["person_id"].nunique() < 8:
        raise ValueError(
            "At least eight distinct person_id values are required "
            "for the grouped train, validation and test split."
        )

    if frame["label"].nunique() < 2:
        raise ValueError("Both labels 0 and 1 are required for stratified splitting.")


def split_by_person(
    examples: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Create patient-level train, validation and test splits."""

    # Patient-Level Split Setup
    frame = pd.DataFrame(examples)

    validate_group_split_input(frame)

    # Outer Train-Test Split
    outer_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    train_validation_indices, test_indices = next(
        outer_splitter.split(
            frame,
            y=frame["label"],
            groups=frame["person_id"],
        )
    )

    train_validation = frame.iloc[train_validation_indices].reset_index(drop=True)

    test = frame.iloc[test_indices].reset_index(drop=True)

    # Inner Train-Validation Split
    inner_splitter = StratifiedGroupKFold(
        n_splits=8,
        shuffle=True,
        random_state=42,
    )

    train_indices, validation_indices = next(
        inner_splitter.split(
            train_validation,
            y=train_validation["label"],
            groups=train_validation["person_id"],
        )
    )

    train = train_validation.iloc[train_indices].reset_index(drop=True)
    validation = train_validation.iloc[validation_indices].reset_index(drop=True)

    # Leakage Check
    train_people = set(train["person_id"])
    validation_people = set(validation["person_id"])
    test_people = set(test["person_id"])

    if train_people & validation_people:
        raise RuntimeError("person_id leakage between train and validation.")

    if train_people & test_people:
        raise RuntimeError("person_id leakage between train and test.")

    if validation_people & test_people:
        raise RuntimeError("person_id leakage between validation and test.")

    return (
        train.to_dict(orient="records"),
        validation.to_dict(orient="records"),
        test.to_dict(orient="records"),
    )


def load_checkpoint(checkpoint_path: Path) -> dict[str, dict]:
    """Load completed re-labelling records from a checkpoint file."""

    # Checkpoint Loading
    completed = {}

    if not checkpoint_path.exists():
        return completed

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            completed[str(record["note_id"])] = record

    return completed


def append_checkpoint(
    checkpoint_path: Path,
    record: dict,
) -> None:
    """Append one completed re-labelling record to the checkpoint file."""

    # Checkpoint Saving
    with open(checkpoint_path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


async def label_example(
    example: dict,
    llm: ChatOpenAI,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Label one source example and return checkpoint metadata."""

    # Single Example Labelling
    new_label, label_failed, error_type = await label_with_azure(
        example["text"],
        llm,
        semaphore,
    )

    if new_label is None:
        return {
            "note_id": example["note_id"],
            "status": "unresolved",
            "label_failed": label_failed,
            "error_type": error_type,
            "unresolved_reason": (
                "api_failure" if label_failed else "labeller_uncertain"
            ),
        }

    return {
        "note_id": example["note_id"],
        "status": "resolved",
        "label": new_label,
        "label_failed": False,
        "error_type": None,
    }


async def relabel_async(
    n_samples: int = N_DATASET_SAMPLES,
) -> None:
    """Run asynchronous re-labelling and save the final dataset."""

    # Output Paths
    azure_dataset_path = resolve_dataset_path(
        AZURE_DATASET_PATH,
        "clinic_letters_azure_labelled.json",
    )

    checkpoint_path = azure_dataset_path.with_suffix(".progress.jsonl")

    azure_dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Run Setup
    print(
        "Re-labelling "
        f"{n_samples:,} samples with Azure OpenAI "
        f"{settings.azure_openai_deployment}..."
    )
    print(f"  Concurrent requests : {MAX_CONCURRENCY}")
    print(f"  Checkpoint           : {checkpoint_path}")

    samples = load_source_samples(n_samples)

    if not samples:
        raise ValueError("No source samples found for re-labelling.")

    # Checkpoint Resume Setup
    completed = load_checkpoint(checkpoint_path)

    pending_samples = [
        example for example in samples if str(example["note_id"]) not in completed
    ]

    print(f"  Already completed   : {len(completed):,}")
    print(f"  Remaining           : {len(pending_samples):,}")

    llm = make_llm()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [
        asyncio.create_task(
            label_example(
                example,
                llm,
                semaphore,
            )
        )
        for example in pending_samples
    ]

    newly_completed = 0

    # Asynchronous Labelling Loop
    for task in asyncio.as_completed(tasks):
        record = await task

        completed[str(record["note_id"])] = record
        append_checkpoint(checkpoint_path, record)

        newly_completed += 1
        total_completed = len(completed)

        if newly_completed % CHECKPOINT_EVERY == 0 or total_completed == len(samples):
            n_resolved = sum(
                item["status"] == "resolved" for item in completed.values()
            )

            n_unresolved = total_completed - n_resolved

            print(
                f"  {total_completed:,}/{len(samples):,}  |  "
                f"resolved: {n_resolved:,}  "
                f"unresolved: {n_unresolved:,}"
            )

    # Re-Labelled Dataset Assembly
    relabelled = []
    unresolved = []
    label_counts = {0: 0, 1: 0}
    n_changed = 0
    n_failed = 0
    n_uncertain = 0

    sample_lookup = {str(example["note_id"]): example for example in samples}

    for example in samples:
        note_id = str(example["note_id"])
        result = completed[note_id]

        if result["status"] == "unresolved":
            unresolved.append(
                {
                    **example,
                    "label_failed": result["label_failed"],
                    "error_type": result.get("error_type"),
                    "unresolved_reason": result["unresolved_reason"],
                }
            )

            n_failed += int(result["label_failed"])
            n_uncertain += int(not result["label_failed"])

            continue

        relabelled_example = build_relabelled_example(
            sample_lookup[note_id],
            int(result["label"]),
        )

        relabelled.append(relabelled_example)
        label_counts[relabelled_example["label"]] += 1
        n_changed += int(relabelled_example["label_changed"])

    if not relabelled:
        raise ValueError("No resolved binary labels were produced.")

    # Patient-Level Dataset Splitting
    train, validation, test = split_by_person(relabelled)

    dataset = {
        "task": "binary_classification",
        "description": (
            f"Azure OpenAI {settings.azure_openai_deployment} "
            "labelled clinic letters with patient-level grouped splits"
        ),
        "labeller": settings.azure_openai_deployment,
        "label_map": {
            "0": "routine_followup",
            "1": "treatment_event",
        },
        "max_concurrency": MAX_CONCURRENCY,
        "n_changed": n_changed,
        "n_failed": n_failed,
        "n_uncertain": n_uncertain,
        "n_unresolved": len(unresolved),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "train": train,
        "validation": validation,
        "test": test,
        "unresolved": unresolved,
    }

    # Dataset Saving
    with open(
        azure_dataset_path,
        "w",
        encoding="utf-8",
    ) as file:
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
    print(f"  Resolved samples    : {len(relabelled):,}")
    print(f"  Labels changed      : {n_changed:,}")
    print(f"  Failed labels       : {n_failed:,}")
    print(f"  Uncertain labels    : {n_uncertain:,}")
    print(f"  Label 0 (routine)   : {label_counts[0]:,}")
    print(f"  Label 1 (treatment) : {label_counts[1]:,}")
    print(f"  Train               : {len(train):,}")
    print(f"  Validation          : {len(validation):,}")
    print(f"  Test                : {len(test):,}")
    print(f"  Output              : {azure_dataset_path}")
    print(f"  Resume checkpoint   : {checkpoint_path}")


def relabel(
    n_samples: int = N_DATASET_SAMPLES,
) -> None:
    """Run the asynchronous re-labelling workflow."""

    # Async Entry Point
    asyncio.run(relabel_async(n_samples))


if __name__ == "__main__":
    relabel()
