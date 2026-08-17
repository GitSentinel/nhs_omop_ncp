"""
Generate Provisional Source Clinic-Letter Dataset

Extracts synthetic OMOP clinic letters and assigns provisional heuristic labels before Azure OpenAI re-labelling.

Run:
uv run python src/scripts/generate_dataset.py
"""

import json
import math
import sys
from pathlib import Path

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project Configuration
from config import (
    SOURCE_DATASET_PATH,
    N_DATASET_SAMPLES,
    TREATMENT_KEYWORDS,
    ROUTINE_KEYWORDS,
)

# OMOP Data Access
from src.data_access.connection import get_table

# Dataset Configuration
MIN_NOTE_CHARS = 100
MAX_NOTE_CHARS = 2000
TEXT_CHAR_LIMIT = 1000
FETCH_MULTIPLIER = 3
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42

# Output Configuration
OUTPUT_PATH = (
    SOURCE_DATASET_PATH
    if SOURCE_DATASET_PATH.suffix
    else SOURCE_DATASET_PATH / "clinic_letters_labelled.json"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalise_label(value) -> int | None:
    """Convert a provisional label value into 0, 1, or None."""

    # Missing Label Handling
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    # Binary Label Handling
    try:
        label = int(value)
    except (TypeError, ValueError):
        return None

    if label in (0, 1):
        return label

    return None


def label_note(note_text: str) -> int | None:
    """Assign a provisional keyword-based label to one clinic letter."""

    # Text Normalisation
    text_lower = str(note_text).lower()

    # Keyword Scoring
    treatment_score = sum(keyword in text_lower for keyword in TREATMENT_KEYWORDS)

    routine_score = sum(keyword in text_lower for keyword in ROUTINE_KEYWORDS)

    # Provisional Label Assignment
    if treatment_score > routine_score:
        return 1

    if routine_score > treatment_score:
        return 0

    return None


def label_name(label: int | None) -> str:
    """Return the text label name for a provisional class."""

    # Label Name Mapping
    return {
        0: "routine_followup",
        1: "treatment_event",
        None: "uncertain",
    }[label]


def format_sample(row) -> dict:
    """Convert a dataframe row into the source JSON format."""

    # Provisional Label Formatting
    label = normalise_label(row.label)

    return {
        "note_id": int(row.note_id),
        "person_id": int(row.person_id),
        "note_date": str(row.note_date),
        "text": str(row.note_text)[:TEXT_CHAR_LIMIT],
        "label": label,
        "label_name": label_name(label),
    }


def generate_dataset(n_samples: int = N_DATASET_SAMPLES) -> None:
    """Generate and save the provisional source clinic-letter dataset."""

    # Run Setup
    print(f"Generating provisional dataset ({n_samples:,} samples)...")

    # OMOP Note Extraction
    note = get_table("note")

    notes_df = (
        note.select(
            "note_id",
            "person_id",
            "note_date",
            "note_text",
        )
        .filter(note.note_text.notnull())
        .order_by(note.note_id)
        .limit(n_samples * FETCH_MULTIPLIER)
        .execute()
    )

    print(f"  Fetched {len(notes_df):,} notes from DuckDB")

    # Note Length Filtering
    notes_df["note_length"] = notes_df["note_text"].str.len()

    notes_df = notes_df[
        notes_df["note_length"].between(
            MIN_NOTE_CHARS,
            MAX_NOTE_CHARS,
        )
    ].copy()

    print(
        f"  After length filter ({MIN_NOTE_CHARS}-{MAX_NOTE_CHARS} characters): "
        f"{len(notes_df):,} notes"
    )

    if len(notes_df) < n_samples:
        raise ValueError(
            f"Requested {n_samples:,} samples, but only "
            f"{len(notes_df):,} valid notes remain after filtering."
        )

    # Provisional Heuristic Labelling
    notes_df["label"] = notes_df["note_text"].apply(label_note)

    # Reproducible Sampling
    samples = notes_df.sample(
        n=n_samples,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    # Label Distribution Summary
    label_counts = (
        samples["label"]
        .value_counts(
            dropna=True,
        )
        .sort_index()
    )

    uncertain_count = int(samples["label"].isna().sum())

    print(f"  Provisional label distribution: {label_counts.to_dict()}")
    print(f"  Provisional uncertain labels  : {uncertain_count:,}")

    # Source Split Retained for Backward Compatibility
    n_train = int(len(samples) * TRAIN_FRACTION)

    train = samples.iloc[:n_train]
    test = samples.iloc[n_train:]

    # Dataset Structure
    dataset = {
        "task": "binary_classification",
        "description": (
            "Synthetic OMOP clinic letters with provisional heuristic "
            "labels. Final labels are assigned by relabel_dataset.py."
        ),
        "label_map": {
            "0": "routine_followup",
            "1": "treatment_event",
            "null": "uncertain",
        },
        "n_total": len(samples),
        "n_train": len(train),
        "n_test": len(test),
        "n_uncertain": uncertain_count,
        "train": [format_sample(row) for _, row in train.iterrows()],
        "test": [format_sample(row) for _, row in test.iterrows()],
    }

    # Dataset Saving
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Dataset Summary
    print()
    print("=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total samples       : {len(samples):,}")
    print(f"  Training samples    : {len(train):,}")
    print(f"  Test samples        : {len(test):,}")
    print(f"  Routine follow-up   : {int(label_counts.get(0, 0)):,}")
    print(f"  Treatment event     : {int(label_counts.get(1, 0)):,}")
    print(f"  Uncertain           : {uncertain_count:,}")
    print(f"  Output path         : {OUTPUT_PATH}")

    # Sample Inspection
    print("\nSample training examples:")

    for example in dataset["train"][:3]:
        print(f"\n  [{example['label_name'].upper()}]")
        print(f"  {example['text'][:200]}...")


if __name__ == "__main__":
    generate_dataset(n_samples=N_DATASET_SAMPLES)
