import json
import random
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

# Reproducibility
random.seed(42)

# Output Configuration
OUTPUT_PATH = (
    SOURCE_DATASET_PATH
    if SOURCE_DATASET_PATH.suffix
    else SOURCE_DATASET_PATH / "clinic_letters_labelled.json"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def label_note(note_text: str) -> int:
    # Text Normalisation
    text_lower = note_text.lower()

    # Keyword Scoring
    treatment_score = sum(
        keyword in text_lower
        for keyword in TREATMENT_KEYWORDS
    )

    routine_score = sum(
        keyword in text_lower
        for keyword in ROUTINE_KEYWORDS
    )

    # Label Assignment
    if treatment_score > routine_score:
        return 1

    if routine_score > treatment_score:
        return 0

    return random.choice([0, 1])


def format_sample(row) -> dict:
    # Label Formatting
    label = int(row.label)

    return {
        "note_id": int(row.note_id),
        "person_id": int(row.person_id),
        "note_date": str(row.note_date),
        "text": row.note_text[:1000],
        "label": label,
        "label_name": (
            "treatment_event"
            if label == 1
            else "routine_followup"
        ),
    }


def generate_dataset(n_samples: int = 600) -> None:
    # Run Setup
    print(f"Generating labelled dataset ({n_samples:,} samples)...")

    # OMOP Note Extraction
    note = get_table("note")

    notes_df = (
        note
        .select(
            "note_id",
            "person_id",
            "note_date",
            "note_text",
        )
        .filter(note.note_text.notnull())
        .order_by(note.note_id)
        .limit(n_samples * 3)
        .execute()
    )

    print(f"  Fetched {len(notes_df):,} notes from DuckDB")

    # Note Length Filtering
    notes_df["note_length"] = notes_df["note_text"].str.len()

    notes_df = notes_df[
        notes_df["note_length"].between(100, 2000)
    ].copy()

    print(
        "  After length filter (100-2000 characters): "
        f"{len(notes_df):,} notes"
    )

    if len(notes_df) < n_samples:
        raise ValueError(
            f"Requested {n_samples:,} samples, but only "
            f"{len(notes_df):,} valid notes remain after filtering."
        )

    # Heuristic Label Generation
    notes_df["label"] = notes_df["note_text"].apply(label_note)

    # Reproducible Sampling
    samples = notes_df.sample(
        n=n_samples,
        random_state=42,
    ).reset_index(drop=True)

    label_counts = samples["label"].value_counts().sort_index()

    print(f"  Label distribution: {label_counts.to_dict()}")

    # Train and Test Split
    n_train = int(len(samples) * 0.8)

    train = samples.iloc[:n_train]
    test = samples.iloc[n_train:]

    # Dataset Structure
    dataset = {
        "task": "binary_classification",
        "description": (
            "Classify synthetic OMOP clinic letters as treatment "
            "event (1) or routine follow-up (0)"
        ),
        "label_map": {
            "0": "routine_followup",
            "1": "treatment_event",
        },
        "n_train": len(train),
        "n_test": len(test),
        "train": [
            format_sample(row)
            for _, row in train.iterrows()
        ],
        "test": [
            format_sample(row)
            for _, row in test.iterrows()
        ],
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
    print(f"  Output path         : {OUTPUT_PATH}")

    # Sample Inspection
    print("\nSample training examples:")

    for example in dataset["train"][:3]:
        print(f"\n  [{example['label_name'].upper()}]")
        print(f"  {example['text'][:200]}...")


if __name__ == "__main__":
    generate_dataset(n_samples=N_DATASET_SAMPLES)