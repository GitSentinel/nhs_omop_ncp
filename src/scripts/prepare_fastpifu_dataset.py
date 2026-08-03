"""
FastPIFU cardiology dataset preparation.

Extracts the supplied FastPIFU cardiology ZIP file, converts source files into a consistent three-class JSON format, removes external-test leakage, and writes train, validation, external-test and challenge splits.

Run:
uv run python src/scripts/prepare_fastpifu_dataset.py --force
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

# Data Splitting
from sklearn.model_selection import train_test_split

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PIFU Configuration
from src.config.pifu_settings import (
    PIFU_CHALLENGE_PATH,
    PIFU_EXTRACTED_DIR,
    PIFU_LABEL_TO_ID,
    PIFU_PROCESSED_DIR,
    PIFU_SUMMARY_PATH,
    PIFU_TEST_PATH,
    PIFU_TRAIN_PATH,
    PIFU_VALIDATION_PATH,
    PIFU_ZIP_PATH,
)


def normalised_hash(text: str) -> str:
    """Return a stable hash after whitespace and case normalisation."""

    # Text Normalisation
    normalised = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip().lower()

    return hashlib.sha256(
        normalised.encode("utf-8")
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dictionaries."""

    # JSONL Loading
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL in {path} at line "
                    f"{line_number}: {error}"
                ) from error

    if not rows:
        raise ValueError(f"No rows found in JSONL file: {path}")

    return rows


def read_json(path: Path) -> list[dict]:
    """Read a JSON file expected to contain a list of records."""

    # JSON Loading
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")

    if not payload:
        raise ValueError(f"{path} contains no records.")

    return payload


def class_counts(samples: list[dict]) -> dict:
    """Return class counts by label name."""

    # Class Count Calculation
    return dict(
        Counter(
            sample["label_name"]
            for sample in samples
        )
    )


def write_dataset(
    path: Path,
    split_name: str,
    samples: list[dict],
) -> None:
    """Write a processed PIFU split to JSON."""

    # Dataset Payload
    payload = {
        "task": "pifu_eligibility_3class",
        "split": split_name,
        "label_map": PIFU_LABEL_TO_ID,
        "n_samples": len(samples),
        "class_counts": class_counts(samples),
        "samples": samples,
    }

    # Dataset Saving
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def is_ignored_zip_member(member_name: str) -> bool:
    """Return whether a ZIP member should be ignored."""

    # System File Filtering
    return (
        member_name.startswith("__MACOSX/")
        or "/.DS_Store" in member_name
        or member_name.endswith("/.DS_Store")
    )


def safe_zip_members(
    archive: zipfile.ZipFile,
    target_dir: Path,
) -> list[zipfile.ZipInfo]:
    """Return ZIP members after filtering and path traversal checks."""

    # Safe Member Selection
    safe_members = []
    target_root = target_dir.resolve()

    for member in archive.infolist():
        if is_ignored_zip_member(member.filename):
            continue

        destination = (target_root / member.filename).resolve()

        try:
            destination.relative_to(target_root)

        except ValueError as error:
            raise RuntimeError(
                f"Unsafe ZIP member path detected: {member.filename}"
            ) from error

        safe_members.append(member)

    return safe_members


def extract_zip(
    source_zip: Path,
    force: bool,
) -> None:
    """Extract the FastPIFU ZIP file into the configured directory."""

    # ZIP Existence Check
    if not source_zip.exists():
        raise FileNotFoundError(
            f"FastPIFU ZIP not found: {source_zip}"
        )

    # Forced Re-Extraction
    if force and PIFU_EXTRACTED_DIR.exists():
        shutil.rmtree(PIFU_EXTRACTED_DIR)

    PIFU_EXTRACTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Safe ZIP Extraction
    with zipfile.ZipFile(source_zip) as archive:
        members = safe_zip_members(
            archive,
            PIFU_EXTRACTED_DIR,
        )

        archive.extractall(
            PIFU_EXTRACTED_DIR,
            members=members,
        )


def locate_required_files() -> dict[str, Path]:
    """Locate required files inside the extracted FastPIFU directory."""

    # Required Source Files
    required_names = {
        "training": "pifu_synthetic_training.jsonl",
        "external_test": "eval_sample_150_v3.json",
        "edge_cases": "pifu_edge_cases.jsonl.json",
        "hard_negatives": "pifu_hard_negatives.jsonl.json",
    }

    located = {}

    # File Discovery
    for key, filename in required_names.items():
        matches = list(PIFU_EXTRACTED_DIR.rglob(filename))

        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {filename} below "
                f"{PIFU_EXTRACTED_DIR}, found {len(matches)}."
            )

        located[key] = matches[0]

    return located


def convert_training_row(row: dict) -> dict:
    """Convert one synthetic training row into the processed format."""

    # Label Extraction
    assessment = row["expected_output"]["pifu_assessment"]
    label_name = str(assessment["eligibility"]).upper()

    if label_name not in PIFU_LABEL_TO_ID:
        raise ValueError(
            f"Unexpected training label: {label_name}"
        )

    # Metadata Extraction
    metadata = dict(row.get("metadata") or {})

    return {
        "sample_id": str(
            metadata.get("document_id")
            or normalised_hash(row["input"])[:16]
        ),
        "text": str(row["input"]),
        "label": PIFU_LABEL_TO_ID[label_name],
        "label_name": label_name,
        "source": "pifu_synthetic_training",
        "metadata": metadata,
    }


def convert_external_test_row(row: dict) -> dict:
    """Convert one external-test row into the processed format."""

    # Label Extraction
    label_name = str(row["_class"]).upper()

    if label_name not in PIFU_LABEL_TO_ID:
        raise ValueError(
            f"Unexpected external test label: {label_name}"
        )

    return {
        "sample_id": f"eval_{row['_idx']}",
        "text": str(row["input"]),
        "label": PIFU_LABEL_TO_ID[label_name],
        "label_name": label_name,
        "source": "eval_sample_150_v3",
        "metadata": dict(row.get("metadata") or {}),
    }


def challenge_label(row: dict) -> str:
    """Map a challenge row to a PIFU eligibility label."""

    # Clock-Stop Rule
    event_type = row.get("clock_stop_event_type")

    if (
        row.get("is_clock_stop") is True
        and event_type == "Active Monitoring - Patient Initiated (PIFU)"
    ):
        return "ELIGIBLE"

    # Challenge Label Rule
    return "NOT_ELIGIBLE"


def convert_challenge_row(
    row: dict,
    source: str,
) -> dict:
    """Convert one edge-case or hard-negative row."""

    # Challenge Label Assignment
    label_name = challenge_label(row)

    return {
        "sample_id": str(row["document_id"]),
        "text": str(row["document_content"]),
        "label": PIFU_LABEL_TO_ID[label_name],
        "label_name": label_name,
        "source": source,
        "metadata": {
            "document_type": row.get("document_type"),
            "specialty": row.get("specialty"),
            "referral_reason": row.get("referral_reason"),
            "clock_stop_event_type": row.get("clock_stop_event_type"),
            "non_clock_stop_scenario": row.get("non_clock_stop_scenario"),
            "hard_negative": row.get("hard_negative", False),
            "expected_reasoning": row.get("expected_reasoning"),
        },
    }


def assert_no_text_overlap(
    first_name: str,
    first_samples: list[dict],
    second_name: str,
    second_samples: list[dict],
) -> None:
    """Raise an error if two splits contain duplicate letter text."""

    # Hash-Based Leakage Check
    first_hashes = {
        normalised_hash(sample["text"])
        for sample in first_samples
    }

    second_hashes = {
        normalised_hash(sample["text"])
        for sample in second_samples
    }

    overlap = first_hashes & second_hashes

    if overlap:
        raise RuntimeError(
            f"Text leakage detected between {first_name} "
            f"and {second_name}: {len(overlap)} duplicate letters."
        )


def remove_external_test_overlap(
    training_all: list[dict],
    external_test: list[dict],
) -> tuple[list[dict], int]:
    """Remove external-test letters from the training pool."""

    # External Hash Set
    external_hashes = {
        normalised_hash(sample["text"])
        for sample in external_test
    }

    # Leakage Removal
    training_pool = [
        sample
        for sample in training_all
        if normalised_hash(sample["text"]) not in external_hashes
    ]

    removed_overlap = len(training_all) - len(training_pool)

    return training_pool, removed_overlap


def split_training_pool(
    training_pool: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Create stratified train and validation splits."""

    # Split Validation
    if len(training_pool) < 10:
        raise ValueError(
            "At least 10 training samples are required for splitting."
        )

    labels = [
        sample["label"]
        for sample in training_pool
    ]

    if len(set(labels)) < 2:
        raise ValueError(
            "At least two classes are required for stratified splitting."
        )

    # Stratified Split
    train_samples, validation_samples = train_test_split(
        training_pool,
        test_size=0.10,
        random_state=42,
        stratify=labels,
    )

    return train_samples, validation_samples


def write_summary(
    summary: dict,
) -> None:
    """Write the dataset preparation summary JSON."""

    # Summary Saving
    PIFU_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PIFU_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )


def prepare(source_zip: Path, force: bool) -> None:
    """Prepare all FastPIFU cardiology processed datasets."""

    # Processed Directory Setup
    PIFU_PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Source ZIP Copy
    if source_zip.resolve() != PIFU_ZIP_PATH.resolve():
        PIFU_ZIP_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_zip,
            PIFU_ZIP_PATH,
        )

    # Source Extraction
    extract_zip(
        PIFU_ZIP_PATH,
        force=force,
    )

    files = locate_required_files()

    # Raw Source Loading
    raw_training = read_jsonl(files["training"])
    raw_external_test = read_json(files["external_test"])
    raw_edge_cases = read_jsonl(files["edge_cases"])
    raw_hard_negatives = read_jsonl(files["hard_negatives"])

    # Source Conversion
    training_all = [
        convert_training_row(row)
        for row in raw_training
    ]

    external_test = [
        convert_external_test_row(row)
        for row in raw_external_test
    ]

    challenge_samples = [
        *[
            convert_challenge_row(
                row,
                "pifu_edge_cases",
            )
            for row in raw_edge_cases
        ],
        *[
            convert_challenge_row(
                row,
                "pifu_hard_negatives",
            )
            for row in raw_hard_negatives
        ],
    ]

    # External-Test Leakage Removal
    training_pool, removed_overlap = remove_external_test_overlap(
        training_all,
        external_test,
    )

    # Train-Validation Split
    train_samples, validation_samples = split_training_pool(
        training_pool
    )

    # Leakage Checks
    assert_no_text_overlap(
        "train",
        train_samples,
        "external test",
        external_test,
    )

    assert_no_text_overlap(
        "validation",
        validation_samples,
        "external test",
        external_test,
    )

    assert_no_text_overlap(
        "train",
        train_samples,
        "validation",
        validation_samples,
    )

    # Dataset Writing
    write_dataset(
        PIFU_TRAIN_PATH,
        "train",
        train_samples,
    )

    write_dataset(
        PIFU_VALIDATION_PATH,
        "validation",
        validation_samples,
    )

    write_dataset(
        PIFU_TEST_PATH,
        "external_test",
        external_test,
    )

    write_dataset(
        PIFU_CHALLENGE_PATH,
        "challenge",
        challenge_samples,
    )

    # Summary Writing
    summary = {
        "source_zip": str(PIFU_ZIP_PATH),
        "raw_training_rows": len(training_all),
        "external_test_rows": len(external_test),
        "external_test_rows_removed_from_training": removed_overlap,
        "training_pool_after_leakage_removal": len(training_pool),
        "train_rows": len(train_samples),
        "validation_rows": len(validation_samples),
        "challenge_rows": len(challenge_samples),
        "train_class_counts": class_counts(train_samples),
        "validation_class_counts": class_counts(validation_samples),
        "external_test_class_counts": class_counts(external_test),
        "challenge_class_counts": class_counts(challenge_samples),
    }

    write_summary(summary)

    # Console Summary
    print("=" * 68)
    print("FASTPIFU DATA PREPARATION COMPLETE")
    print("=" * 68)
    print(
        "Evaluation letters removed from training: "
        f"{removed_overlap}"
    )
    print(f"Train        : {len(train_samples):,}")
    print(f"Validation   : {len(validation_samples):,}")
    print(f"External test: {len(external_test):,}")
    print(f"Challenge    : {len(challenge_samples):,}")
    print(f"Summary      : {PIFU_SUMMARY_PATH}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    # CLI Argument Setup
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        type=Path,
        default=PIFU_ZIP_PATH,
        help="Path to fastpifu-cardiology.zip",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract the ZIP before preparing data.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the FastPIFU data preparation workflow."""

    # CLI Setup
    args = parse_args()

    # Dataset Preparation
    prepare(
        source_zip=args.zip,
        force=args.force,
    )


if __name__ == "__main__":
    main()