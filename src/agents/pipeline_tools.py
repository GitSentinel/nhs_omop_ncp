"""
Pipeline Execution Tools

Defines LangChain tools for validating the project, running approved pipeline scripts, selecting GPUs, collecting logs/artifacts, and extracting structured metrics for the final pipeline report.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# LangChain Tooling
from langchain.tools import tool

# Pipeline Models
from src.agents.pipeline_models import (
    ScriptStepResult,
    StepStatus,
)

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Approved Step Names
StepName = Literal[
    "original_generate",
    "original_relabel",
    "original_finetune",
    "original_evaluate",
    "pifu_prepare",
    "pifu_finetune",
    "pifu_evaluate",
]

# Approved Script Registry
APPROVED_SCRIPTS = {
    "original_generate": "src/scripts/generate_dataset.py",
    "original_relabel": "src/scripts/relabel_dataset.py",
    "original_finetune": "src/scripts/finetune.py",
    "original_evaluate": "src/scripts/evaluate_model.py",
    "pifu_prepare": "src/scripts/prepare_fastpifu_dataset.py",
    "pifu_finetune": "src/scripts/finetune_pifu.py",
    "pifu_evaluate": "src/scripts/evaluate_pifu.py",
}

# Step Dataset Mapping
STEP_DATASET = {
    "original_generate": "original",
    "original_relabel": "original",
    "original_finetune": "original",
    "original_evaluate": "original",
    "pifu_prepare": "pifu",
    "pifu_finetune": "pifu",
    "pifu_evaluate": "pifu",
}


def utc_now() -> datetime:
    """Return the current UTC timestamp."""

    # UTC Timestamp
    return datetime.now(UTC)


def gpu_ids_for_step(
    step: StepName,
    train_gpus: list[int],
    evaluation_gpu: int,
) -> list[int]:
    """Return the physical GPU IDs used by one pipeline step."""

    # Training GPU Assignment
    if step in {
        "original_finetune",
        "pifu_finetune",
    }:
        return train_gpus

    # Evaluation GPU Assignment
    if step in {
        "original_evaluate",
        "pifu_evaluate",
    }:
        return [evaluation_gpu]

    return []


def parse_gpu_csv(value: str) -> list[int]:
    """Parse a comma-separated list of physical GPU IDs."""

    # CSV Parsing
    values = [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]

    gpu_ids = []

    for item in values:
        if not item.isdigit():
            raise ValueError(
                f"GPU identifier must be an integer, received {item!r}."
            )

        gpu_ids.append(int(item))

    # Duplicate Check
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("Duplicate GPU identifiers were supplied.")

    return gpu_ids


def active_compute_gpu_uuids() -> set[str]:
    """Return GPU UUIDs currently used by active compute processes."""

    # Active Process Query
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    except FileNotFoundError:
        return set()

    if completed.returncode != 0:
        return set()

    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    }


def select_idle_gpus(
    count: int,
    max_memory_mib: int = 1500,
    max_utilisation_percent: int = 10,
) -> list[int]:
    """Select physical GPUs with no active compute process and low usage."""

    # Request Validation
    if count < 1:
        raise ValueError("At least one GPU must be requested.")

    busy_uuids = active_compute_gpu_uuids()

    # GPU Status Query
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except FileNotFoundError as error:
        raise RuntimeError(
            "nvidia-smi was not found. GPU selection requires NVIDIA GPUs."
        ) from error

    selected: list[int] = []

    # Idle GPU Selection
    for line in completed.stdout.splitlines():
        values = [
            value.strip()
            for value in line.split(",")
        ]

        if len(values) != 4:
            continue

        index_text, uuid, memory_text, utilisation_text = values

        index = int(index_text)
        memory = int(memory_text)
        utilisation = int(utilisation_text)

        if uuid in busy_uuids:
            continue

        if memory > max_memory_mib:
            continue

        if utilisation > max_utilisation_percent:
            continue

        selected.append(index)

        if len(selected) == count:
            return selected

    raise RuntimeError(
        f"Requested {count} idle GPU(s), but only found {len(selected)}. "
        "Run nvidia-smi and pass explicit GPU IDs if necessary."
    )


def resolve_train_gpus(value: str) -> list[int]:
    """Resolve training GPU IDs from explicit input or auto-selection."""

    # Automatic Training GPU Selection
    if value.strip().lower() == "auto":
        return select_idle_gpus(count=2)

    # Explicit Training GPU Selection
    gpu_ids = parse_gpu_csv(value)

    if not gpu_ids:
        raise ValueError("No training GPUs were selected.")

    return gpu_ids


def resolve_evaluation_gpu(value: str) -> int:
    """Resolve the single evaluation GPU ID."""

    # Automatic Evaluation GPU Selection
    if value.strip().lower() == "auto":
        return select_idle_gpus(count=1)[0]

    # Explicit Evaluation GPU Selection
    gpu_ids = parse_gpu_csv(value)

    if len(gpu_ids) != 1:
        raise ValueError("Evaluation requires exactly one GPU.")

    return gpu_ids[0]


def required_steps_for_target(
    target: Literal["original", "pifu", "both"],
    mode: Literal[
        "full",
        "train-and-evaluate",
        "post-finetune",
    ],
) -> list[StepName]:
    """Return the ordered pipeline steps for the selected target and mode."""

    # Required Step Construction
    steps: list[StepName] = []

    if target in {"original", "both"}:
        if mode == "full":
            steps.extend([
                "original_generate",
                "original_relabel",
            ])

        if mode in {
            "full",
            "train-and-evaluate",
        }:
            steps.append("original_finetune")

        steps.append("original_evaluate")

    if target in {"pifu", "both"}:
        if mode == "full":
            steps.append("pifu_prepare")

        if mode in {
            "full",
            "train-and-evaluate",
        }:
            steps.append("pifu_finetune")

        steps.append("pifu_evaluate")

    return steps


def build_command(
    step: StepName,
    train_gpus: list[int],
    evaluation_gpu: int,
    force_prepare: bool,
) -> tuple[list[str], dict[str, str]]:
    """Build a safe command for one approved project step."""

    # Script Validation
    script = APPROVED_SCRIPTS[step]
    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        raise FileNotFoundError(
            f"Required script does not exist: {script_path}"
        )

    # Environment Setup
    environment = os.environ.copy()

    environment.setdefault("TOKENIZERS_PARALLELISM", "false")

    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    # Distributed Fine-Tuning Commands
    if step in {
        "original_finetune",
        "pifu_finetune",
    }:
        if not train_gpus:
            raise ValueError(f"{step} requires at least one training GPU.")

        environment["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_id)
            for gpu_id in train_gpus
        )

        master_port = (
            "29500"
            if step == "original_finetune"
            else "29510"
        )

        command = [
            "uv",
            "run",
            "--extra",
            "finetune",
            "python",
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(train_gpus)}",
            f"--master_port={master_port}",
            script,
        ]

        return command, environment

    # Evaluation Commands
    if step in {
        "original_evaluate",
        "pifu_evaluate",
    }:
        environment["CUDA_VISIBLE_DEVICES"] = str(evaluation_gpu)

        command = [
            "uv",
            "run",
            "--extra",
            "finetune",
            "python",
            script,
        ]

        return command, environment

    # FastPIFU Preparation Command
    if step == "pifu_prepare":
        command = [
            "uv",
            "run",
            "--extra",
            "finetune",
            "python",
            script,
        ]

        if force_prepare:
            command.append("--force")

        return command, environment

    # Original Dataset Commands
    command = [
        "uv",
        "run",
        "python",
        script,
    ]

    return command, environment


def find_expected_artifacts(step: StepName) -> list[Path]:
    """Find expected artifacts produced by a completed step."""

    # Candidate Artifact Patterns
    candidate_patterns = {
        "original_generate": [
            "data/finetune/clinic_letters_labelled.json",
        ],
        "original_relabel": [
            "data/finetune/clinic_letters_azure_labelled.json",
        ],
        "original_finetune": [
            "data/finetune/**/adapter_config.json",
            "data/finetune/**/adapter_model.safetensors",
        ],
        "original_evaluate": [
            "mlflow_runs/mlflow.db",
        ],
        "pifu_prepare": [
            "data/external/fastpifu_cardiology/processed/pifu_train.json",
            "data/external/fastpifu_cardiology/processed/pifu_validation.json",
            "data/external/fastpifu_cardiology/processed/pifu_external_test_150.json",
        ],
        "pifu_finetune": [
            "data/finetune/qwen35_9b_pifu_lora_3epochs_same_protocol/adapter_config.json",
            "data/finetune/qwen35_9b_pifu_lora_3epochs_same_protocol/adapter_model.safetensors",
        ],
        "pifu_evaluate": [
            "data/evaluations/pifu_cardiology_same_protocol/*_metrics.json",
            "data/evaluations/pifu_cardiology_same_protocol/*_predictions.csv",
        ],
    }

    artifacts: list[Path] = []

    # Artifact Discovery
    for pattern in candidate_patterns[step]:
        artifacts.extend(
            path.resolve()
            for path in PROJECT_ROOT.glob(pattern)
            if path.is_file()
        )

    return sorted(set(artifacts))


def run_approved_step(
    step: StepName,
    run_dir: Path,
    train_gpus: list[int],
    evaluation_gpu: int,
    force_prepare: bool,
) -> ScriptStepResult:
    """Run one approved pipeline step and return structured metadata."""

    # Run Directory Setup
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started_at = utc_now()
    started_clock = time.monotonic()

    log_path = run_dir / f"{step}.log"
    stdout_tail: deque[str] = deque(maxlen=120)

    try:
        # Command Construction
        command, environment = build_command(
            step=step,
            train_gpus=train_gpus,
            evaluation_gpu=evaluation_gpu,
            force_prepare=force_prepare,
        )

        # Process Execution and Log Capture
        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                "COMMAND: "
                + " ".join(command)
                + "\n\n"
            )

            log_file.flush()

            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None

            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                stdout_tail.append(line.rstrip())

            return_code = process.wait()

        finished_at = utc_now()
        duration = time.monotonic() - started_clock

        status = (
            StepStatus.SUCCEEDED
            if return_code == 0
            else StepStatus.FAILED
        )

        return ScriptStepResult(
            step=step,
            dataset=STEP_DATASET[step],
            status=status,
            command=command,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            return_code=return_code,
            log_path=log_path.resolve(),
            stdout_tail=list(stdout_tail),
            artifacts=find_expected_artifacts(step),
            gpu_ids=gpu_ids_for_step(
                step=step,
                train_gpus=train_gpus,
                evaluation_gpu=evaluation_gpu,
            ),
            error=(
                None
                if return_code == 0
                else f"Command exited with status {return_code}."
            ),
        )

    except Exception as error:
        # Structured Failure Result
        finished_at = utc_now()

        return ScriptStepResult(
            step=step,
            dataset=STEP_DATASET[step],
            status=StepStatus.FAILED,
            command=[],
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.monotonic() - started_clock,
            return_code=None,
            log_path=(
                log_path.resolve()
                if log_path.exists()
                else None
            ),
            stdout_tail=list(stdout_tail),
            artifacts=[],
            gpu_ids=gpu_ids_for_step(
                step=step,
                train_gpus=train_gpus,
                evaluation_gpu=evaluation_gpu,
            ),
            error=f"{type(error).__name__}: {error}",
        )


@tool("validate_pipeline_project")
def validate_pipeline_project(
    target: Literal["original", "pifu", "both"],
    mode: Literal[
        "full",
        "train-and-evaluate",
        "post-finetune",
    ],
) -> dict:
    """Validate required scripts and directories before running the pipeline."""

    # Required Step Resolution
    steps = required_steps_for_target(
        target=target,
        mode=mode,
    )

    # Missing Script Detection
    missing = [
        str((PROJECT_ROOT / APPROVED_SCRIPTS[step]).resolve())
        for step in steps
        if not (PROJECT_ROOT / APPROVED_SCRIPTS[step]).exists()
    ]

    return {
        "ok": not missing,
        "project_root": str(PROJECT_ROOT.resolve()),
        "required_steps": steps,
        "missing_scripts": missing,
    }


@tool("run_project_pipeline_step")
def run_project_pipeline_step(
    step: StepName,
    run_dir: str,
    train_gpu_ids: str = "auto",
    evaluation_gpu_id: str = "auto",
    force_prepare: bool = False,
) -> dict:
    """
    Run one approved project pipeline step.

    Arbitrary shell commands are not accepted. The step must be one of the
    fixed project scripts defined by the pipeline.
    """

    # GPU Resolution
    needs_training_gpu = step in {
        "original_finetune",
        "pifu_finetune",
    }

    needs_evaluation_gpu = step in {
        "original_evaluate",
        "pifu_evaluate",
    }

    train_gpus = (
        resolve_train_gpus(train_gpu_ids)
        if needs_training_gpu
        else []
    )

    evaluation_gpu = (
        resolve_evaluation_gpu(evaluation_gpu_id)
        if needs_evaluation_gpu
        else 0
    )

    # Approved Step Execution
    result = run_approved_step(
        step=step,
        run_dir=Path(run_dir),
        train_gpus=train_gpus,
        evaluation_gpu=evaluation_gpu,
        force_prepare=force_prepare,
    )

    return result.model_dump(mode="json")


def parse_original_metrics_from_log(
    log_path: Path,
) -> list[dict]:
    """Parse original binary-task metrics from the evaluation log."""

    # Log Existence Check
    if not log_path.exists():
        return []

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    section_pattern = re.compile(
        r"(?P<section>BASE MODEL|FINE-TUNED MODEL)\s+Results:\s*"
        r"(?P<body>.*?)(?=\n[A-Z][A-Z -]+ Results:|\n=+\nCOMPARISON|\Z)",
        re.DOTALL,
    )

    key_map = {
        "F1": "f1",
        "Macro F1": "macro_f1",
        "Accuracy": "accuracy",
        "Balanced accuracy": "balanced_accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "ROC AUC": "roc_auc",
        "PR AUC": "pr_auc",
    }

    bundles = []

    # Section Metric Extraction
    for match in section_pattern.finditer(text):
        metrics: dict[str, float] = {}
        body = match.group("body")

        for label, key in key_map.items():
            metric_match = re.search(
                rf"^\s*{re.escape(label)}\s*:\s*([0-9.]+)",
                body,
                re.MULTILINE | re.IGNORECASE,
            )

            if metric_match:
                metrics[key] = float(metric_match.group(1))

        if metrics:
            bundles.append({
                "dataset": "original",
                "model": (
                    "base"
                    if match.group("section") == "BASE MODEL"
                    else "fine_tuned"
                ),
                "split": "test",
                "metrics": metrics,
                "source": str(log_path.resolve()),
            })

    return bundles


def collect_json_metric_files() -> list[dict]:
    """Collect JSON metric files produced by evaluation scripts."""

    # Metric File Patterns
    patterns = [
        "data/evaluations/**/*_metrics.json",
        "data/finetune/**/evaluation/*metrics.json",
    ]

    bundles: list[dict] = []

    # Metric Bundle Loading
    for pattern in patterns:
        for path in PROJECT_ROOT.glob(pattern):
            if not path.is_file():
                continue

            try:
                payload = json.loads(
                    path.read_text(encoding="utf-8")
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue

            path_text = str(path).lower()
            name_text = path.name.lower()

            dataset = (
                "pifu"
                if "pifu" in path_text
                else "original"
            )

            model = (
                "base"
                if "base" in name_text
                else "fine_tuned"
            )

            if "challenge" in name_text:
                split = "challenge"

            elif "external" in name_text:
                split = "external_test"

            else:
                split = "test"

            bundles.append({
                "dataset": dataset,
                "model": model,
                "split": split,
                "metrics": payload,
                "source": str(path.resolve()),
            })

    return bundles


@tool("collect_pipeline_metrics")
def collect_pipeline_metrics(
    run_dir: str,
) -> list[dict]:
    """Collect structured metrics from evaluation files and script logs."""

    # Metric Bundle Collection
    run_path = Path(run_dir)

    bundles = collect_json_metric_files()

    original_log = run_path / "original_evaluate.log"

    bundles.extend(
        parse_original_metrics_from_log(original_log)
    )

    # Deduplication
    unique: dict[
        tuple[str, str, str, str],
        dict,
    ] = {}

    for bundle in bundles:
        key = (
            bundle["dataset"],
            bundle["model"],
            bundle["split"],
            bundle["source"],
        )

        unique[key] = bundle

    return list(unique.values())