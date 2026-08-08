"""
Agentic pipeline workflow graph.

Builds the LangGraph workflow used to validate, run, evaluate, collect metrics, assess results, save reports, and log pipeline artifacts.
"""

from __future__ import annotations

import json
import operator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

# Experiment Tracking
import mlflow

# LangGraph Components
from langgraph.graph import END, START, StateGraph

# Typing Utilities
from typing_extensions import TypedDict

# Pipeline Models
from src.agents.pipeline_models import (
    AgentAssessment,
    MetricBundle,
    PipelineMode,
    PipelineReport,
    PipelineTarget,
    ScriptStepResult,
    StepStatus,
)

# Reporting Agent
from src.agents.pipeline_report_agent import create_agent_assessment

# Pipeline Tools
from src.agents.pipeline_tools import (
    collect_pipeline_metrics,
    run_project_pipeline_step,
    validate_pipeline_project,
)

# Runtime Settings
from src.config.settings import settings

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MLflow Experiment
PIPELINE_MLFLOW_EXPERIMENT = "nhs_omop_pipeline_agent"


class PipelineState(TypedDict, total=False):
    """State passed through the agentic pipeline graph."""

    config: dict[str, Any]
    started_at: str
    steps: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]
    metrics: list[dict[str, Any]]
    assessment: dict[str, Any]
    final_report: dict[str, Any]


def now_iso() -> str:
    """Return the current UTC timestamp as an ISO string."""

    # UTC Timestamp
    return datetime.now(UTC).isoformat()


def has_failed_step(
    state: PipelineState,
) -> bool:
    """Return whether any previous pipeline step has failed."""

    # Failure Detection
    return any(
        step.get("status") == StepStatus.FAILED.value
        for step in state.get("steps", [])
    )


def should_run_step(
    config: dict[str, Any],
    step: str,
) -> bool:
    """Return whether a step is required for the selected target and mode."""

    # Target and Mode Resolution
    target = PipelineTarget(config["target"])
    mode = PipelineMode(config["mode"])

    is_original = step.startswith("original_")
    is_pifu = step.startswith("pifu_")

    # Target Filtering
    if is_original and target not in {
        PipelineTarget.ORIGINAL,
        PipelineTarget.BOTH,
    }:
        return False

    if is_pifu and target not in {
        PipelineTarget.PIFU,
        PipelineTarget.BOTH,
    }:
        return False

    # Mode Filtering
    if mode == PipelineMode.POST_FINETUNE:
        return step.endswith("_evaluate")

    if mode == PipelineMode.TRAIN_AND_EVALUATE:
        return step.endswith("_finetune") or step.endswith("_evaluate")

    return True


def skipped_result(
    step: str,
    dataset: str,
    reason: str,
) -> dict[str, Any]:
    """Create a structured skipped-step result."""

    # Skipped Step Result
    now = datetime.now(UTC)

    return ScriptStepResult(
        step=step,
        dataset=dataset,
        status=StepStatus.SKIPPED,
        command=[],
        started_at=now,
        finished_at=now,
        duration_seconds=0,
        return_code=None,
        log_path=None,
        stdout_tail=[],
        artifacts=[],
        gpu_ids=[],
        error=reason,
    ).model_dump(mode="json")


def validate_node(
    state: PipelineState,
) -> dict[str, Any]:
    """Validate required scripts before running the pipeline."""

    # Configuration Extraction
    config = state["config"]

    # Project Validation
    result = validate_pipeline_project.invoke({
        "target": config["target"],
        "mode": config["mode"],
    })

    now = datetime.now(UTC)

    # Validation Step Result
    step = ScriptStepResult(
        step="validate",
        dataset="shared",
        status=(
            StepStatus.SUCCEEDED
            if result["ok"]
            else StepStatus.FAILED
        ),
        command=[],
        started_at=now,
        finished_at=now,
        duration_seconds=0,
        return_code=(
            0
            if result["ok"]
            else 1
        ),
        log_path=None,
        stdout_tail=[
            json.dumps(
                result,
                ensure_ascii=False,
            )
        ],
        artifacts=[],
        gpu_ids=[],
        error=(
            None
            if result["ok"]
            else (
                "Missing required scripts: "
                + ", ".join(result["missing_scripts"])
            )
        ),
    )

    return {
        "steps": [
            step.model_dump(mode="json"),
        ],
    }


def make_script_node(
    step: str,
    dataset: str,
):
    """Create a graph node that runs one approved pipeline script."""

    def node(
        state: PipelineState,
    ) -> dict[str, Any]:
        """Run or skip one pipeline script step."""

        # Configuration Extraction
        config = state["config"]

        # Target and Mode Skip
        if not should_run_step(
            config,
            step,
        ):
            return {
                "steps": [
                    skipped_result(
                        step,
                        dataset,
                        (
                            "Not required for "
                            f"target={config['target']} "
                            f"mode={config['mode']}."
                        ),
                    )
                ],
            }

        # Failure Skip
        if config.get("stop_on_error", True) and has_failed_step(state):
            return {
                "steps": [
                    skipped_result(
                        step,
                        dataset,
                        "Skipped because an earlier stage failed.",
                    )
                ],
            }

        # Approved Step Execution
        result = run_project_pipeline_step.invoke({
            "step": step,
            "run_dir": config["run_dir"],
            "train_gpu_ids": config["train_gpus"],
            "evaluation_gpu_id": config["evaluation_gpu"],
            "force_prepare": config["force_prepare"],
        })

        return {
            "steps": [
                result,
            ],
        }

    return node


def collect_node(
    state: PipelineState,
) -> dict[str, Any]:
    """Collect structured metric bundles after script execution."""

    # Metric Collection
    metrics = collect_pipeline_metrics.invoke({
        "run_dir": state["config"]["run_dir"],
    })

    return {
        "metrics": metrics,
    }


async def assessment_node(
    state: PipelineState,
) -> dict[str, Any]:
    """Create the final structured agent assessment."""

    # Evidence Construction
    evidence = {
        "target": state["config"]["target"],
        "mode": state["config"]["mode"],
        "steps": state.get("steps", []),
        "metrics": state.get("metrics", []),
    }

    # Agent Assessment
    assessment = await create_agent_assessment(
        evidence=evidence,
        use_llm=state["config"]["use_llm_report"],
    )

    return {
        "assessment": assessment.model_dump(mode="json"),
    }


def markdown_report(
    report: PipelineReport,
) -> str:
    """Create the Markdown version of the final pipeline report."""

    # Report Header
    lines = [
        f"# Agentic Pipeline Report - {report.run_id}",
        "",
        f"- Target: `{report.target.value}`",
        f"- Mode: `{report.mode.value}`",
        f"- Started: `{report.started_at.isoformat()}`",
        f"- Finished: `{report.finished_at.isoformat()}`",
        f"- Training GPUs: `{report.train_gpus}`",
        f"- Evaluation GPU: `{report.evaluation_gpu}`",
        f"- Overall status: `{report.assessment.overall_status}`",
        "",
        "## Executive summary",
        "",
        report.assessment.executive_summary,
        "",
        "## Pipeline steps",
        "",
        "| Step | Dataset | Status | Duration (s) | Return code |",
        "|---|---|---:|---:|---:|",
    ]

    # Step Table
    for step in report.steps:
        lines.append(
            "| "
            f"{step.step} | "
            f"{step.dataset} | "
            f"{step.status.value} | "
            f"{step.duration_seconds:.1f} | "
            f"{step.return_code} |"
        )

    # Key Findings
    lines.extend([
        "",
        "## Key findings",
        "",
    ])

    for finding in report.assessment.key_findings:
        lines.append(f"- {finding}")

    # Safety Flags
    lines.extend([
        "",
        "## Safety flags",
        "",
    ])

    for flag in report.assessment.safety_flags:
        lines.append(f"- {flag}")

    # Recommended Actions
    lines.extend([
        "",
        "## Recommended actions",
        "",
    ])

    for action in report.assessment.recommended_actions:
        lines.append(f"- {action}")

    # Optional Comparison
    if report.assessment.comparison_statement:
        lines.extend([
            "",
            "## Comparison",
            "",
            report.assessment.comparison_statement,
        ])

    # Structured Metrics
    lines.extend([
        "",
        "## Structured metric bundles",
        "",
        "```json",
        json.dumps(
            [
                metric.model_dump(mode="json")
                for metric in report.metrics
            ],
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        (
            "> Synthetic research data only. Human clinical and "
            "methodological review is required."
        ),
    ])

    return "\n".join(lines)


def infer_used_gpus(
    steps: list[ScriptStepResult],
) -> tuple[list[int], int | None]:
    """Infer the training and evaluation GPUs used by completed steps."""

    # Training GPU Candidates
    train_gpu_candidates = [
        step.gpu_ids
        for step in steps
        if step.step.endswith("_finetune") and step.gpu_ids
    ]

    # Evaluation GPU Candidates
    evaluation_gpu_candidates = [
        step.gpu_ids[0]
        for step in steps
        if step.step.endswith("_evaluate") and step.gpu_ids
    ]

    train_gpus = (
        train_gpu_candidates[-1]
        if train_gpu_candidates
        else []
    )

    evaluation_gpu = (
        evaluation_gpu_candidates[-1]
        if evaluation_gpu_candidates
        else None
    )

    return train_gpus, evaluation_gpu


def save_node(
    state: PipelineState,
) -> dict[str, Any]:
    """Validate, save, and log the final pipeline report."""

    # Run Directory Setup
    config = state["config"]
    run_dir = Path(config["run_dir"])

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_json = run_dir / "pipeline_report.json"
    report_markdown = run_dir / "pipeline_report.md"

    # Pydantic Validation
    validated_steps = [
        ScriptStepResult.model_validate(step)
        for step in state.get("steps", [])
    ]

    validated_metrics = [
        MetricBundle.model_validate(metric)
        for metric in state.get("metrics", [])
    ]

    assessment = AgentAssessment.model_validate(
        state["assessment"]
    )

    train_gpus, evaluation_gpu = infer_used_gpus(
        validated_steps
    )

    # Report Construction
    report = PipelineReport(
        run_id=config["run_id"],
        target=PipelineTarget(config["target"]),
        mode=PipelineMode(config["mode"]),
        started_at=datetime.fromisoformat(state["started_at"]),
        finished_at=datetime.now(UTC),
        train_gpus=train_gpus,
        evaluation_gpu=evaluation_gpu,
        steps=validated_steps,
        metrics=validated_metrics,
        assessment=assessment,
        report_json=report_json.resolve(),
        report_markdown=report_markdown.resolve(),
    )

    # Report Saving
    report_json.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    report_markdown.write_text(
        markdown_report(report),
        encoding="utf-8",
    )

    # MLflow Logging
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(PIPELINE_MLFLOW_EXPERIMENT)

    with mlflow.start_run(
        run_name=f"pipeline_{report.run_id}",
        tags={
            "target": report.target.value,
            "mode": report.mode.value,
            "synthetic_data_only": "true",
            "human_review_required": "true",
        },
    ):
        mlflow.log_params({
            "run_id": report.run_id,
            "target": report.target.value,
            "mode": report.mode.value,
            "train_gpus": str(report.train_gpus),
            "evaluation_gpu": report.evaluation_gpu,
            "step_count": len(report.steps),
            "metric_bundle_count": len(report.metrics),
            "overall_status": report.assessment.overall_status,
        })

        mlflow.log_artifact(
            str(report_json),
            artifact_path="report",
        )

        mlflow.log_artifact(
            str(report_markdown),
            artifact_path="report",
        )

        # Step Log Artifacts
        for step in report.steps:
            if step.log_path and step.log_path.exists():
                mlflow.log_artifact(
                    str(step.log_path),
                    artifact_path="logs",
                )

    return {
        "final_report": report.model_dump(mode="json"),
    }


def ordered_pipeline_steps() -> list[tuple[str, str]]:
    """Return the fixed ordered list of possible pipeline steps."""

    # Ordered Step List
    return [
        (
            "original_generate",
            "original",
        ),
        (
            "original_relabel",
            "original",
        ),
        (
            "original_finetune",
            "original",
        ),
        (
            "original_evaluate",
            "original",
        ),
        (
            "pifu_prepare",
            "pifu",
        ),
        (
            "pifu_finetune",
            "pifu",
        ),
        (
            "pifu_evaluate",
            "pifu",
        ),
    ]


def build_pipeline_graph():
    """Build and compile the agentic pipeline LangGraph."""

    # Graph Initialisation
    graph = StateGraph(PipelineState)

    # Validation Node
    graph.add_node(
        "validate",
        validate_node,
    )

    # Script Nodes
    previous = "validate"

    for step, dataset in ordered_pipeline_steps():
        graph.add_node(
            step,
            make_script_node(
                step,
                dataset,
            ),
        )

    # Final Processing Nodes
    graph.add_node(
        "collect_metrics",
        collect_node,
    )

    graph.add_node(
        "agent_assessment",
        assessment_node,
    )

    graph.add_node(
        "save_report",
        save_node,
    )

    # Graph Edges
    graph.add_edge(
        START,
        "validate",
    )

    for step, _ in ordered_pipeline_steps():
        graph.add_edge(
            previous,
            step,
        )
        previous = step

    graph.add_edge(
        previous,
        "collect_metrics",
    )

    graph.add_edge(
        "collect_metrics",
        "agent_assessment",
    )

    graph.add_edge(
        "agent_assessment",
        "save_report",
    )

    graph.add_edge(
        "save_report",
        END,
    )

    return graph.compile()