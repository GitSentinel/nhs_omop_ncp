"""
Agentic Pipeline Workflow Graphs

Builds the LangGraph workflow used to validate, run, evaluate,
collect metrics, assess results, save reports, and log pipeline artifacts.
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
        step.get("status") == StepStatus.FAILED.value for step in state.get("steps", [])
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
    result = validate_pipeline_project.invoke(
        {
            "target": config["target"],
            "mode": config["mode"],
        }
    )

    now = datetime.now(UTC)

    # Validation Step Result
    step = ScriptStepResult(
        step="validate",
        dataset="shared",
        status=(StepStatus.SUCCEEDED if result["ok"] else StepStatus.FAILED),
        command=[],
        started_at=now,
        finished_at=now,
        duration_seconds=0,
        return_code=(0 if result["ok"] else 1),
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
            else ("Missing required scripts: " + ", ".join(result["missing_scripts"]))
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
        result = run_project_pipeline_step.invoke(
            {
                "step": step,
                "run_dir": config["run_dir"],
                "train_gpu_ids": config["train_gpus"],
                "evaluation_gpu_id": config["evaluation_gpu"],
                "force_prepare": config["force_prepare"],
            }
        )

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
    metrics = collect_pipeline_metrics.invoke(
        {
            "run_dir": state["config"]["run_dir"],
        }
    )

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


def format_metric(
    value: object,
) -> str:
    """Format a metric for the Markdown report."""

    if isinstance(value, float):
        return f"{value:.4f}"

    if value is None:
        return "-"

    return str(value)


def find_metric_bundle(
    report: PipelineReport,
    *,
    dataset: str,
    model: str,
    split: str | None = None,
) -> MetricBundle | None:
    """Find one matching metric bundle."""

    # Candidate Selection
    candidates = [
        bundle
        for bundle in report.metrics
        if (
            bundle.dataset == dataset
            and bundle.model == model
            and (
                split is None
                or bundle.split == split
            )
        )
    ]

    if not candidates:
        return None

    return candidates[-1]


def metric_value(
    bundle: MetricBundle | None,
    key: str,
) -> str:
    """Return one formatted metric value."""

    if bundle is None:
        return "-"

    return format_metric(
        bundle.metrics.get(key)
    )


def comparison_row(
    label: str,
    base_bundle: MetricBundle | None,
    fine_tuned_bundle: MetricBundle | None,
    metric_key: str,
) -> str:
    """Create one Base vs Fine-tuned Markdown table row."""

    # Base Model Metric Value
    base_value = metric_value(
        base_bundle,
        metric_key,
    )

    # Fine-tuned Model Metric Value
    fine_tuned_value = metric_value(
        fine_tuned_bundle,
        metric_key,
    )

    return (
        f"| {label} | "
        f"{base_value} | "
        f"{fine_tuned_value} |"
    )


def markdown_report(
    report: PipelineReport,
) -> str:
    """Create a compact human-readable pipeline report."""

    status_icon = {
        "success": "✅",
        "partial_success": "⚠️",
        "failed": "❌",
    }.get(
        report.assessment.overall_status,
        "",
    )

    # Report Lines
    lines = [
        "# Agentic Model Pipeline Report",
        "",
        "## Run Summary",
        "",
        "| Item | Value |",
        "|---|---|",
        (
            f"| Status | {status_icon} "
            f"{report.assessment.overall_status.upper()} |"
        ),
        f"| Target | `{report.target.value}` |",
        f"| Mode | `{report.mode.value}` |",
        f"| Run ID | `{report.run_id}` |",
        f"| Evaluation GPU | `{report.evaluation_gpu}` |",
        "",
    ]

    # Original Base Model Clinic-Letter Evaluation
    original_base = find_metric_bundle(
        report,
        dataset="original",
        model="base",
        split="test",
    )

    # Original Fine-tuned Model Clinic-Letter Evaluation
    original_ft = find_metric_bundle(
        report,
        dataset="original",
        model="fine_tuned",
        split="test",
    )

    # Base vs Fine-tuned Comparison Table
    if original_base or original_ft:
        lines.extend(
            [
                "## Original Clinic-Letter Model",
                "",
                "| Metric | Base | Fine-tuned |",
                "|---|---:|---:|",
                comparison_row(
                    "F1",
                    original_base,
                    original_ft,
                    "f1",
                ),
                comparison_row(
                    "Macro F1",
                    original_base,
                    original_ft,
                    "macro_f1",
                ),
                comparison_row(
                    "Accuracy",
                    original_base,
                    original_ft,
                    "accuracy",
                ),
                comparison_row(
                    "Balanced accuracy",
                    original_base,
                    original_ft,
                    "balanced_accuracy",
                ),
                comparison_row(
                    "Precision",
                    original_base,
                    original_ft,
                    "precision",
                ),
                comparison_row(
                    "Recall",
                    original_base,
                    original_ft,
                    "recall",
                ),
                comparison_row(
                    "ROC AUC",
                    original_base,
                    original_ft,
                    "roc_auc",
                ),
                comparison_row(
                    "PR AUC",
                    original_base,
                    original_ft,
                    "pr_auc",
                ),
                "",
            ]
        )

    
    # PIFU External Base Model Evaluation
    pifu_external_base = find_metric_bundle(
        report,
        dataset="pifu",
        model="base",
        split="external_test",
    )

    # PIFU External Fine-tuned Model Evaluation
    pifu_external_ft = find_metric_bundle(
        report,
        dataset="pifu",
        model="fine_tuned",
        split="external_test",
    )

    # Base vs Fine-tuned Comparison Table
    if pifu_external_base or pifu_external_ft:
        lines.extend(
            [
                "## PIFU — External Test",
                "",
                "| Metric | Base | Fine-tuned |",
                "|---|---:|---:|",
                comparison_row(
                    "Macro F1",
                    pifu_external_base,
                    pifu_external_ft,
                    "macro_f1",
                ),
                comparison_row(
                    "Accuracy",
                    pifu_external_base,
                    pifu_external_ft,
                    "accuracy",
                ),
                comparison_row(
                    "Balanced accuracy",
                    pifu_external_base,
                    pifu_external_ft,
                    "balanced_accuracy",
                ),
                comparison_row(
                    "NOT_ELIGIBLE recall",
                    pifu_external_base,
                    pifu_external_ft,
                    "not_eligible_recall",
                ),
                comparison_row(
                    "BORDERLINE recall",
                    pifu_external_base,
                    pifu_external_ft,
                    "borderline_recall",
                ),
                comparison_row(
                    "ELIGIBLE precision",
                    pifu_external_base,
                    pifu_external_ft,
                    "eligible_precision",
                ),
                comparison_row(
                    "ELIGIBLE recall",
                    pifu_external_base,
                    pifu_external_ft,
                    "eligible_recall",
                ),
                comparison_row(
                    "Unsafe eligible errors",
                    pifu_external_base,
                    pifu_external_ft,
                    "unsafe_eligible_count",
                ),
                comparison_row(
                    "Manual review rate",
                    pifu_external_base,
                    pifu_external_ft,
                    "manual_review_rate",
                ),
                "",
            ]
        )

    # PIFU Challenge Set Base Model Evaluation
    pifu_challenge_base = find_metric_bundle(
        report,
        dataset="pifu",
        model="base",
        split="challenge",
    )

    # PIFU Challenge Set Fine-tuned Model Evaluation
    pifu_challenge_ft = find_metric_bundle(
        report,
        dataset="pifu",
        model="fine_tuned",
        split="challenge",
    )

    # Base vs Fine-tuned Comparison Table
    if pifu_challenge_base or pifu_challenge_ft:
        lines.extend(
            [
                "## PIFU — Challenge Set",
                "",
                "| Metric | Base | Fine-tuned |",
                "|---|---:|---:|",
                comparison_row(
                    "Macro F1",
                    pifu_challenge_base,
                    pifu_challenge_ft,
                    "macro_f1",
                ),
                comparison_row(
                    "Accuracy",
                    pifu_challenge_base,
                    pifu_challenge_ft,
                    "accuracy",
                ),
                comparison_row(
                    "Balanced accuracy",
                    pifu_challenge_base,
                    pifu_challenge_ft,
                    "balanced_accuracy",
                ),
                comparison_row(
                    "NOT_ELIGIBLE recall",
                    pifu_challenge_base,
                    pifu_challenge_ft,
                    "not_eligible_recall",
                ),
                comparison_row(
                    "BORDERLINE recall",
                    pifu_challenge_base,
                    pifu_challenge_ft,
                    "borderline_recall",
                ),
                comparison_row(
                    "ELIGIBLE precision",
                    pifu_challenge_base,
                    pifu_challenge_ft,
                    "eligible_precision",
                ),
                comparison_row(
                    "ELIGIBLE recall",
                    pifu_challenge_base,
                    pifu_challenge_ft,
                    "eligible_recall",
                ),
                comparison_row(
                    "Unsafe eligible errors",
                    pifu_challenge_base,
                    pifu_challenge_ft,
                    "unsafe_eligible_count",
                ),
                comparison_row(
                    "Manual review rate",
                    pifu_challenge_base,
                    pifu_challenge_ft,
                    "manual_review_rate",
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Assessment",
            "",
            report.assessment.executive_summary,
            "",
        ]
    )

    # Key Findings, Safety Flags, Recommended Actions
    if report.assessment.key_findings:
        lines.extend(
            [
                "### Key Findings",
                "",
            ]
        )

        for finding in report.assessment.key_findings[:3]:
            lines.append(
                f"- {finding}"
            )

        lines.append("")

    if report.assessment.safety_flags:
        lines.extend(
            [
                "### Safety",
                "",
            ]
        )

        for flag in report.assessment.safety_flags[:3]:
            lines.append(
                f"- {flag}"
            )

        lines.append("")

    if report.assessment.recommended_actions:
        lines.extend(
            [
                "### Recommended Actions",
                "",
            ]
        )

        for action in report.assessment.recommended_actions[:3]:
            lines.append(
                f"- {action}"
            )

        lines.append("")

    # Comparison Statement
    lines.extend(
        [
            "## Pipeline Execution",
            "",
            "| Step | Dataset | Status | Time (s) |",
            "|---|---|---|---:|",
        ]
    )

    # Step Execution Summary
    for step in report.steps:
        if step.status == StepStatus.SKIPPED:
            continue

        step_icon = {
            StepStatus.SUCCEEDED: "✅",
            StepStatus.FAILED: "❌",
        }.get(
            step.status,
            "⚠️",
        )

        lines.append(
            "| "
            f"{step.step} | "
            f"{step.dataset} | "
            f"{step_icon} {step.status.value} | "
            f"{step.duration_seconds:.1f} |"
        )
    
    lines.extend(
        [
            "",
            "---",
            "",
            (
                "*Synthetic research data only. "
                "Human clinical review is required.*"
            ),
            "",
            (
                "Full metrics, confusion matrices, classification "
                "reports and provenance are available in "
                "`pipeline_report.json` and the evaluation artifacts."
            ),
        ]
    )

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

    train_gpus = train_gpu_candidates[-1] if train_gpu_candidates else []

    evaluation_gpu = (
        evaluation_gpu_candidates[-1] if evaluation_gpu_candidates else None
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
        ScriptStepResult.model_validate(step) for step in state.get("steps", [])
    ]

    validated_metrics = [
        MetricBundle.model_validate(metric) for metric in state.get("metrics", [])
    ]

    assessment = AgentAssessment.model_validate(state["assessment"])

    train_gpus, evaluation_gpu = infer_used_gpus(validated_steps)

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
        mlflow.log_params(
            {
                "run_id": report.run_id,
                "target": report.target.value,
                "mode": report.mode.value,
                "train_gpus": str(report.train_gpus),
                "evaluation_gpu": report.evaluation_gpu,
                "step_count": len(report.steps),
                "metric_bundle_count": len(report.metrics),
                "overall_status": report.assessment.overall_status,
            }
        )

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
