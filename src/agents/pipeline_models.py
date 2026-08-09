"""
Pipeline Data Models

Defines the structured Pydantic schemas used by the agentic pipeline runner,
execution tools, reporting agent, and saved pipeline reports.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

# Pydantic Models
from pydantic import BaseModel, Field


class PipelineTarget(StrEnum):
    """Supported pipeline targets."""

    ORIGINAL = "original"
    PIFU = "pifu"
    BOTH = "both"


class PipelineMode(StrEnum):
    """Supported pipeline execution modes."""

    FULL = "full"
    TRAIN_AND_EVALUATE = "train-and-evaluate"
    POST_FINETUNE = "post-finetune"


class StepStatus(StrEnum):
    """Execution status for one pipeline step."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ScriptStepResult(BaseModel):
    """Structured result for one approved pipeline script."""

    step: str
    dataset: Literal["original", "pifu", "shared"]
    status: StepStatus

    command: list[str] = Field(default_factory=list)

    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)

    return_code: int | None = None
    log_path: Path | None = None
    stdout_tail: list[str] = Field(default_factory=list)

    artifacts: list[Path] = Field(default_factory=list)
    gpu_ids: list[int] = Field(default_factory=list)

    error: str | None = None


class MetricBundle(BaseModel):
    """Structured metrics collected from one model evaluation output."""

    dataset: Literal["original", "pifu"]
    model: str
    split: str

    metrics: dict[
        str,
        float | int | str | list[Any] | dict[str, Any],
    ]

    source: Path | str


class AgentAssessment(BaseModel):
    """Final structured assessment produced by the reporting agent."""

    overall_status: Literal[
        "success",
        "partial_success",
        "failed",
    ]

    executive_summary: str

    key_findings: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)

    comparison_statement: str | None = None


class PipelineReport(BaseModel):
    """Complete structured report for one pipeline run."""

    run_id: str
    target: PipelineTarget
    mode: PipelineMode

    started_at: datetime
    finished_at: datetime

    train_gpus: list[int] = Field(default_factory=list)
    evaluation_gpu: int | None = None

    steps: list[ScriptStepResult]
    metrics: list[MetricBundle]
    assessment: AgentAssessment

    report_json: Path
    report_markdown: Path

    synthetic_data_only: bool = True
    human_review_required: bool = True
