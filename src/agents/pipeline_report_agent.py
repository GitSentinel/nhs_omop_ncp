"""
Pipeline Assessment Agent

Creates the final structured assessment for an agentic model pipeline run. Uses a Pydantic AI reporting agent when enabled, and falls back to a deterministic assessment if LLM reporting is disabled or unavailable.
"""

from __future__ import annotations

import json
from typing import Any

# Pydantic AI Components
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Pipeline Models
from src.agents.pipeline_models import AgentAssessment

# Runtime Settings
from src.config.settings import settings


def _fallback_assessment(
    evidence: dict[str, Any],
    reason: str,
) -> AgentAssessment:
    """Create a deterministic assessment without calling an LLM."""

    # Evidence Extraction
    steps = evidence.get(
        "steps",
        [],
    )

    metrics = evidence.get(
        "metrics",
        [],
    )

    # Step Status Summary
    failures = [
        step
        for step in steps
        if step.get("status") == "failed"
    ]

    succeeded = [
        step
        for step in steps
        if step.get("status") == "succeeded"
    ]

    if failures:
        overall_status = (
            "partial_success"
            if succeeded
            else "failed"
        )
    else:
        overall_status = "success"

    # Safety Flags
    safety_flags = [
        "All data and model outputs are synthetic/research outputs.",
        (
            "PIFU eligibility predictions require clinician review "
            "and must not be used autonomously."
        ),
    ]

    if failures:
        safety_flags.append(
            "One or more pipeline stages failed; downstream results may be incomplete."
        )

    # Recommended Action
    recommended_action = (
        "Review failed-stage logs before interpreting or comparing model performance."
        if failures
        else (
            "Review the saved metrics, confusion matrices, and false predictions "
            "before drawing conclusions."
        )
    )

    return AgentAssessment(
        overall_status=overall_status,
        executive_summary=(
            "The deterministic pipeline assessment completed with "
            f"{len(succeeded)} successful step(s) and "
            f"{len(failures)} failed step(s). "
            f"LLM assessment fallback used: {reason}"
        ),
        key_findings=[
            (
                f"Collected {len(metrics)} structured metric bundle(s) "
                "from the workflow evidence."
            ),
        ],
        safety_flags=safety_flags,
        recommended_actions=[
            recommended_action,
        ],
        comparison_statement=None,
    )


def make_report_agent() -> Agent[None, AgentAssessment]:
    """Create the Pydantic AI reporting agent."""

    # Azure OpenAI Model Setup
    model = OpenAIChatModel(
        settings.azure_openai_deployment,
        provider=OpenAIProvider(
            base_url=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
        ),
    )

    # Structured Report Agent
    return Agent(
        model,
        output_type=AgentAssessment,
        instructions=(
            "You are an ML evaluation auditor for a synthetic NHS research project. "
            "Use only the JSON evidence supplied by the workflow. "
            "Never invent metrics, files, successful stages, or clinical conclusions. "
            "Preserve numeric values exactly. "
            "Distinguish the original binary treatment-event task from the "
            "three-class PIFU eligibility task. "
            "For PIFU, prioritise NOT_ELIGIBLE recall, ELIGIBLE precision, "
            "unsafe eligible count/rate, BORDERLINE recall, and manual-review rate. "
            "For the original task, prioritise treatment-event recall, precision, "
            "macro F1, balanced accuracy, PR AUC, and false negatives when available. "
            "State that outputs are for synthetic research and require human review."
        ),
    )


def azure_openai_settings_available() -> bool:
    """Return whether the Azure OpenAI reporting settings are complete."""

    # Settings Check
    return bool(
        settings.azure_openai_api_key
        and settings.azure_openai_endpoint
        and settings.azure_openai_deployment
    )


def build_agent_prompt(
    evidence: dict[str, Any],
) -> str:
    """Build the reporting prompt from structured workflow evidence."""

    # Evidence Serialisation
    evidence_json = json.dumps(
        evidence,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return (
        "Create the final structured pipeline assessment from this evidence. "
        "Use only the evidence supplied below.\n\n"
        f"{evidence_json}"
    )


async def create_agent_assessment(
    evidence: dict[str, Any],
    use_llm: bool = True,
) -> AgentAssessment:
    """Create the final structured pipeline assessment."""

    # Deterministic Reporting Mode
    if not use_llm:
        return _fallback_assessment(
            evidence,
            reason="LLM reporting was disabled.",
        )

    # Azure OpenAI Availability Check
    if not azure_openai_settings_available():
        return _fallback_assessment(
            evidence,
            reason="Azure OpenAI settings were incomplete.",
        )

    # LLM-Based Assessment
    try:
        agent = make_report_agent()
        prompt = build_agent_prompt(evidence)

        result = await agent.run(prompt)

        if not isinstance(result.output, AgentAssessment):
            return _fallback_assessment(
                evidence,
                reason=(
                    "Pydantic AI returned an unexpected output type: "
                    f"{type(result.output).__name__}"
                ),
            )

        return result.output

    except Exception as error:
        return _fallback_assessment(
            evidence,
            reason=f"{type(error).__name__}: {error}",
        )