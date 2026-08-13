"""
FastPIFU Explanation Agent

Creates a structured narrative explanation for a deterministic FastPIFU
classifier result. The classifier prediction is authoritative; the agent
only explains the supplied prediction and safety assessment.
"""

from __future__ import annotations

import json

# Pydantic AI Components
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# FastPIFU Decision Models
from src.agents.pifu_decision_models import (
    PIFUExplanation,
    PIFUModelPrediction,
    PIFUSafetyAssessment,
)

# Runtime Settings
from src.config.settings import settings


def make_pifu_explanation_agent() -> Agent[None, PIFUExplanation]:
    """Create the structured PIFU explanation agent."""

    # Azure OpenAI Model Setup
    model = OpenAIChatModel(
        settings.azure_openai_deployment,
        provider=OpenAIProvider(
            base_url=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
        ),
    )

    # Structured Explanation Agent
    return Agent(
        model,
        output_type=PIFUExplanation,
        instructions=(
            "You are explaining the output of a research PIFU classifier "
            "operating on synthetic NHS cardiology letters. "
            "The classifier prediction supplied to you is authoritative. "
            "Do not change, replace, or independently choose the PIFU class. "
            "Use only information contained in the clinic letter and supplied "
            "classifier evidence. "
            "Briefly summarise factors relevant to PIFU suitability, such as "
            "stability, active management, timed follow-up, discharge pathway, "
            "uncertainty, or missing information. "
            "Do not invent clinical facts or provide treatment recommendations. "
            "Keep the explanation concise and state important limitations."
        ),
    )


def azure_settings_available() -> bool:
    """Check whether Azure OpenAI settings are available."""

    # Azure Settings Check
    return bool(
        settings.azure_openai_api_key
        and settings.azure_openai_endpoint
        and settings.azure_openai_deployment
    )


def fallback_explanation(
    reason: str,
) -> PIFUExplanation:
    """Return a deterministic fallback narrative."""

    # Fallback Explanation
    return PIFUExplanation(
        clinical_summary="Automated narrative explanation was unavailable.",
        evidence_summary=[],
        limitations=[
            reason,
            "Interpret the classifier output only with human clinical review.",
        ],
    )


def build_explanation_prompt(
    *,
    text: str,
    prediction: PIFUModelPrediction,
    safety: PIFUSafetyAssessment,
) -> str:
    """Build the explanation prompt from validated classifier evidence."""

    # Evidence Bundle
    evidence = {
        "clinic_letter": str(text),
        "classifier_prediction": prediction.model_dump(mode="json"),
        "review_assessment": safety.model_dump(mode="json"),
    }

    # Prompt Construction
    return (
        "Explain the supplied deterministic classifier result. "
        "Do not alter the prediction.\n\n"
        + json.dumps(
            evidence,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


async def create_pifu_explanation(
    *,
    text: str,
    prediction: PIFUModelPrediction,
    safety: PIFUSafetyAssessment,
    use_llm: bool = True,
) -> PIFUExplanation:
    """Generate a structured explanation for a PIFU classifier result."""

    # Deterministic Reporting Mode
    if not use_llm:
        return fallback_explanation(
            "LLM explanation was disabled."
        )

    # Azure OpenAI Availability Check
    if not azure_settings_available():
        return fallback_explanation(
            "Azure OpenAI settings were incomplete."
        )

    # Prompt Construction
    prompt = build_explanation_prompt(
        text=text,
        prediction=prediction,
        safety=safety,
    )

    # LLM-Based Explanation
    try:
        agent = make_pifu_explanation_agent()

        result = await agent.run(prompt)

        if not isinstance(result.output, PIFUExplanation):
            return fallback_explanation(
                "Unexpected Pydantic AI output type."
            )

        return result.output

    except Exception as error:
        return fallback_explanation(
            f"{type(error).__name__}: {error}"
        )