"""
FastPIFU Deterministic Safety Rules

Applies research-only human-review rules to a validated FastPIFU model
prediction. These rules do not alter the classifier prediction; they only
describe uncertainty and review requirements.
"""

from __future__ import annotations

# FastPIFU Decision Models
from src.agents.pifu_decision_models import (
    PIFULabel,
    PIFUModelPrediction,
    PIFUSafetyAssessment,
)

# Safety Thresholds
CONFIDENCE_THRESHOLD = 0.80
MARGIN_THRESHOLD = 0.15


def class_probability_values(
    prediction: PIFUModelPrediction,
) -> list[float]:
    """Return PIFU class probabilities in fixed label order."""

    # Probability Extraction
    return [
        prediction.probabilities.not_eligible,
        prediction.probabilities.borderline,
        prediction.probabilities.eligible,
    ]


def top_two_probability_margin(
    prediction: PIFUModelPrediction,
) -> float:
    """Calculate the margin between the two highest class probabilities."""

    # Probability Ranking
    ranked_probabilities = sorted(
        class_probability_values(prediction),
        reverse=True,
    )

    return float(ranked_probabilities[0] - ranked_probabilities[1])


def safety_flags_for_prediction(
    prediction: PIFUModelPrediction,
    margin: float,
) -> list[str]:
    """Return deterministic review flags for one PIFU prediction."""

    # Flag Construction
    flags: list[str] = []

    if prediction.predicted_class == PIFULabel.BORDERLINE:
        flags.append("Classifier returned BORDERLINE.")

    if prediction.confidence < CONFIDENCE_THRESHOLD:
        flags.append(f"Classifier confidence is below {CONFIDENCE_THRESHOLD:.2f}.")

    if margin < MARGIN_THRESHOLD:
        flags.append("The two highest class probabilities are close.")

    return flags


def build_review_reason(
    flags: list[str],
) -> str:
    """Build the human-review reason text."""

    # Base Review Reason
    reason = "Research-only PIFU classification requires human clinical review."

    # Additional Review Triggers
    if flags:
        reason += " Additional review triggers: " + " ".join(flags)

    return reason


def assess_pifu_safety(
    prediction: PIFUModelPrediction,
) -> PIFUSafetyAssessment:
    """Apply research-only uncertainty rules to a PIFU prediction."""

    # Margin Calculation
    margin = top_two_probability_margin(prediction)

    # Flag Assessment
    flags = safety_flags_for_prediction(
        prediction=prediction,
        margin=margin,
    )

    # Review Reason
    reason = build_review_reason(flags)

    return PIFUSafetyAssessment(
        requires_human_review=True,
        review_reason=reason,
        flags=flags,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        margin_threshold=MARGIN_THRESHOLD,
        top_two_margin=margin,
    )
