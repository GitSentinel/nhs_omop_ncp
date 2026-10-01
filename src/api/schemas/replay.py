"""
Schemas for replaying frozen dissertation model predictions
"""

from typing import Literal
from pydantic import BaseModel, Field

# PIFU Label Types
PIFULabel = Literal["NOT_ELIGIBLE", "BORDERLINE", "ELIGIBLE"]


class PIFUClassScores(BaseModel):
    """Schema for PIFU class scores"""
    not_eligible: float = Field(ge=0.0, le=1.0)
    borderline: float = Field(ge=0.0, le=1.0)
    eligible: float = Field(ge=0.0, le=1.0)


class PIFUReplayPrediction(BaseModel):
    """Schema for a frozen PIFU prediction from the dissertation evaluation artefacts."""
    sample_id: str
    source: str
    reference_label: PIFULabel
    predicted_label: PIFULabel
    class_scores: PIFUClassScores

    replay_mode: Literal[True] = True
    live_inference: Literal[False] = False