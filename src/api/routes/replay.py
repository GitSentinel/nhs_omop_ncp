"""API routes for replaying frozen dissertation predictions"""
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas.replay import PIFUReplayPrediction
from src.services.replay_provider import (
    ReplayPredictionNotFoundError,
    ReplayPredictionProvider,
)

# API Router for the replay routes
router = APIRouter(
    prefix="/api/v1/replay",
    tags=["Research Replay"],
)


# Dependency to get the replay provider
def get_replay_provider() -> ReplayPredictionProvider:
    """Provide access to the frozen dissertation prediction artefacts"""
    return ReplayPredictionProvider()


# Route to get a frozen PIFU prediction from the dissertation evaluation
@router.get(
    "/pifu/{sample_id}",
    response_model=PIFUReplayPrediction,
)
def get_pifu_replay_prediction(
    sample_id: str,
    provider: ReplayPredictionProvider = Depends(get_replay_provider),
) -> PIFUReplayPrediction:
    """Return a frozen PIFU prediction from the dissertation evaluation"""
    try:
        # Get the prediction from the provider
        return provider.get_prediction(sample_id)

    except ReplayPredictionNotFoundError as exc:
        # If the prediction is not found, raise a 404 error
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        # If the replay artefact is not found, raise a 503 error
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research replay artefact is unavailable.",
        ) from exc