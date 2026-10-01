"""Tests for the frozen dissertation replay provider"""
from pathlib import Path

import pytest

from src.services.replay_provider import (
    ReplayPredictionNotFoundError,
    ReplayPredictionProvider,
)


def test_replay_provider_returns_saved_prediction() -> None:
    """A known dissertation sample should return its frozen prediction"""
    provider = ReplayPredictionProvider()

    prediction = provider.get_prediction("eval_3239")

    # Check the prediction attributes
    assert prediction.sample_id == "eval_3239"
    assert prediction.source == "eval_sample_150_v3"
    assert prediction.reference_label == "ELIGIBLE"
    assert prediction.predicted_label == "ELIGIBLE"

    # Check the class scores
    assert prediction.class_scores.not_eligible == pytest.approx(
        0.0015010947827249765
    )
    assert prediction.class_scores.borderline == pytest.approx(
        5.8203710068482906e-05
    )
    assert prediction.class_scores.eligible == pytest.approx(
        0.998440682888031
    )

    # Check the replay mode and live inference
    assert prediction.replay_mode is True
    assert prediction.live_inference is False


def test_replay_provider_rejects_unknown_sample() -> None:
    """An unknown sample ID should raise a clear lookup error"""
    provider = ReplayPredictionProvider()

    with pytest.raises(
        ReplayPredictionNotFoundError,
        match="No frozen prediction found",
    ):
        provider.get_prediction("does-not-exist")


def test_replay_provider_rejects_missing_csv(tmp_path: Path) -> None:
    """A missing replay artefact should raise FileNotFoundError"""
    missing_path = tmp_path / "missing_predictions.csv"
    provider = ReplayPredictionProvider(csv_path=missing_path)

    with pytest.raises(FileNotFoundError, match="Replay prediction file not found"):
        provider.get_prediction("eval_3239")