"""Tests for the Research Replay API"""
from fastapi.testclient import TestClient
from src.api.main import app

# Create the test client
client = TestClient(app)

# Test to check that a known dissertation sample returns its saved prediction
def test_replay_endpoint_returns_frozen_prediction() -> None:
    """A known dissertation sample should return its saved prediction"""
    response = client.get("/api/v1/replay/pifu/eval_3239")

    # Check that the response is OK 
    assert response.status_code == 200

    # Check the response payload
    payload = response.json()

    # Check the sample ID
    assert payload["sample_id"] == "eval_3239"
    assert payload["source"] == "eval_sample_150_v3"
    assert payload["reference_label"] == "ELIGIBLE"
    assert payload["predicted_label"] == "ELIGIBLE"

    # Check the replay mode and live inference
    assert payload["replay_mode"] is True
    assert payload["live_inference"] is False

    # Check the class scores
    assert payload["class_scores"]["eligible"] > 0.99


def test_replay_endpoint_returns_404_for_unknown_sample() -> None:
    """An unknown replay sample should return HTTP 404"""
    response = client.get("/api/v1/replay/pifu/does-not-exist")

    # Check that the response is a 404
    assert response.status_code == 404

    # Check the response payload
    assert response.json() == {
        "detail": (
            "No frozen prediction found for "
            "sample_id='does-not-exist'"
        )
    }