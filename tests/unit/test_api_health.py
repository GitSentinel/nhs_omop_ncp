"""Tests for the FastAPI health endpoint"""
from fastapi.testclient import TestClient
from src.api.main import app

# Create the test client
client = TestClient(app)


# Define the test for the health check endpoint
def test_health_check_returns_ok() -> None:
    """The health endpoint should confirm that the API is available."""
    response = client.get("/health")

    # Check that the response is OK
    assert response.status_code == 200
    
    # Check that the response is as expected
    assert response.json() == {
        "status": "ok",
        "service": "nhs-omop-ncp-api",
    }