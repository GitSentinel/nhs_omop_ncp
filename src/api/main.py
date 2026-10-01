"""FastAPI application for the NHS OMOP NCP end-to-end platform"""

from fastapi import FastAPI

# Create the FastAPI app
app = FastAPI(
    title="NHS OMOP NCP API",
    description=(
        "Research-only API for the synthetic NHS OMOP and PIFU "
        "decision-support platform."
    ),
    version="0.1.0",
)


# Define the health check endpoint
@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""
    return {
        "status": "ok",
        "service": "nhs-omop-ncp-api",
    }