# FastMCP stdio server exposing patient-level OMOP CDM v5.4 tools
import logging
import math
from datetime import date, datetime
from typing import Any

from fastmcp import FastMCP

from src.config.settings import settings
from src.data_access.connection import get_table
from src.data_access.omop_queries import (
    get_conditions,
    get_measurements,
    get_medications,
    get_notes,
    get_observations,
    get_person,
    get_procedures,
    get_visits,
)

# Configure server logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP(
    name="nhs-omop-mcp",
    instructions=(
        "OMOP CDM v5.4 clinical data server using the delphi-100k "
        "synthetic dataset. Use get_patient_summary first, then call "
        "the relevant clinical-domain tools."
    ),
)


def _serialise(value: Any) -> Any:
    # Convert values into JSON-compatible Python types
    if value is None:
        return None

    if isinstance(value, dict):
        return {key: _serialise(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    # Convert NumPy scalar values to standard Python values
    if hasattr(value, "item"):
        try:
            return _serialise(value.item())
        except (TypeError, ValueError):
            pass

    # Convert missing floating-point values to None
    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def _validate_limit(limit: int, maximum: int = 100) -> int:
    # Restrict a requested limit to a valid range
    return max(1, min(int(limit), maximum))


@mcp.tool()
def get_patient_summary(person_id: int) -> dict:
    # Return demographic information for a patient
    log.info("get_patient_summary person_id=%s", person_id)
    return _serialise(get_person(person_id))


@mcp.tool()
def get_patient_conditions(person_id: int) -> list[dict]:
    # Return condition records for a patient
    log.info("get_patient_conditions person_id=%s", person_id)
    return _serialise(get_conditions(person_id))


@mcp.tool()
def get_patient_medications(person_id: int) -> list[dict]:
    # Return medication records for a patient
    log.info("get_patient_medications person_id=%s", person_id)
    return _serialise(get_medications(person_id))


@mcp.tool()
def get_patient_visits(person_id: int) -> list[dict]:
    # Return healthcare visit records for a patient
    log.info("get_patient_visits person_id=%s", person_id)
    return _serialise(get_visits(person_id))


@mcp.tool()
def get_patient_measurements(
    person_id: int,
    limit: int = 50
) -> list[dict]:
    # Return recent measurement records for a patient
    limit = _validate_limit(limit)

    log.info(
        "get_patient_measurements person_id=%s limit=%s",
        person_id,
        limit
    )

    return _serialise(
        get_measurements(person_id, limit=limit)
    )


@mcp.tool()
def get_patient_observations(person_id: int) -> list[dict]:
    # Return observation records for a patient
    log.info("get_patient_observations person_id=%s", person_id)
    return _serialise(get_observations(person_id))


@mcp.tool()
def get_patient_notes(person_id: int) -> list[dict]:
    # Return clinical notes for a patient
    log.info("get_patient_notes person_id=%s", person_id)
    return _serialise(get_notes(person_id))


@mcp.tool()
def get_patient_procedures(person_id: int) -> list[dict]:
    # Return procedure records for a patient
    log.info("get_patient_procedures person_id=%s", person_id)
    return _serialise(get_procedures(person_id))


@mcp.tool()
def list_available_patients(limit: int = 20) -> list[int]:
    # Return a sample of available patient identifiers
    limit = _validate_limit(limit)

    log.info("list_available_patients limit=%s", limit)

    patient_ids = (
        get_table("person")
        .select("person_id")
        .order_by("person_id")
        .limit(limit)
        .execute()
    )

    return [
        int(person_id)
        for person_id in patient_ids["person_id"].tolist()
    ]


# Start the server when this file is run directly
if __name__ == "__main__":
    log.info("Starting NHS OMOP MCP server using stdio")
    log.info("DuckDB path: %s", settings.duckdb_path)
    mcp.run(transport="stdio")