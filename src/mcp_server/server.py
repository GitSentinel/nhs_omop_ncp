# FastMCP server exposing patient-level OMOP CDM v5.4 tools.
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import ibis
from fastmcp import Context, FastMCP

from src.config.settings import settings
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


@asynccontextmanager
async def omop_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    # Resolve and validate the configured database path
    db_path = Path(settings.duckdb_path).expanduser().resolve()

    if not db_path.is_file():
        raise FileNotFoundError(
            f"OMOP DuckDB database not found at: {db_path}"
        )

    log.info("OMOP MCP server starting")
    log.info("Connecting to DuckDB: %s", db_path)

    # Open one read-only connection when the server starts
    connection = ibis.duckdb.connect(
        database=str(db_path),
        read_only=True
    )

    log.info(
        "DuckDB connection established with %d tables",
        len(connection.list_tables())
    )

    try:
        # Make the connection available through the MCP context
        yield {"db": connection}
    finally:
        # Close the database connection when the server stops
        connection.disconnect()
        log.info("DuckDB connection closed")


# Create the MCP server and register its lifespan manager
mcp = FastMCP(
    name="nhs-omop-mcp",
    instructions=(
        "OMOP CDM v5.4 clinical data server using the delphi-100k "
        "synthetic dataset. Use get_patient_summary first, then call "
        "the relevant clinical-domain tools."
    ),
    lifespan=omop_lifespan
)


def _serialise(value: Any) -> Any:
    # Convert values into JSON-compatible Python types
    if value is None:
        return None

    # Recursively serialise dictionaries
    if isinstance(value, dict):
        return {
            key: _serialise(item)
            for key, item in value.items()
        }

    # Recursively serialise lists and tuples
    if isinstance(value, (list, tuple)):
        return [
            _serialise(item)
            for item in value
        ]

    # Convert dates into ISO-formatted strings
    if isinstance(value, (date, datetime)):
        return value.isoformat()

    # Convert NumPy scalar values into standard Python values
    if hasattr(value, "item"):
        try:
            return _serialise(value.item())
        except (TypeError, ValueError):
            pass

    # Convert missing floating-point values into None
    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def _validate_limit(limit: int, maximum: int = 100) -> int:
    # Ensure the limit remains between 1 and the maximum
    return max(1, min(int(limit), maximum))


@mcp.tool()
def get_patient_summary(person_id: int, ctx: Context) -> dict:
    # Record the tool call for monitoring
    log.info("get_patient_summary person_id=%s", person_id)
    return _serialise(get_person(person_id))


@mcp.tool()
def get_patient_conditions(
    person_id: int,
    ctx: Context
) -> list[dict]:
    # Return condition records for a patient
    log.info("get_patient_conditions person_id=%s", person_id)
    return _serialise(get_conditions(person_id))


@mcp.tool()
def get_patient_medications(
    person_id: int,
    ctx: Context
) -> list[dict]:
    # Return medication records for a patient
    log.info("get_patient_medications person_id=%s", person_id)
    return _serialise(get_medications(person_id))


@mcp.tool()
def get_patient_visits(
    person_id: int,
    ctx: Context
) -> list[dict]:
    # Return healthcare visit records for a patient
    log.info("get_patient_visits person_id=%s", person_id)
    return _serialise(get_visits(person_id))


@mcp.tool()
def get_patient_measurements(
    person_id: int,
    ctx: Context,
    limit: int = 50
) -> list[dict]:
    # Restrict the number of returned measurement records
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
def get_patient_observations(
    person_id: int,
    ctx: Context
) -> list[dict]:
    # Return observation records for a patient
    log.info("get_patient_observations person_id=%s", person_id)
    return _serialise(get_observations(person_id))


@mcp.tool()
def get_patient_notes(
    person_id: int,
    ctx: Context
) -> list[dict]:
    # Return clinical notes for a patient
    log.info("get_patient_notes person_id=%s", person_id)
    return _serialise(get_notes(person_id))


@mcp.tool()
def get_patient_procedures(
    person_id: int,
    ctx: Context
) -> list[dict]:
    # Return procedure records for a patient
    log.info("get_patient_procedures person_id=%s", person_id)
    return _serialise(get_procedures(person_id))


@mcp.tool()
def list_available_patients(
    ctx: Context,
    limit: int = 20
) -> list[int]:    
    # Restrict the requested number of identifiers
    limit = _validate_limit(limit)

    # Access the shared lifespan database connection
    database = ctx.lifespan_context["db"]
    log.info("list_available_patients limit=%s", limit)

    # Retrieve a consistent ordered sample of patient identifiers
    patient_ids = (
        database
        .table("person")
        .select("person_id")
        .order_by("person_id")
        .limit(limit)
        .execute()
    )

    # Convert NumPy integer values into standard Python integers
    return [
        int(person_id)
        for person_id in patient_ids["person_id"].tolist()
    ]


# Start the stdio server when this file is executed directly
if __name__ == "__main__":
    log.info("Starting NHS OMOP MCP server using stdio")
    log.info("DuckDB path: %s", settings.duckdb_path)

    mcp.run(transport="stdio")