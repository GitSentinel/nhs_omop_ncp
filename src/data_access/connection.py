# Create and manage the Ibis connection to the OMOP DuckDB database
from functools import lru_cache
from pathlib import Path

import ibis
from ibis import BaseBackend

from src.config.settings import settings


@lru_cache(maxsize=2)
def get_connection(read_only: bool = True) -> BaseBackend:
    # Return a cached Ibis DuckDB connection
    db_path = Path(settings.duckdb_path).expanduser().resolve()

    # Check that the configured database file exists
    if not db_path.is_file():
        raise FileNotFoundError(
            f"OMOP DuckDB database not found at: {db_path}\n"
            "Run `uv run python scripts/extract_omop.py` first."
        )

    # Open the database using the requested access mode
    return ibis.duckdb.connect(
        database=str(db_path),
        read_only=read_only
    )


def list_omop_tables() -> list[str]:
    # Return the available OMOP table names in alphabetical order
    return sorted(get_connection().list_tables())


def get_table(table_name: str):
    # Return a lazy Ibis expression for an OMOP table
    connection = get_connection()
    available_tables = connection.list_tables()

    # Reject table names that are not present in the database
    if table_name not in available_tables:
        raise ValueError(
            f"Table '{table_name}' was not found. "
            f"Available tables: {sorted(available_tables)}"
        )

    return connection.table(table_name)