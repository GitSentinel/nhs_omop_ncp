import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# MCP Server Library
from fastmcp import Context, FastMCP

# Project Configuration
from config import (
    CONDITION_SPECIALTY_MAP,
    SKILLS_DIR,
    SPECIALTY_MAP,
)

# Logging Setup
logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


def normalise_specialty_name(specialty: str) -> str:
    return (
        specialty
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )


def get_available_skills() -> list[str]:
    # Available Skill Detection
    return sorted([
        specialty
        for specialty, filename in SPECIALTY_MAP.items()
        if (SKILLS_DIR / filename).exists()
    ])


@asynccontextmanager
async def skills_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    # Startup Metadata
    log.info("Skills MCP server starting up")

    available_skills = get_available_skills()

    log.info("Available skills: %s", available_skills)

    try:
        yield {"available_skills": available_skills}
    finally:
        log.info("Skills MCP server shut down")


# MCP Server Setup
mcp = FastMCP(
    name="nhs-fastpifu-skills",
    instructions=(
        "FastPIFU clinical skills server for Lancashire Teaching "
        "Hospitals NHS FT. Provides specialty-specific PIFU clinical "
        "protocol documents. Call list_skills first, then get_skill "
        "for the relevant specialty."
    ),
    lifespan=skills_lifespan,
)


@mcp.tool()
def list_skills(ctx: Context) -> list[str]:
    # Skill List Retrieval
    available_skills = ctx.lifespan_context["available_skills"]

    log.info("list_skills - %d skills available", len(available_skills))

    return available_skills


@mcp.tool()
def get_skill(specialty: str, ctx: Context) -> str:
    # Specialty Lookup
    specialty_key = normalise_specialty_name(specialty)

    if specialty_key not in SPECIALTY_MAP:
        available_skills = ctx.lifespan_context["available_skills"]

        return (
            f"Specialty '{specialty_key}' not found. "
            f"Available: {', '.join(available_skills)}"
        )

    # Skill File Loading
    skill_path = SKILLS_DIR / SPECIALTY_MAP[specialty_key]

    if not skill_path.exists():
        return f"Skill file for '{specialty_key}' not downloaded yet."

    log.info("get_skill specialty=%s", specialty_key)

    return skill_path.read_text(encoding="utf-8")


@mcp.tool()
def get_skill_for_condition(condition_name: str, ctx: Context) -> str:
    # Condition Matching
    condition_lower = condition_name.lower().strip()

    for specialty, keywords in CONDITION_SPECIALTY_MAP.items():
        if any(keyword.lower() in condition_lower for keyword in keywords):
            log.info(
                "get_skill_for_condition %s -> %s",
                condition_name,
                specialty,
            )

            return get_skill(specialty, ctx)

    # Fallback Response
    available_skills = ctx.lifespan_context["available_skills"]

    return (
        f"No specific skill matched '{condition_name}'. "
        f"Available: {', '.join(available_skills)}. "
        "Try get_skill() with the most relevant specialty."
    )


@mcp.tool()
def get_omop_reasoning_guide(ctx: Context) -> str:
    # Reasoning Guide Loading
    guide_path = SKILLS_DIR / "omop_clinical_reasoning.md"

    if not guide_path.exists():
        return "OMOP reasoning guide not found."

    log.info("get_omop_reasoning_guide loaded")

    return guide_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    # Server Execution
    log.info("Starting NHS FastPIFU Skills MCP server over stdio")

    mcp.run(transport="stdio")