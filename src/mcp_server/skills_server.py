import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import Context, FastMCP


# Configure server logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Define the directory containing downloaded skill documents
SKILLS_DIR = (
    Path(__file__).resolve().parent
    / "resources"
    / "skills"
)

# Map accepted specialty names to their local files
SPECIALTY_MAP = {
    "cardiology": "cardiology.md",
    "dermatology": "dermatology.md",
    "ent": "ent.md",
    "gastroenterology": "gastroenterology.md",
    "general_surgery": "general_surgery.md",
    "gynaecology": "gynaecology.md",
    "omfs": "omfs.md",
    "ophthalmology": "ophthalmology.md",
    "orthopaedics": "orthopaedics.md",
    "spinal": "spinal.md",
    "urology": "urology.md",
    "omop_clinical_reasoning": "omop_clinical_reasoning.md",
}

# Define keywords used for approximate condition routing
CONDITION_SPECIALTY_MAP = {
    # Cardiology
    "cardiology": (
        "heart",
        "cardiac",
        "arrhythmia",
        "atrial fibrillation",
        "heart failure",
        "angina",
        "myocardial",
        "coronary",
        "cardiomyopathy",
        "pericarditis",
        "valve disease",
        "bundle branch block",
        "postural tachycardia syndrome",
    ),

    # Gastroenterology
    "gastroenterology": (
        "liver",
        "cirrhosis",
        "hepatitis",
        "crohn disease",
        "crohn's disease",
        "colitis",
        "coeliac disease",
        "gastrointestinal",
        "inflammatory bowel disease",
        "fatty liver disease",
        "masld",
        "nafld",
    ),

    # Dermatology
    "dermatology": (
        "skin",
        "dermatology",
        "dermatological",
        "melanoma",
        "acne",
        "psoriasis",
        "eczema",
        "keratosis",
        "skin carcinoma",
        "skin lesion",
    ),

    # Orthopaedics
    "orthopaedics": (
        "joint",
        "knee",
        "hip",
        "shoulder",
        "fracture",
        "osteoarthritis",
        "carpal tunnel",
        "tendon",
        "ligament",
        "bone injury",
    ),

    # Ophthalmology
    "ophthalmology": (
        "eye",
        "ocular",
        "retina",
        "retinal",
        "glaucoma",
        "cataract",
        "macular",
        "visual impairment",
        "diabetic retinopathy",
    ),

    # Urology
    "urology": (
        "bladder",
        "prostate",
        "urology",
        "urological",
        "urinary",
        "incontinence",
        "kidney stone",
        "renal stone",
        "ureter",
        "urethra",
    ),

    # Gynaecology
    "gynaecology": (
        "gynaecology",
        "gynaecological",
        "uterine",
        "ovarian",
        "pelvic",
        "endometriosis",
        "uterine prolapse",
        "vaginal prolapse",
    ),

    # ENT
    "ent": (
        "ear",
        "nose",
        "throat",
        "sinus",
        "tonsil",
        "hearing",
        "nasal",
        "septum",
        "otitis",
        "tinnitus",
    ),

    # Spinal
    "spinal": (
        "spine",
        "spinal",
        "lumbar",
        "intervertebral disc",
        "scoliosis",
        "vertebra",
        "back pain",
        "sciatica",
    ),

    # General Surgery
    "general_surgery": (
        "hernia",
        "gallbladder",
        "colorectal",
        "cholecystectomy",
        "haemorrhoid",
        "anal fistula",
        "bowel resection",
    ),
    
    # OMFS
    "omfs": (
        "jaw",
        "dental",
        "oral",
        "maxillofacial",
        "mandibular",
        "temporomandibular",
        "tmj",
    ),
}


def _normalise_specialty(specialty: str) -> str:
    return (
        specialty
        .casefold()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _normalise_condition(condition_name: str) -> str:
    # Replace punctuation with spaces and remove repeated whitespace
    cleaned = re.sub(
        r"[^a-z0-9]+",
        " ",
        condition_name.casefold()
    )

    return " ".join(cleaned.split())


def _keyword_matches(
    condition_name: str,
    keyword: str
) -> bool:
    normalised_keyword = _normalise_condition(keyword)

    # Use word boundaries to avoid partial matches such as ear in heart
    pattern = rf"\b{re.escape(normalised_keyword)}\b"

    return re.search(pattern, condition_name) is not None


@asynccontextmanager
async def skills_lifespan(
    _server: FastMCP
) -> AsyncIterator[dict]:
    log.info("FastPIFU Skills MCP server starting")
    log.info("Loading skills from: %s", SKILLS_DIR)

    loaded_skills = {}

    # Load each available skill document into memory
    for specialty, filename in SPECIALTY_MAP.items():
        skill_path = SKILLS_DIR / filename

        if not skill_path.is_file():
            log.warning(
                "Skill file not found for %s: %s",
                specialty,
                skill_path
            )
            continue

        try:
            loaded_skills[specialty] = skill_path.read_text(
                encoding="utf-8"
            )
        except (OSError, UnicodeDecodeError) as error:
            log.warning(
                "Could not load skill file for %s: %s",
                specialty,
                error
            )

    log.info(
        "Loaded %d FastPIFU skills: %s",
        len(loaded_skills),
        sorted(loaded_skills)
    )

    try:
        # Share the loaded documents with every tool request
        yield {"skills": loaded_skills}
    finally:
        loaded_skills.clear()
        log.info("FastPIFU Skills MCP server stopped")


# Create the FastPIFU skills server
mcp = FastMCP(
    name="nhs-fastpifu-skills",
    instructions=(
        "FastPIFU clinical skills server for Lancashire Teaching Hospitals NHS Foundation Trust. \nCall list_skills first, then use get_skill to retrieve the relevant specialty protocol. \nCondition-based routing is a keyword-based suggestion and must not be treated as a clinical decision."
    ),
    lifespan=skills_lifespan
)


def _get_loaded_skills(ctx: Context) -> dict[str, str]:
    return ctx.lifespan_context["skills"]


def _get_skill_content(
    specialty: str,
    ctx: Context
) -> str:
    specialty = _normalise_specialty(specialty)
    loaded_skills = _get_loaded_skills(ctx)
    available = sorted(loaded_skills)

    if specialty not in SPECIALTY_MAP:
        return (
            f"Specialty '{specialty}' is not recognised. \nAvailable specialties: {', '.join(available)}"
        )

    if specialty not in loaded_skills:
        return (
            f"The skill file for '{specialty}' is not available. \nRun: uv run python src/scripts/download_skills.py"
        )

    log.info("Returning FastPIFU skill: %s", specialty)

    return loaded_skills[specialty]


@mcp.tool()
def list_skills(ctx: Context) -> list[str]:
    available = sorted(_get_loaded_skills(ctx))

    log.info(
        "list_skills available_count=%d",
        len(available)
    )

    return available


@mcp.tool()
def get_skill(
    specialty: str,
    ctx: Context
) -> str:
    return _get_skill_content(
        specialty,
        ctx
    )


@mcp.tool()
def get_skill_for_condition(
    condition_name: str,
    ctx: Context
) -> str:
    condition_name = condition_name.strip()

    if not condition_name:
        return "A condition name must be provided."

    normalised_condition = _normalise_condition(
        condition_name
    )

    # Search the specialty routing keywords in their defined order
    for specialty, keywords in CONDITION_SPECIALTY_MAP.items():
        matched_keyword = next(
            (
                keyword
                for keyword in keywords
                if _keyword_matches(
                    normalised_condition,
                    keyword
                )
            ),
            None
        )

        if matched_keyword is None:
            continue

        log.info(
            "Condition '%s' matched specialty '%s' using keyword '%s'",
            condition_name,
            specialty,
            matched_keyword
        )

        return _get_skill_content(
            specialty,
            ctx
        )

    available = sorted(_get_loaded_skills(ctx))

    return (
        f"No FastPIFU specialty was matched for '{condition_name}'. \nAvailable specialties: {', '.join(available)}. \nCall get_skill with the most appropriate specialty."
    )

@mcp.tool()
def get_omop_reasoning_guide(ctx: Context) -> str:
    # Return the OMOP clinical reasoning guide for this agent.
    path = SKILLS_DIR / "omop_clinical_reasoning.md"

    if not path.exists():
        return "OMOP reasoning guide not found. Run download_skills.py first."
    
    log.info("get_omop_reasoning_guide loaded")
    return path.read_text(encoding="utf-8")


# Start the stdio server when executed directly
if __name__ == "__main__":
    log.info(
        "Starting NHS FastPIFU Skills MCP server using stdio"
    )

    mcp.run(transport="stdio")