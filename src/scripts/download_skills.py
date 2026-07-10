from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Define the local directory used to store skill files
SKILLS_DIR = Path("src/mcp_server/resources/skills")

# Define the raw GitHub location containing the specialty skills
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "lsc-sde/fastpifu-skills/main/skills"
)

# Define the specialties available in the FastPIFU repository
SPECIALTIES = (
    "cardiology",
    "dermatology",
    "ent",
    "gastroenterology",
    "general_surgery",
    "gynaecology",
    "omfs",
    "ophthalmology",
    "orthopaedics",
    "spinal",
    "urology",
)

REQUEST_TIMEOUT = 20
USER_AGENT = "OMOP-FastPIFU-Skill-Downloader/1.0"


def build_skill_url(specialty: str) -> str:
    # Convert Python-style names into repository folder names
    specialty_slug = specialty.replace("_", "-")

    return (
        f"{RAW_BASE_URL}/"
        f"fastpifu-{specialty_slug}/SKILL.md"
    )


def download_skill(specialty: str) -> Path:
    url = build_skill_url(specialty)

    # Add a user agent to identify the download request
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT}
    )

    # Download the raw Markdown content
    with urlopen(
        request,
        timeout=REQUEST_TIMEOUT
    ) as response:
        content = response.read().decode("utf-8")

    # Reject empty or unexpectedly rendered HTML responses
    if not content.strip():
        raise ValueError("Downloaded file is empty.")

    if "<html" in content.lower():
        raise ValueError(
            "Received HTML instead of raw Markdown."
        )

    output_path = SKILLS_DIR / f"{specialty}.md"

    # Save the original Markdown content without modification
    output_path.write_text(
        content,
        encoding="utf-8"
    )

    return output_path


def main() -> None:
    # Create the destination directory when it does not exist
    SKILLS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    successful = []
    failed = []

    # Download each specialty independently
    for specialty in SPECIALTIES:
        print(f"Fetching {specialty}...")

        try:
            output_path = download_skill(specialty)
            successful.append(specialty)

            print(f"  ✓ Saved to {output_path}")

        except (
            HTTPError,
            URLError,
            TimeoutError,
            UnicodeDecodeError,
            ValueError,
            OSError,
        ) as error:
            failed.append({
                "specialty": specialty,
                "error": str(error),
            })

            print(f"  ✗ Failed: {error}")

    print("\n" + "=" * 60)
    print("FASTPIFU SKILL DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Successful : {len(successful)}")
    print(f"Failed     : {len(failed)}")
    print(f"Directory  : {SKILLS_DIR}")

    # Return a non-zero exit status when downloads fail
    if failed:
        print("\nFailed specialties:")

        for failure in failed:
            print(
                f"  - {failure['specialty']}: "
                f"{failure['error']}"
            )

        raise SystemExit(1)


# Run the downloader when executed directly
if __name__ == "__main__":
    main()