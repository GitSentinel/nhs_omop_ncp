import json
from pathlib import Path

# Define the directory containing cohort-run results
OUTPUT_DIR = Path("data/cohort_run_outputs")


def get_latest_output() -> Path:
    # Find all timestamped cohort-run output files
    output_files = sorted(
        OUTPUT_DIR.glob("cohort_run_*.json")
    )

    if not output_files:
        raise FileNotFoundError(
            f"No cohort-run files found in: {OUTPUT_DIR}"
        )

    return output_files[-1]


def load_results(output_file: Path) -> list[dict]:
    # Read the stored cohort results using UTF-8 encoding
    with output_file.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def print_patient_summary(result: dict) -> None:
    person_id = result.get("person_id")
    query = result.get("query", "")
    status = result.get("status", "unknown")
    response = result.get("response")
    error = result.get("error")

    print(f"\nPatient {person_id}")
    print(f"  Query   : {query[:70]}")
    print(f"  Status  : {status}")

    # Print response statistics when a response is available
    if response:
        lines = response.strip().splitlines()

        print(f"  Lines   : {len(lines)}")
        print(f"  Length  : {len(response)} characters")
        print(f"  Preview : {response[:200]}")

    # Print the recorded failure message when present
    if error:
        print(f"  Error   : {error[:150]}")


def print_cohort_statistics(results: list[dict]) -> None:
    successful = [
        result
        for result in results
        if result.get("status") == "success"
    ]

    failed = [
        result
        for result in results
        if result.get("status") == "failed"
    ]

    total = len(results)

    print("\n" + "=" * 65)
    print(
        f"Success: {len(successful)}/{total}  |  "
        f"Failed: {len(failed)}/{total}"
    )

    if not successful:
        print("No successful responses available.")
        return

    # Calculate response lengths for successful runs
    response_lengths = [
        len(result.get("response", ""))
        for result in successful
    ]

    average_length = (
        sum(response_lengths)
        / len(response_lengths)
    )

    print(
        f"Average response length: "
        f"{average_length:.0f} characters"
    )
    print(
        f"Minimum response length: "
        f"{min(response_lengths)} characters"
    )
    print(
        f"Maximum response length: "
        f"{max(response_lengths)} characters"
    )


def main() -> None:
    # Identify and load the latest output file
    latest_output = get_latest_output()
    results = load_results(latest_output)

    print(f"Reading: {latest_output}\n")
    print("=" * 65)
    print("COHORT RUN SUMMARY")
    print("=" * 65)

    # Display each patient-level result
    for result in results:
        print_patient_summary(result)

    print_cohort_statistics(results)


# Run the summary when executed directly
if __name__ == "__main__":
    main()