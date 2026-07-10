# Run the OMOP clinical agent across a synthetic patient cohort
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from src.agents.omop_agent import _setup_mlflow, run_agent
from src.data_access.connection import get_table


# Define the clinical questions used across the cohort
QUERIES = [
    "Summarise this patient's medical history including conditions, medications and recent visits.",
    "What conditions does this patient have and what medications have been prescribed?",
    "What recent measurements and laboratory values are recorded for this patient?",
    "Provide a clinical summary of this patient's visit history and procedures.",
    "Based on this patient's conditions, which FastPIFU specialty skill is most relevant and what does it say about PIFU suitability?",
]

# Define where cohort results will be stored
OUTPUT_DIR = Path("data/cohort_run_outputs")


def get_cohort(n: int = 10) -> list[int]:
    # Return an ordered sample of patient identifiers
    n = int(n)

    if n <= 0:
        raise ValueError("n must be a positive integer.")

    person = get_table("person")

    # Select a consistent cohort by ordering patient identifiers
    patient_ids = (
        person
        .select("person_id")
        .order_by("person_id")
        .limit(n)
        .execute()
    )

    return [
        int(person_id)
        for person_id in patient_ids["person_id"].tolist()
    ]


def run_cohort(n_patients: int = 10) -> None:
    # Run the clinical agent for a cohort of patients

    # Create the output directory and configure MLflow
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_mlflow()

    cohort = get_cohort(n_patients)

    print(f"Running cohort of {len(cohort)} patients")
    print(f"Patients: {cohort}")
    print("=" * 60)

    results = []

    # Assign queries to patients in a repeating sequence
    for index, person_id in enumerate(cohort):
        query = QUERIES[index % len(QUERIES)]

        print(
            f"\nPatient {person_id} | "
            f"Query: {query[:60]}..."
        )

        # Create one MLflow run for each patient
        with mlflow.start_run(
            run_name=f"cohort_patient_{person_id}",
            tags={
                "person_id": str(person_id),
                "sprint": "sprint_2",
                "run_type": "cohort",
                "placement": "lancashire_teaching_hospitals",
            }
        ):
            mlflow.log_param("person_id", person_id)
            mlflow.log_text(query, "query.txt")

            try:
                # Run the asynchronous agent inside the active MLflow run
                response = asyncio.run(
                    run_agent(person_id, query)
                )

                mlflow.set_tag("status", "success")
                mlflow.log_text(
                    response,
                    "agent_response.txt"
                )

                results.append({
                    "person_id": person_id,
                    "query": query,
                    "response": response,
                    "status": "success",
                    "error": None,
                })

                print(
                    f"  ✓ Success — {len(response)} characters"
                )
                print(
                    f"  Response preview: "
                    f"{response[:150]}..."
                )

            except Exception as error:
                error_message = str(error)

                mlflow.set_tag("status", "failed")
                mlflow.log_text(
                    error_message,
                    "error.txt"
                )

                results.append({
                    "person_id": person_id,
                    "query": query,
                    "response": None,
                    "status": "failed",
                    "error": error_message,
                })

                print(
                    f"  ✗ Failed — "
                    f"{error_message[:100]}"
                )
    
    # Create a timestamped output filename
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_file = (
        OUTPUT_DIR
        / f"cohort_run_{timestamp}.json"
    )

    # Save all cohort responses and failures
    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Create a timestamped output filename
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_file = (
        OUTPUT_DIR
        / f"cohort_run_{timestamp}.json"
    )

    # Save all cohort responses and failures
    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    successful = sum(
        result["status"] == "success"
        for result in results
    )
    failed = sum(
        result["status"] == "failed"
        for result in results
    )

    print("\n" + "=" * 60)
    print("COHORT RUN COMPLETE")
    print("=" * 60)
    print(f"  Total patients : {len(results)}")
    print(f"  Successful     : {successful}")
    print(f"  Failed         : {failed}")
    print(f"  Output saved   : {output_file}")
    print("  MLflow runs    : http://127.0.0.1:5000")


def main() -> None:
    # Configure the command-line cohort size
    parser = argparse.ArgumentParser(
        description="Run the OMOP agent across a patient cohort"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of patients to process"
    )

    args = parser.parse_args()
    run_cohort(args.n)


# Run the cohort evaluation when executed directly
if __name__ == "__main__":
    main()