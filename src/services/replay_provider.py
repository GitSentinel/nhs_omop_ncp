"""
Provider for frozen dissertation PIFU predictions
"""

import csv
from pathlib import Path

from src.api.schemas.replay import PIFUClassScores, PIFUReplayPrediction

# Default path to the frozen PIFU predictions
DEFAULT_REPLAY_PATH = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "dissertation_results"
    / "02_pifu_model"
    / "artifacts"
    / "finetuned_external_test_predictions.csv"
)


# Error raised when a requested replay prediction does not exist
class ReplayPredictionNotFoundError(LookupError):
    """Error raised when a requested replay prediction does not exist"""


class ReplayPredictionProvider:
    """Provider for frozen PIFU predictions produced during dissertation evaluation"""

    def __init__(self, csv_path: Path = DEFAULT_REPLAY_PATH) -> None:
        self.csv_path = csv_path

    def get_prediction(self, sample_id: str) -> PIFUReplayPrediction:
        """Return the saved prediction for one dissertation evaluation sample"""
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Replay prediction file not found: {self.csv_path}"
            )

        # Open the CSV file and read the predictions
        with self.csv_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            # Iterate over the rows in the CSV file
            for row in reader:
                # Check if the sample_id matches the requested sample_id
                if row["sample_id"] == sample_id:
                    # Return the PIFUReplayPrediction object
                    return PIFUReplayPrediction(
                        sample_id=row["sample_id"],
                        source=row["source"],
                        reference_label=row["true_label_name"],
                        predicted_label=row["predicted_label_name"],
                        class_scores=PIFUClassScores(
                            not_eligible=float(row["prob_not_eligible"]),
                            borderline=float(row["prob_borderline"]),
                            eligible=float(row["prob_eligible"]),
                        ),
                    )

        # If no prediction is found, raise an error
        raise ReplayPredictionNotFoundError(
            f"No frozen prediction found for sample_id={sample_id!r}"
        )