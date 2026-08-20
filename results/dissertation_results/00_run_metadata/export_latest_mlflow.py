import json
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.config.settings import settings


experiment_name = sys.argv[1]
output_dir = Path(sys.argv[2])

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

mlflow.set_tracking_uri(
    settings.mlflow_tracking_uri
)

client = MlflowClient()

experiment = client.get_experiment_by_name(
    experiment_name
)

if experiment is None:
    raise SystemExit(
        f"Experiment not found: {experiment_name}"
    )

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

if not runs:
    raise SystemExit(
        f"No runs found for {experiment_name}"
    )

run = runs[0]

metadata = {
    "run_id": run.info.run_id,
    "run_name": run.info.run_name,
    "experiment_id": run.info.experiment_id,
    "status": run.info.status,
    "start_time": run.info.start_time,
    "end_time": run.info.end_time,
    "artifact_uri": run.info.artifact_uri,
}

(output_dir / "run_id.txt").write_text(
    run.info.run_id + "\n",
    encoding="utf-8",
)

(output_dir / "run_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

(output_dir / "params.json").write_text(
    json.dumps(run.data.params, indent=2),
    encoding="utf-8",
)

(output_dir / "metrics.json").write_text(
    json.dumps(run.data.metrics, indent=2),
    encoding="utf-8",
)

(output_dir / "tags.json").write_text(
    json.dumps(run.data.tags, indent=2),
    encoding="utf-8",
)

artifact_dir = output_dir / "artifacts"
artifact_dir.mkdir(exist_ok=True)

try:
    mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path="",
        dst_path=str(artifact_dir),
        tracking_uri=settings.mlflow_tracking_uri,
    )
except Exception as exc:
    (output_dir / "artifact_download_error.txt").write_text(
        str(exc),
        encoding="utf-8",
    )

print("Experiment:", experiment_name)
print("Run ID:", run.info.run_id)
print("Saved:", output_dir)
