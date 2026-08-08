"""
<<<<<<< HEAD
Agentic model pipeline runner.
=======
<<<<<<< HEAD
Agentic model pipeline runner.
=======
Agentic Model Pipeline Runner
>>>>>>> d2ab366 ([ADD]: Sequential script execution timeline added!)
>>>>>>> 4250a8f ([ADD]: Sequential script execution timeline added!)

Runs the original and/or FastPIFU workflow in a fixed validated order and produces a structured pipeline report.

Run:
uv run python src/scripts/run_pipeline.py

Examples:
uv run python src/scripts/run_pipeline.py --target pifu --mode post-finetune
uv run python src/scripts/run_pipeline.py --target pifu --mode full --train-gpus auto --eval-gpu auto
uv run python src/scripts/run_pipeline.py --target pifu --mode post-finetune --no-llm-report
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

# Pipeline Components
from src.agents.pipeline_models import (
    PipelineMode,
    PipelineTarget,
)
from src.agents.pipeline_workflow import build_pipeline_graph

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline runner."""

    # CLI Parser Setup
    parser = argparse.ArgumentParser(
        description=(
            "Run the original and/or FastPIFU scripts in a fixed, validated order and produce a Pydantic structured report."
        )
    )

    # Pipeline Target
    parser.add_argument(
        "--target",
        choices=[
            target.value
            for target in PipelineTarget
        ],
        default=PipelineTarget.PIFU.value,
        help="Pipeline target to run.",
    )

    # Pipeline Mode
    parser.add_argument(
        "--mode",
        choices=[
            mode.value
            for mode in PipelineMode
        ],
        default=PipelineMode.POST_FINETUNE.value,
        help=(
            "full: prepare data, train, evaluate; "
            "train-and-evaluate: reuse prepared data; "
            "post-finetune: evaluate and report only."
        ),
    )

    # Training GPU Selection
    parser.add_argument(
        "--train-gpus",
        default="auto",
        help="Physical GPU IDs such as 0,2, or 'auto' to select two idle GPUs.",
    )

    # Evaluation GPU Selection
    parser.add_argument(
        "--eval-gpu",
        default="auto",
        help="One physical GPU ID, or 'auto' to select one idle GPU.",
    )

    # Data Preparation Control
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Force re-extraction of the FastPIFU ZIP.",
    )

    # Error Handling Policy
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue later independent stages after a failed stage. "
            "The default is to skip downstream scripts after failure."
        ),
    )

    # Reporting Mode
    parser.add_argument(
        "--no-llm-report",
        action="store_true",
        help=(
            "Create a deterministic Pydantic report without calling "
            "the Pydantic AI reporting agent."
        ),
    )

    # Optional Run ID
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run ID. Defaults to a UTC timestamp.",
    )

    return parser.parse_args()


def build_run_config(
    args: argparse.Namespace,
    run_id: str,
    run_dir: Path,
) -> dict:
    """Build the pipeline runtime configuration dictionary."""

    # Runtime Configuration
    return {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "target": args.target,
        "mode": args.mode,
        "train_gpus": args.train_gpus,
        "evaluation_gpu": args.eval_gpu,
        "force_prepare": args.force_prepare,
        "stop_on_error": not args.continue_on_error,
        "use_llm_report": not args.no_llm_report,
    }


def print_config(config: dict) -> None:
    """Print the pipeline configuration before execution."""

    # Configuration Summary
    print("=" * 72)
    print("AGENTIC MODEL PIPELINE")
    print("=" * 72)
    print(
        json.dumps(
            config,
            indent=2,
        )
    )
    print("=" * 72)


def print_final_report(report: dict) -> None:
    """Print the final pipeline report locations."""

    # Final Summary
    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print(f"Status: {report['assessment']['overall_status']}")
    print(f"JSON  : {report['report_json']}")
    print(f"MD    : {report['report_markdown']}")


async def main() -> None:
    """Run the compiled agentic pipeline graph."""

    # CLI Setup
    args = parse_args()

    # Run Directory Setup
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    run_dir = (
        PROJECT_ROOT
        / "data"
        / "pipeline_runs"
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Pipeline Configuration
    config = build_run_config(
        args=args,
        run_id=run_id,
        run_dir=run_dir,
    )

    print_config(config)

    # Pipeline Graph Execution
    graph = build_pipeline_graph()

    result = await graph.ainvoke({
        "config": config,
        "started_at": datetime.now(UTC).isoformat(),
        "steps": [],
        "metrics": [],
    })

    # Final Report Output
    report = result["final_report"]

    print_final_report(report)


if __name__ == "__main__":
    asyncio.run(main())