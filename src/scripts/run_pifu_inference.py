"""
FastPIFU Inference CLI

Runs the fine-tuned FastPIFU inference agent on one synthetic clinic letter
provided either as a text file or as direct command-line text.

Run with a file:
uv run --extra finetune python src/scripts/run_pifu_inference.py --letter-file letter.txt

Run with direct text:
uv run --extra finetune python src/scripts/run_pifu_inference.py --text "Clinic letter text here"

Disable LLM explanation:
uv run --extra finetune python src/scripts/run_pifu_inference.py --letter-file letter.txt --no-llm-explanation
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


def parse_args() -> argparse.Namespace:
    """Parse inference CLI arguments."""

    # CLI Parser Setup
    parser = argparse.ArgumentParser(
        description="Run the fine-tuned PIFU inference agent."
    )

    # Input Source
    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--letter-file",
        type=Path,
        help="Path to a synthetic clinic letter text file.",
    )

    source.add_argument(
        "--text",
        type=str,
        help="Clinic-letter text.",
    )

    # GPU Selection
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Physical GPU ID used for model inference.",
    )

    # Explanation Control
    parser.add_argument(
        "--no-llm-explanation",
        action="store_true",
        help="Skip Pydantic AI narrative generation.",
    )

    # Optional Run ID
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run ID. Defaults to a UTC timestamp.",
    )

    return parser.parse_args()


def load_text(
    args: argparse.Namespace,
) -> str:
    """Load clinic-letter text from CLI text or a file."""

    # Direct Text Input
    if args.text is not None:
        text = str(args.text).strip()

        if not text:
            raise ValueError("Clinic-letter text must not be empty.")

        return text

    # File Text Input
    path = Path(args.letter_file)

    if not path.exists():
        raise FileNotFoundError(f"Letter file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Letter path is not a file: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Letter file is empty: {path}")

    return text


def configure_runtime_environment(
    gpu_id: int,
) -> None:
    """Configure runtime environment before model imports."""

    # GPU Visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Tokeniser Runtime Setting
    os.environ.setdefault(
        "TOKENIZERS_PARALLELISM",
        "false",
    )


def build_run_directory(
    run_id: str,
) -> Path:
    """Return the output directory for one inference run."""

    # Run Directory Path
    return PROJECT_ROOT / "data" / "inference_runs" / run_id


def print_final_summary(
    report: dict,
) -> None:
    """Print the final inference summary."""

    # Report Extraction
    prediction = report["prediction"]

    # Console Summary
    print()
    print("=" * 68)
    print("PIFU INFERENCE COMPLETE")
    print("=" * 68)
    print(f"Prediction : {prediction['predicted_class']}")
    print(f"Confidence : {prediction['confidence']:.4f}")
    print("Review     : REQUIRED")
    print(f"JSON       : {report['report_json']}")
    print(f"Markdown   : {report['report_markdown']}")


async def main() -> None:
    """Run PIFU inference."""

    # CLI Setup
    args = parse_args()

    # Runtime Setup
    configure_runtime_environment(args.gpu)

    # Input Loading
    text = load_text(args)

    # Run ID and Directory
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = build_run_directory(run_id)

    # Import after CUDA_VISIBLE_DEVICES has been configured.
    from src.agents.pifu_inference_workflow import build_pifu_inference_graph

    # Graph Execution
    graph = build_pifu_inference_graph()

    result = await graph.ainvoke(
        {
            "config": {
                "run_id": run_id,
                "run_dir": str(run_dir.resolve()),
                "use_llm_explanation": not args.no_llm_explanation,
            },
            "text": text,
        }
    )

    # Final Summary
    print_final_summary(result["final_report"])


if __name__ == "__main__":
    asyncio.run(main())
