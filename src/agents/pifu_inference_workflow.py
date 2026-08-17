"""
FastPIFU Single-Letter Inference Workflow

Builds a LangGraph workflow for validating one clinic letter, running the
fine-tuned FastPIFU classifier, applying deterministic safety rules,
generating an explanation, saving reports, and logging MLflow artifacts.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Experiment Tracking
import mlflow

# LangGraph Components
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

# Typing Utilities
from typing_extensions import TypedDict

# FastPIFU Decision Models
from src.agents.pifu_decision_models import (
    PIFUExplanation,
    PIFUInferenceReport,
    PIFUModelPrediction,
    PIFUSafetyAssessment,
)

# FastPIFU Explanation Agent
from src.agents.pifu_inference_agent import create_pifu_explanation

# FastPIFU Safety Rules
from src.agents.pifu_safety import assess_pifu_safety

# FastPIFU Configuration
from src.config.pifu_settings import (
    PIFU_BASE_MODEL,
    PIFU_OUTPUT_DIR,
)

# Runtime Settings
from src.config.settings import settings

# FastPIFU Classifier
from src.inference.pifu_classifier import PIFUClassifier

# MLflow Experiment
PIFU_INFERENCE_EXPERIMENT = "nhs_omop_pifu_inference"


class PIFUInferenceState(
    TypedDict,
    total=False,
):
    """State for one PIFU inference run."""

    config: dict[str, Any]
    text: str
    prediction: dict[str, Any]
    safety: dict[str, Any]
    explanation: dict[str, Any]
    final_report: dict[str, Any]


def validate_input_node(
    state: PIFUInferenceState,
) -> dict[str, Any]:
    """Validate the clinic-letter input."""

    # Text Validation
    text = str(state["text"]).strip()

    if not text:
        raise ValueError("Clinic letter must not be empty.")

    return {
        "text": text,
    }


def text_sha256(text: str) -> str:
    """Return the SHA-256 hash of the clinic-letter text."""

    # Text Hashing
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def markdown_report(
    report: PIFUInferenceReport,
) -> str:
    """Create a human-readable inference report."""

    # Report Components
    prediction = report.prediction
    probabilities = prediction.probabilities

    # Report Header
    lines = [
        "# PIFU Inference Report",
        "",
        "## Prediction",
        "",
        f"**Class:** {prediction.predicted_class.value}",
        "",
        f"**Confidence:** {prediction.confidence:.4f}",
        "",
        "| Class | Probability |",
        "|---|---:|",
        f"| NOT_ELIGIBLE | {probabilities.not_eligible:.4f} |",
        f"| BORDERLINE | {probabilities.borderline:.4f} |",
        f"| ELIGIBLE | {probabilities.eligible:.4f} |",
        "",
        "## Human Review",
        "",
        "**Required: YES**",
        "",
        report.safety.review_reason,
        "",
        "## Explanation",
        "",
        report.explanation.clinical_summary,
        "",
    ]

    # Evidence Section
    if report.explanation.evidence_summary:
        lines.extend(
            [
                "### Evidence",
                "",
            ]
        )

        for item in report.explanation.evidence_summary:
            lines.append(f"- {item}")

        lines.append("")

    # Review Flags Section
    if report.safety.flags:
        lines.extend(
            [
                "### Review Flags",
                "",
            ]
        )

        for flag in report.safety.flags:
            lines.append(f"- {flag}")

        lines.append("")

    # Limitations Section
    if report.explanation.limitations:
        lines.extend(
            [
                "### Limitations",
                "",
            ]
        )

        for limitation in report.explanation.limitations:
            lines.append(f"- {limitation}")

        lines.append("")

    # Safety Footer
    lines.extend(
        [
            "---",
            "",
            "*Synthetic research system. Human clinical review is required.*",
        ]
    )

    return "\n".join(lines)


def validate_config(config: dict[str, Any]) -> None:
    """Validate required inference configuration keys."""

    # Required Config Check
    required_keys = {
        "run_id",
        "run_dir",
        "use_llm_explanation",
    }

    missing = [key for key in required_keys if key not in config]

    if missing:
        raise KeyError("Missing PIFU inference config key(s): " + ", ".join(missing))


def save_report_files(
    report: PIFUInferenceReport,
) -> None:
    """Save the JSON and Markdown inference reports."""

    # JSON Report Saving
    report.report_json.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Markdown Report Saving
    report.report_markdown.write_text(
        markdown_report(report),
        encoding="utf-8",
    )


def log_inference_to_mlflow(
    report: PIFUInferenceReport,
) -> None:
    """Log the inference report and scalar metrics to MLflow."""

    # MLflow Setup
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(PIFU_INFERENCE_EXPERIMENT)

    # MLflow Run
    with mlflow.start_run(
        run_name=f"pifu_inference_{report.run_id}",
        tags={
            "task": "pifu_inference",
            "research_use_only": "true",
            "human_review_required": "true",
        },
    ):
        # Parameter Logging
        mlflow.log_params(
            {
                "run_id": report.run_id,
                "model": report.model_name,
                "adapter": str(report.adapter_path),
                "predicted_class": report.prediction.predicted_class.value,
                "predicted_label": report.prediction.predicted_label,
                "human_review": "true",
                "text_sha256": report.text_sha256,
            }
        )

        # Metric Logging
        mlflow.log_metrics(
            {
                "confidence": report.prediction.confidence,
                "prob_not_eligible": report.prediction.probabilities.not_eligible,
                "prob_borderline": report.prediction.probabilities.borderline,
                "prob_eligible": report.prediction.probabilities.eligible,
                "top_two_margin": report.safety.top_two_margin,
            }
        )

        # Report Artifact Logging
        mlflow.log_artifact(
            str(report.report_json),
            artifact_path="report",
        )

        mlflow.log_artifact(
            str(report.report_markdown),
            artifact_path="report",
        )


def build_pifu_inference_graph():
    """Build the PIFU inference graph."""

    # Lazy Classifier Holder
    classifier: PIFUClassifier | None = None

    def classify_node(
        state: PIFUInferenceState,
    ) -> dict[str, Any]:
        """Run the fine-tuned classifier."""

        # Lazy Classifier Loading
        nonlocal classifier

        if classifier is None:
            classifier = PIFUClassifier()

        # Prediction
        result = classifier.predict(state["text"])

        return {
            "prediction": result.model_dict(),
        }

    def safety_node(
        state: PIFUInferenceState,
    ) -> dict[str, Any]:
        """Apply deterministic review rules."""

        # Prediction Validation
        prediction = PIFUModelPrediction.model_validate(state["prediction"])

        # Safety Assessment
        safety = assess_pifu_safety(prediction)

        return {
            "safety": safety.model_dump(mode="json"),
        }

    async def explanation_node(
        state: PIFUInferenceState,
    ) -> dict[str, Any]:
        """Create the narrative explanation."""

        # Evidence Validation
        prediction = PIFUModelPrediction.model_validate(state["prediction"])

        safety = PIFUSafetyAssessment.model_validate(state["safety"])

        # Explanation Generation
        explanation = await create_pifu_explanation(
            text=state["text"],
            prediction=prediction,
            safety=safety,
            use_llm=state["config"]["use_llm_explanation"],
        )

        return {
            "explanation": explanation.model_dump(mode="json"),
        }

    def save_node(
        state: PIFUInferenceState,
    ) -> dict[str, Any]:
        """Save JSON/Markdown reports and log MLflow artifacts."""

        # Config Validation
        config = state["config"]
        validate_config(config)

        # Run Directory Setup
        run_dir = Path(config["run_dir"])

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_json = run_dir / "prediction.json"
        report_markdown = run_dir / "prediction.md"

        # Component Validation
        prediction = PIFUModelPrediction.model_validate(state["prediction"])

        safety = PIFUSafetyAssessment.model_validate(state["safety"])

        explanation = PIFUExplanation.model_validate(state["explanation"])

        # Report Construction
        report = PIFUInferenceReport(
            run_id=config["run_id"],
            created_at=datetime.now(UTC),
            model_name=PIFU_BASE_MODEL,
            adapter_path=PIFU_OUTPUT_DIR.resolve(),
            text_sha256=text_sha256(state["text"]),
            prediction=prediction,
            safety=safety,
            explanation=explanation,
            report_json=report_json.resolve(),
            report_markdown=report_markdown.resolve(),
        )

        # Report Saving and Logging
        save_report_files(report)
        log_inference_to_mlflow(report)

        return {
            "final_report": report.model_dump(mode="json"),
        }

    # Graph Initialisation
    graph = StateGraph(PIFUInferenceState)

    # Graph Nodes
    graph.add_node(
        "validate_input",
        validate_input_node,
    )

    graph.add_node(
        "classify",
        classify_node,
    )

    graph.add_node(
        "safety_review",
        safety_node,
    )

    graph.add_node(
        "explain",
        explanation_node,
    )

    graph.add_node(
        "save",
        save_node,
    )

    # Graph Edges
    graph.add_edge(
        START,
        "validate_input",
    )

    graph.add_edge(
        "validate_input",
        "classify",
    )

    graph.add_edge(
        "classify",
        "safety_review",
    )

    graph.add_edge(
        "safety_review",
        "explain",
    )

    graph.add_edge(
        "explain",
        "save",
    )

    graph.add_edge(
        "save",
        END,
    )

    return graph.compile()
