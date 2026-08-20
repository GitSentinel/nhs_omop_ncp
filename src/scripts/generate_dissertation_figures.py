"""
Generate dissertation figures from the frozen final-results archive.

This script does not run any models. It only reads previously saved
evaluation outputs and creates publication-ready figures.

Example:
uv run python src/scripts/generate_dissertation_figures.py \sss
    --results-root results/dissertation_final_20260820T145106Z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# FastPIFU Class Names
CLASS_NAMES = [
    "NOT_ELIGIBLE",
    "BORDERLINE",
    "ELIGIBLE",
]

# Dissertation Figure Colours
BASE_MODEL_COLOR = "#0072B2"
FINE_TUNED_COLOR = "#D55E00"

JUDGE_COLORS = [
    "#0072B2",
    "#009E73",
    "#E69F00",
    "#CC79A7",
]

JUDGE_OUTCOME_COLORS = [
    "#009E73",
    "#D55E00",
    "#CC79A7",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    # CLI Parser Setup
    parser = argparse.ArgumentParser(
        description="Generate dissertation figures from frozen results."
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Path to the frozen dissertation results directory.",
    )

    return parser.parse_args()


def load_json(
    path: Path,
) -> dict:
    """Load one required JSON file."""

    # File Validation
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    # JSON Loading
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    filename: str,
) -> None:
    """Save one figure as PDF and high-resolution PNG."""

    # Output Paths
    pdf_path = output_dir / f"{filename}.pdf"
    png_path = output_dir / f"{filename}.png"

    # PDF Output
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    # High-Resolution PNG Output
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def add_bar_labels(
    axis: plt.Axes,
    bars,
    decimals: int = 3,
    offset: float = 0.015,
) -> None:
    """Add numerical labels above bars."""

    # Bar Annotations
    for bar in bars:
        height = float(bar.get_height())

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def metric_value(
    metrics: dict,
    key: str,
) -> float:
    """Return one required metric as a float."""

    # Metric Validation
    if key not in metrics:
        raise KeyError(f"Metric '{key}' not found. Available keys: {sorted(metrics)}")

    value = float(metrics[key])

    if not np.isfinite(value):
        raise ValueError(f"Metric '{key}' is not finite: {value}")

    return value


def figure_binary_model_comparison(
    tables_dir: Path,
    output_dir: Path,
) -> None:
    """Plot binary base versus fine-tuned model performance."""

    # Input File
    path = tables_dir / "binary_base_vs_finetuned.csv"

    if not path.exists():
        raise FileNotFoundError(f"Binary comparison file not found: {path}")

    data = pd.read_csv(path)

    # Required Columns
    required_columns = {
        "metric",
        "base",
        "fine_tuned",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}"
        )

    # Metric Display Order
    metric_specs = [
        (
            "f1",
            "F1",
        ),
        (
            "macro_f1",
            "Macro F1",
        ),
        (
            "accuracy",
            "Accuracy",
        ),
        (
            "balanced_accuracy",
            "Balanced\naccuracy",
        ),
        (
            "precision",
            "Precision",
        ),
        (
            "recall",
            "Recall",
        ),
        (
            "roc_auc",
            "ROC AUC",
        ),
        (
            "pr_auc",
            "PR AUC",
        ),
    ]

    data = data.set_index("metric")

    available_specs = [
        (
            metric,
            display,
        )
        for metric, display in metric_specs
        if metric in data.index
    ]

    if not available_specs:
        raise ValueError(f"No compatible binary metrics were found in {path}.")

    labels = [display for _, display in available_specs]

    base_values = [float(data.loc[metric, "base"]) for metric, _ in available_specs]

    fine_tuned_values = [
        float(data.loc[metric, "fine_tuned"]) for metric, _ in available_specs
    ]

    positions = np.arange(len(labels))

    width = 0.38

    # Figure Construction
    figure, axis = plt.subplots(
        figsize=(11, 6),
    )

    base_bars = axis.bar(
        positions - width / 2,
        base_values,
        width,
        label="Base model",
        color=BASE_MODEL_COLOR,
        edgecolor="black",
        linewidth=0.7,
    )

    fine_tuned_bars = axis.bar(
        positions + width / 2,
        fine_tuned_values,
        width,
        label="Fine-tuned model",
        color=FINE_TUNED_COLOR,
        edgecolor="black",
        linewidth=0.7,
    )

    # Axis Formatting
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.08)

    axis.set_xticks(positions)

    axis.set_xticklabels(labels)

    axis.legend(
        frameon=False,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    # Numerical Labels
    add_bar_labels(
        axis,
        base_bars,
    )

    add_bar_labels(
        axis,
        fine_tuned_bars,
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "figure_01_binary_base_vs_finetuned",
    )


def plot_pifu_comparison(
    base_metrics: dict,
    fine_tuned_metrics: dict,
    output_dir: Path,
    filename: str,
) -> None:
    """Plot base versus fine-tuned FastPIFU performance."""

    # Selected Dissertation Metrics
    metric_specs = [
        (
            "accuracy",
            "Accuracy",
        ),
        (
            "balanced_accuracy",
            "Balanced\naccuracy",
        ),
        (
            "macro_f1",
            "Macro F1",
        ),
        (
            "not_eligible_recall",
            "NOT_ELIGIBLE\nrecall",
        ),
        (
            "eligible_precision",
            "ELIGIBLE\nprecision",
        ),
    ]

    labels = [display for _, display in metric_specs]

    base_values = [
        metric_value(
            base_metrics,
            key,
        )
        for key, _ in metric_specs
    ]

    fine_tuned_values = [
        metric_value(
            fine_tuned_metrics,
            key,
        )
        for key, _ in metric_specs
    ]

    positions = np.arange(len(labels))

    width = 0.38

    # Figure Construction
    figure, axis = plt.subplots(
        figsize=(9, 6),
    )

    base_bars = axis.bar(
        positions - width / 2,
        base_values,
        width,
        label="Base model",
    )

    fine_tuned_bars = axis.bar(
        positions + width / 2,
        fine_tuned_values,
        width,
        label="Fine-tuned model",
    )

    # Axis Formatting
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.08)

    axis.set_xticks(positions)

    axis.set_xticklabels(labels)

    axis.legend(
        frameon=False,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    # Numerical Labels
    add_bar_labels(
        axis,
        base_bars,
    )

    add_bar_labels(
        axis,
        fine_tuned_bars,
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        filename,
    )


def figure_pifu_model_comparisons(
    tables_dir: Path,
    output_dir: Path,
) -> None:
    """Generate external-test and challenge FastPIFU comparisons."""

    # External-Test Metrics
    base_external = load_json(tables_dir / "base_external_test_metrics.json")

    fine_tuned_external = load_json(tables_dir / "finetuned_external_test_metrics.json")

    # Challenge-Set Metrics
    base_challenge = load_json(tables_dir / "base_challenge_metrics.json")

    fine_tuned_challenge = load_json(tables_dir / "finetuned_challenge_metrics.json")

    # External-Test Comparison
    plot_pifu_comparison(
        base_external,
        fine_tuned_external,
        output_dir,
        "figure_02_pifu_external_base_vs_finetuned",
    )

    # Challenge-Set Comparison
    plot_pifu_comparison(
        base_challenge,
        fine_tuned_challenge,
        output_dir,
        "figure_03_pifu_challenge_base_vs_finetuned",
    )


def load_confusion_matrix(
    path: Path,
) -> np.ndarray:
    """Load a 3x3 confusion matrix from CSV."""

    # File Validation
    if not path.exists():
        raise FileNotFoundError(f"Confusion matrix not found: {path}")

    # CSV Loading
    data = pd.read_csv(
        path,
        header=None,
    )

    matrix = data.to_numpy()

    # Handle Optional Row or Column Labels
    if matrix.shape != (3, 3):
        numeric = data.apply(
            pd.to_numeric,
            errors="coerce",
        )

        numeric = numeric.dropna(
            axis=0,
            how="all",
        )

        numeric = numeric.dropna(
            axis=1,
            how="all",
        )

        matrix = numeric.to_numpy()

    # Matrix Shape Validation
    if matrix.shape != (3, 3):
        raise ValueError(
            f"Expected 3x3 confusion matrix in {path}, found {matrix.shape}."
        )

    return matrix.astype(int)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_dir: Path,
    filename: str,
    cmap: str,
) -> None:
    """Plot one annotated FastPIFU confusion matrix."""

    # Figure Construction
    figure, axis = plt.subplots(
        figsize=(7, 6),
    )

    image = axis.imshow(
        matrix,
        cmap=cmap,
    )

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    # Axis Labels
    axis.set_xlabel("Predicted class")

    axis.set_ylabel("True class")

    axis.set_xticks(np.arange(len(CLASS_NAMES)))

    axis.set_yticks(np.arange(len(CLASS_NAMES)))

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=30,
        ha="right",
    )

    axis.set_yticklabels(
        CLASS_NAMES,
    )

    # Annotation Contrast Threshold
    threshold = matrix.max() / 2 if matrix.size else 0

    # Matrix Annotations
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(
                matrix[
                    row,
                    column,
                ]
            )

            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=("white" if value > threshold else "black"),
                fontsize=12,
            )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        filename,
    )


def figure_pifu_confusion_matrices(
    tables_dir: Path,
    output_dir: Path,
) -> None:
    """Generate fine-tuned FastPIFU confusion matrices."""

    # External-Test Matrix
    external_matrix = load_confusion_matrix(
        tables_dir / "finetuned_external_test_confusion_matrix.csv"
    )

    # Challenge-Set Matrix
    challenge_matrix = load_confusion_matrix(
        tables_dir / "finetuned_challenge_confusion_matrix.csv"
    )

    plot_confusion_matrix(
        external_matrix,
        output_dir,
        "figure_04_pifu_external_finetuned_confusion_matrix",
        cmap="Blues",
    )

    plot_confusion_matrix(
        challenge_matrix,
        output_dir,
        "figure_05_pifu_challenge_finetuned_confusion_matrix",
        cmap="Oranges",
    )


def figure_judge_quality_scores(
    tables_dir: Path,
    output_dir: Path,
) -> None:
    """Plot mean LLM-as-a-Judge quality scores."""

    # Judge Metrics
    metrics = load_json(tables_dir / "llm_judge_summary.json")

    metric_specs = [
        (
            "mean_judge_faithfulness",
            "Faithfulness",
        ),
        (
            "mean_judge_grounding",
            "Grounding",
        ),
        (
            "mean_judge_prediction_consistency",
            "Prediction\nconsistency",
        ),
        (
            "mean_judge_safety",
            "Safety",
        ),
    ]

    labels = [display for _, display in metric_specs]

    values = [
        metric_value(
            metrics,
            key,
        )
        for key, _ in metric_specs
    ]

    positions = np.arange(len(labels))

    # Figure Construction
    figure, axis = plt.subplots(
        figsize=(8, 6),
    )

    bars = axis.bar(
        positions,
        values,
        width=0.6,
        color=JUDGE_COLORS,
        edgecolor="black",
        linewidth=0.7,
    )

    # Axis Formatting
    axis.set_ylabel("Mean judge score")

    axis.set_ylim(
        0,
        5.4,
    )

    axis.set_xticks(positions)

    axis.set_xticklabels(labels)

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    # Numerical Labels
    add_bar_labels(
        axis,
        bars,
        decimals=2,
        offset=0.08,
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "figure_06_llm_judge_quality_scores",
    )


def figure_judge_outcomes(
    tables_dir: Path,
    output_dir: Path,
) -> bool:
    """Plot judge pass, hallucination and unsupported-claim rates."""

    # Judge Metrics
    metrics = load_json(tables_dir / "llm_judge_summary.json")

    possible_metrics = [
        (
            "judge_pass_rate",
            "Pass rate",
        ),
        (
            "judge_hallucination_rate",
            "Hallucination\nrate",
        ),
        (
            "unsupported_claim_rate",
            "Unsupported-\nclaim rate",
        ),
    ]

    # Available Metrics
    available = [
        (
            key,
            label,
        )
        for key, label in possible_metrics
        if (key in metrics and metrics[key] is not None)
    ]

    if not available:
        print("Skipping judge outcome figure: no compatible rate metrics found.")
        return False

    labels = [label for _, label in available]

    values = [
        metric_value(
            metrics,
            key,
        )
        for key, _ in available
    ]

    colors = JUDGE_OUTCOME_COLORS[: len(values)]
    positions = np.arange(len(labels))

    # Figure Construction
    figure, axis = plt.subplots(
        figsize=(7, 6),
    )

    bars = axis.bar(
        positions,
        values,
        width=0.6,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )

    # Axis Formatting
    axis.set_ylabel("Proportion")

    axis.set_ylim(
        0,
        1.08,
    )

    axis.set_xticks(positions)

    axis.set_xticklabels(labels)

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    # Numerical Labels
    add_bar_labels(
        axis,
        bars,
        decimals=3,
        offset=0.02,
    )

    figure.tight_layout()

    save_figure(
        figure,
        output_dir,
        "figure_07_llm_judge_outcomes",
    )

    return True


def write_figure_manifest(
    output_dir: Path,
) -> None:
    """Write a manifest describing successfully generated figures."""

    # Figure Descriptions
    descriptions = {
        "figure_01_binary_base_vs_finetuned": (
            "Binary RTT base versus fine-tuned model performance."
        ),
        "figure_02_pifu_external_base_vs_finetuned": (
            "PIFU external-test base versus fine-tuned performance."
        ),
        "figure_03_pifu_challenge_base_vs_finetuned": (
            "PIFU challenge-set base versus fine-tuned performance."
        ),
        "figure_04_pifu_external_finetuned_confusion_matrix": (
            "Fine-tuned PIFU external-test confusion matrix."
        ),
        "figure_05_pifu_challenge_finetuned_confusion_matrix": (
            "Fine-tuned PIFU challenge-set confusion matrix."
        ),
        "figure_06_llm_judge_quality_scores": (
            "Mean LLM-as-a-Judge explanation-quality scores."
        ),
        "figure_07_llm_judge_outcomes": (
            "LLM-as-a-Judge pass and flagged-outcome rates."
        ),
    }

    # Manifest Construction
    rows = []

    for figure_name, description in descriptions.items():
        pdf_path = output_dir / f"{figure_name}.pdf"
        png_path = output_dir / f"{figure_name}.png"

        if not pdf_path.exists() and not png_path.exists():
            continue

        rows.append(
            {
                "figure": figure_name,
                "description": description,
                "pdf": (pdf_path.name if pdf_path.exists() else None),
                "png": (png_path.name if png_path.exists() else None),
            }
        )

    # Manifest Saving
    manifest = pd.DataFrame(rows)

    path = output_dir / "figure_manifest.csv"

    manifest.to_csv(
        path,
        index=False,
    )

    print(f"Saved: {path}")


def main() -> None:
    """Generate all dissertation figures."""

    # CLI Setup
    args = parse_args()

    # Results Directory Resolution
    results_root = args.results_root.expanduser().resolve()

    if not results_root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")

    tables_dir = results_root / "04_dissertation_tables"

    if not tables_dir.exists():
        raise FileNotFoundError(
            f"Dissertation tables directory does not exist: {tables_dir}"
        )

    # Figure Output Directory
    output_dir = results_root / "05_dissertation_figures"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Console Header
    print("=" * 68)
    print("GENERATING DISSERTATION FIGURES")
    print("=" * 68)
    print(f"Results root : {results_root}")
    print(f"Input tables : {tables_dir}")
    print(f"Output       : {output_dir}")
    print()

    # Binary RTT Comparison
    figure_binary_model_comparison(
        tables_dir,
        output_dir,
    )

    # FastPIFU Model Comparisons
    figure_pifu_model_comparisons(
        tables_dir,
        output_dir,
    )

    # FastPIFU Confusion Matrices
    figure_pifu_confusion_matrices(
        tables_dir,
        output_dir,
    )

    # LLM-as-a-Judge Quality Scores
    figure_judge_quality_scores(
        tables_dir,
        output_dir,
    )

    # LLM-as-a-Judge Outcome Rates
    figure_judge_outcomes(
        tables_dir,
        output_dir,
    )

    # Figure Manifest
    write_figure_manifest(output_dir)

    print()
    print("=" * 68)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 68)


if __name__ == "__main__":
    main()
