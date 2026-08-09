# Agentic Model Pipeline Report

## Run Summary

| Item | Value |
|---|---|
| Status | ✅ SUCCESS |
| Target | `both` |
| Mode | `post-finetune` |
| Run ID | `20260808T235642Z` |
| Evaluation GPU | `0` |

## Original Clinic-Letter Model

| Metric | Base | Fine-tuned |
|---|---:|---:|
| F1 | 0.7831 | 0.8971 |
| Macro F1 | 0.4864 | 0.8386 |
| Accuracy | 0.6578 | 0.8598 |
| Balanced accuracy | 0.5366 | 0.8235 |
| Precision | 0.6607 | 0.8492 |
| Recall | 0.9610 | 0.9506 |
| ROC AUC | 0.6214 | 0.9547 |
| PR AUC | 0.7948 | 0.9728 |

## PIFU — External Test

| Metric | Base | Fine-tuned |
|---|---:|---:|
| Macro F1 | 0.2994 | 0.9468 |
| Accuracy | 0.3533 | 0.9467 |
| Balanced accuracy | 0.3533 | 0.9467 |
| NOT_ELIGIBLE recall | 0.0600 | 0.9600 |
| BORDERLINE recall | 0.6800 | 0.9200 |
| ELIGIBLE precision | 0.3077 | 0.9796 |
| ELIGIBLE recall | 0.3200 | 0.9600 |
| Unsafe eligible errors | 20 | 0 |
| Manual review rate | 0.6200 | 0.3200 |

## PIFU — Challenge Set

| Metric | Base | Fine-tuned |
|---|---:|---:|
| Macro F1 | 0.0635 | 0.6162 |
| Accuracy | 0.0833 | 0.7917 |
| Balanced accuracy | 0.0526 | 0.8684 |
| NOT_ELIGIBLE recall | 0.1053 | 0.7368 |
| BORDERLINE recall | 0.0000 | 0.0000 |
| ELIGIBLE precision | 0.0000 | 1.0000 |
| ELIGIBLE recall | 0.0000 | 1.0000 |
| Unsafe eligible errors | 7 | 0 |
| Manual review rate | 0.6250 | 0.2083 |

## Assessment

Original binary treatment-event task (post-finetune) achieves strong discrimination with ROC AUC 0.9547 and PR AUC 0.9728; recall 0.9506 and precision 0.8492, with F1 0.8971 and macro F1 0.8386, balanced accuracy 0.8235. For the three-class PIFU eligibility task, external-test shows NOT_ELIGIBLE recall 0.9600 and ELIGIBLE precision 0.9795918367346939, unsafe_eligible_count 0, manual_review_rate 0.32, and macro F1 0.9468, indicating robust performance but non-trivial manual review workload. These outputs are for synthetic research and require human review.

### Key Findings

- Original binary treatment-event post-finetune results: ROC AUC 0.9547, PR AUC 0.9728, recall 0.9506, precision 0.8492, F1 0.8971, macro F1 0.8386, balanced accuracy 0.8235.
- PIFU three-class external-test results: NOT_ELIGIBLE recall 0.9600, ELIGIBLE precision 0.9795918367346939, unsafe_eligible_count 0, manual_review_rate 0.32, macro F1 0.9468.
- Macro-level gains from fine-tuning: original macro F1 improved from 0.4864 (base) to 0.8386 (fine-tuned); ROC AUC improved to 0.9547 and PR AUC to 0.9728 in the post-finetune evaluation.

### Safety

- Manual review workload in PIFU external-test is non-trivial (manual_review_rate 0.32).
- Warning observed in evaluation logs: y_pred contains classes not in y_true, indicating potential calibration issues during base evaluation.
- Balanced accuracy around 0.8235 for the original task suggests sensitivity to class distribution; risk of reduced performance with distribution shifts.

### Recommended Actions

- Pursue additional external validation on synthetic datasets to confirm stability of original-task recall/precision and macro-F1 across domains; monitor ROC AUC and PR AUC with new data.
- Implement strategies to reduce manual_review_rate in PIFU (e.g., better calibration, threshold tuning, or ambiguous-borderline handling) while preserving ELIGIBLE precision.
- Investigate and address evaluation-time warnings (y_pred vs y_true class mismatch) and perform additional robustness checks before deployment; document the synthetic nature of results and plan human-in-the-loop review for real-world use.

## Pipeline Execution

| Step | Dataset | Status | Time (s) |
|---|---|---|---:|
| validate | shared | ✅ succeeded | 0.0 |
| original_evaluate | original | ✅ succeeded | 267.9 |
| pifu_evaluate | pifu | ✅ succeeded | 146.8 |

---

*Synthetic research data only. Human clinical review is required.*

Full metrics, confusion matrices, classification reports and provenance are available in `pipeline_report.json` and the evaluation artifacts.