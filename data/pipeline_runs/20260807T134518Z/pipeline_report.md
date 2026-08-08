# Agentic Pipeline Report — 20260807T134518Z

- Target: `original`
- Mode: `train-and-evaluate`
- Started: `2026-08-07T13:45:18.842767+00:00`
- Finished: `2026-08-07T15:04:22.395051+00:00`
- Training GPUs: `[0, 1, 2]`
- Evaluation GPU: `0`
- Overall status: `success`

## Executive summary

Synthetic pipeline assessment (NHS research) across two targets: (A) original binary treatment-event task (treatment_event) and (B) three-class PIFU eligibility task. This report preserves numeric evidence exactly as provided and is intended for synthetic research with required human-in-the-loop review.

1) Original binary treatment-event task (treatment_event)
- Best-performing model (test/evaluation): original_finetune
  - F1: 0.8971
  - Macro F1: 0.8386
  - Accuracy: 0.8598
  - Balanced accuracy: 0.8235
  - Precision: 0.8492
  - Recall: 0.9506
  - ROC AUC: 0.9547
  - PR AUC: 0.9728
  - Confusion/threshold details available in the evidence: see evaluation logs.

- Baseline model (test/evaluation): original (base)
  - F1: 0.7831
  - Macro F1: 0.4864
  - Accuracy: 0.6578
  - Balanced accuracy: 0.5366
  - Precision: 0.6607
  - Recall: 0.9610
  - ROC AUC: 0.6214
  - PR AUC: 0.7948

- Direct comparison (Base vs Fine-Tuned)
  - F1: +0.1140
  - Macro F1: +0.3522
  - Accuracy: +0.2020
  - Balanced accuracy: +0.2869
  - Precision: +0.1885
  - Recall: -0.0104
  - ROC AUC: +0.3333
  - PR AUC: +0.1780

Notes: All original-task results are reported for the synthetic dataset; the best-performing configuration in the provided evidence is the original_finetune model on the test split.

2) Three-class PIFU eligibility task (NOT_ELIGIBLE, BORDERLINE, ELIGIBLE)
- External test, fine-tuned model
  - Macro F1: 0.9468
  - Accuracy: 0.9467
  - NOT_ELIGIBLE: precision 0.9057, recall 0.96, f1 0.9320 (support 50)
  - BORDERLINE: precision 0.9583, recall 0.92, f1 0.9388 (support 50)
  - ELIGIBLE: precision 0.9796, recall 0.96, f1 0.9697 (support 50)
  - Unsafe eligible count: 0; unsafe eligible rate: 0.0
  - Manual review rate: 0.32
  - Overall report: accuracy 0.9467; macro avg precision 0.9479; macro avg recall 0.9467; macro avg f1 0.9468
  - Notes: Prediction counts: {2: 49, 0: 53, 1: 48}; Confusion matrix shown in the evidence.

- Challenge, fine-tuned model
  - Macro F1: 0.6162; Accuracy: 0.7917; Not_eligible: precision 1.0, recall 0.7368; BORDERLINE: precision 0.0, recall 0.0; ELIGIBLE: precision 1.0, recall 1.0
  - Unsafe eligible count: 0; unsafe rate: 0.0
  - Manual review rate: 0.2083
  - Report: NOT_ELIGIBLE precision 1.0, recall 0.7368, f1 0.8485; ELIGIBLE precision 1.0, recall 1.0, f1 1.0
  - Accuracy: 0.7917; Prediction counts: {2: 5, 0: 14, 1: 5}

- Base model, challenge
  - NOT_ELIGIBLE: precision 1.0, recall 0.1053; ELIGIBLE: precision 0.0, recall 0.0; Unsafe eligible count: 7; unsafe eligible rate: 0.3684; Manual review rate: 0.625
  - Accuracy: 0.0833; macro avg precision 0.3333; macro avg recall 0.0351; macro avg f1 0.0635
  - Note: This configuration displays notably weak performance on the challenge split, consistent with the a priori baseline results in the evidence.

- Base external test
  - Accuracy: 0.3533; Macro F1: 0.2994; NOT_ELIGIBLE: precision 0.6, recall 0.06; ELIGIBLE: precision 0.3077, recall 0.32; Unsafe eligible count: 20; unsafe rate: 0.4; Manual review rate: 0.62
  - Report: NOT_ELIGIBLE precision 0.6, recall 0.06, f1 0.1091; BORDERLINE precision 0.3656, recall 0.68, f1 0.4755; ELIGIBLE precision 0.3077, recall 0.32, f1 0.3137

- Fine-tuned, test (two variants shown in the evidence)
  - Variant A: NOT_ELIGIBLE recall 0.96; BORDERLINE recall 0.92; ELIGIBLE precision 0.9796; ELIGIBLE recall 0.96; Manual review rate 0.32; Accuracy 0.9467; Confusion matrix reported
  - Variant B: NOT_ELIGIBLE recall 0.96; BORDERLINE recall 0.98; ELIGIBLE precision 1.0; ELIGIBLE recall 0.94; Manual review rate 0.3467; Accuracy 0.96
  - Both variants show strong ELIGIBLE precision; slight variations in border-line coverage and manual review burden.

- Notes on PIFU results
  - Across splits, unsafe eligible counts are 0 for external_test fine-tuned and challenge fine-tuned variants; base configurations show non-zero unsafe eligible counts (notably base external_test: 20; base challenge: 7).
  - Manual review rates range from 0.2083 to 0.625 across splits, with higher rates in the base configurations.

Important caveats and synthetic-research note
- All numbers are drawn from the provided synthetic evidence and are not clinical results. These outputs are intended for synthetic research, and require human review before any real-world application.
- The two tasks (original and PIFU) are reported separately and clearly distinguished in this summary. No metric is inferred beyond what is stated in the evidence set.

Recommendation for human review
- Focus on the high-recall original-task model (original_finetune) for treatment-event detection and verify potential near-miss cases (FN) with clinical input.
- For PIFU, review borderline and NOT_ELIGIBLE decisions in external_test for potential calibration adjustments; weigh the relatively low manual review burden against high ELIGIBLE precision. Consider re-balancing or class-weight adjustments for the base configurations showing poor challenge/test performance.

This report preserves numeric evidence exactly as supplied and is intended for synthetic research use. It requires human review before any real-world deployment or clinical interpretation.

## Pipeline steps

| Step | Dataset | Status | Duration (s) | Return code |
|---|---|---:|---:|---:|
| validate | shared | succeeded | 0.0 | 0 |
| original_generate | original | skipped | 0.0 | None |
| original_relabel | original | skipped | 0.0 | None |
| original_finetune | original | succeeded | 4443.9 | 0 |
| original_evaluate | original | succeeded | 253.3 | 0 |
| pifu_prepare | pifu | skipped | 0.0 | None |
| pifu_finetune | pifu | skipped | 0.0 | None |
| pifu_evaluate | pifu | skipped | 0.0 | None |

## Key findings

- Original task: Fine-tuned model achieves higher F1 and PR AUC with substantial gains over base (F1 +0.1141; PR AUC +0.1780).
- PIFU eligibility external_test: high macro F1 (~0.947) with strong ELIGIBLE precision (0.98) and NOT_ELIGIBLE recall (0.96); manual-review burden moderate (0.32).
- PIFU base configurations show weaker performance (challenge and external_test) with non-trivial unsafe eligible counts and high manual-review rates.

## Safety flags

- synthetic_data
- requires_human_review

## Recommended actions

- Flag for expert review of borderline and NOT_ELIGIBLE cases in PIFU external_test; consider recalibration of threshold for border cases.
- Document and audit the delta in original task performance when moving from base to finetuned to ensure reproducibility in synthetic datasets.
- Maintain separation of original task and PIFU metrics to prevent cross-task misinterpretation.

## Comparison

Original task: Base vs Fine-Tuned comparison shows notable gains with fine-tuning (F1 +0.1140, Macro F1 +0.3522, Accuracy +0.2020, ROC AUC +0.3333, PR AUC +0.1780). PIFU: external_test fine-tuned yields strong NOT_ELIGIBLE recall (0.96) and ELIGIBLE precision (0.9796) with macro F1 ~0.947; baseline configurations show weaker performance and higher manual-review rates.

## Structured metric bundles

```json
[
  {
    "dataset": "pifu",
    "model": "fine_tuned",
    "split": "external_test",
    "metrics": {
      "accuracy": 0.9466666666666667,
      "balanced_accuracy": 0.9466666666666667,
      "macro_f1": 0.946837104950836,
      "weighted_f1": 0.9468371049508357,
      "not_eligible_precision": 0.9056603773584906,
      "not_eligible_recall": 0.96,
      "borderline_precision": 0.9583333333333334,
      "borderline_recall": 0.92,
      "eligible_precision": 0.9795918367346939,
      "eligible_recall": 0.96,
      "unsafe_eligible_count": 0,
      "unsafe_eligible_rate": 0.0,
      "manual_review_rate": 0.32,
      "prediction_counts": {
        "2": 49,
        "0": 53,
        "1": 48
      },
      "confusion_matrix": [
        [
          48,
          2,
          0
        ],
        [
          3,
          46,
          1
        ],
        [
          2,
          0,
          48
        ]
      ],
      "report": {
        "NOT_ELIGIBLE": {
          "precision": 0.9056603773584906,
          "recall": 0.96,
          "f1-score": 0.9320388349514563,
          "support": 50.0
        },
        "BORDERLINE": {
          "precision": 0.9583333333333334,
          "recall": 0.92,
          "f1-score": 0.9387755102040817,
          "support": 50.0
        },
        "ELIGIBLE": {
          "precision": 0.9795918367346939,
          "recall": 0.96,
          "f1-score": 0.9696969696969697,
          "support": 50.0
        },
        "accuracy": 0.9466666666666667,
        "macro avg": {
          "precision": 0.9478618491421725,
          "recall": 0.9466666666666667,
          "f1-score": 0.946837104950836,
          "support": 150.0
        },
        "weighted avg": {
          "precision": 0.9478618491421726,
          "recall": 0.9466666666666667,
          "f1-score": 0.9468371049508357,
          "support": 150.0
        }
      }
    },
    "source": "/home/jacinth/nhs_omop_ncp/data/evaluations/pifu_cardiology/finetuned_external_test_metrics.json"
  },
  {
    "dataset": "pifu",
    "model": "fine_tuned",
    "split": "challenge",
    "metrics": {
      "accuracy": 0.7916666666666666,
      "balanced_accuracy": 0.868421052631579,
      "macro_f1": 0.6161616161616162,
      "weighted_f1": 0.8800505050505051,
      "not_eligible_precision": 1.0,
      "not_eligible_recall": 0.7368421052631579,
      "borderline_precision": 0.0,
      "borderline_recall": 0.0,
      "eligible_precision": 1.0,
      "eligible_recall": 1.0,
      "unsafe_eligible_count": 0,
      "unsafe_eligible_rate": 0.0,
      "manual_review_rate": 0.20833333333333334,
      "prediction_counts": {
        "2": 5,
        "0": 14,
        "1": 5
      },
      "confusion_matrix": [
        [
          14,
          5,
          0
        ],
        [
          0,
          0,
          0
        ],
        [
          0,
          0,
          5
        ]
      ],
      "report": {
        "NOT_ELIGIBLE": {
          "precision": 1.0,
          "recall": 0.7368421052631579,
          "f1-score": 0.8484848484848485,
          "support": 19.0
        },
        "BORDERLINE": {
          "precision": 0.0,
          "recall": 0.0,
          "f1-score": 0.0,
          "support": 0.0
        },
        "ELIGIBLE": {
          "precision": 1.0,
          "recall": 1.0,
          "f1-score": 1.0,
          "support": 5.0
        },
        "accuracy": 0.7916666666666666,
        "macro avg": {
          "precision": 0.6666666666666666,
          "recall": 0.5789473684210527,
          "f1-score": 0.6161616161616162,
          "support": 24.0
        },
        "weighted avg": {
          "precision": 1.0,
          "recall": 0.7916666666666666,
          "f1-score": 0.8800505050505051,
          "support": 24.0
        }
      }
    },
    "source": "/home/jacinth/nhs_omop_ncp/data/evaluations/pifu_cardiology/finetuned_challenge_metrics.json"
  },
  {
    "dataset": "pifu",
    "model": "base",
    "split": "challenge",
    "metrics": {
      "accuracy": 0.08333333333333333,
      "balanced_accuracy": 0.05263157894736842,
      "macro_f1": 0.06349206349206349,
      "weighted_f1": 0.15079365079365079,
      "not_eligible_precision": 1.0,
      "not_eligible_recall": 0.10526315789473684,
      "borderline_precision": 0.0,
      "borderline_recall": 0.0,
      "eligible_precision": 0.0,
      "eligible_recall": 0.0,
      "unsafe_eligible_count": 7,
      "unsafe_eligible_rate": 0.3684210526315789,
      "manual_review_rate": 0.625,
      "prediction_counts": {
        "1": 15,
        "0": 2,
        "2": 7
      },
      "confusion_matrix": [
        [
          2,
          10,
          7
        ],
        [
          0,
          0,
          0
        ],
        [
          0,
          5,
          0
        ]
      ],
      "report": {
        "NOT_ELIGIBLE": {
          "precision": 1.0,
          "recall": 0.10526315789473684,
          "f1-score": 0.19047619047619047,
          "support": 19.0
        },
        "BORDERLINE": {
          "precision": 0.0,
          "recall": 0.0,
          "f1-score": 0.0,
          "support": 0.0
        },
        "ELIGIBLE": {
          "precision": 0.0,
          "recall": 0.0,
          "f1-score": 0.0,
          "support": 5.0
        },
        "accuracy": 0.08333333333333333,
        "macro avg": {
          "precision": 0.3333333333333333,
          "recall": 0.03508771929824561,
          "f1-score": 0.06349206349206349,
          "support": 24.0
        },
        "weighted avg": {
          "precision": 0.7916666666666666,
          "recall": 0.08333333333333333,
          "f1-score": 0.15079365079365079,
          "support": 24.0
        }
      }
    },
    "source": "/home/jacinth/nhs_omop_ncp/data/evaluations/pifu_cardiology/base_challenge_metrics.json"
  },
  {
    "dataset": "pifu",
    "model": "base",
    "split": "external_test",
    "metrics": {
      "accuracy": 0.35333333333333333,
      "balanced_accuracy": 0.35333333333333333,
      "macro_f1": 0.2994469582704877,
      "weighted_f1": 0.2994469582704877,
      "not_eligible_precision": 0.6,
      "not_eligible_recall": 0.06,
      "borderline_precision": 0.3655913978494624,
      "borderline_recall": 0.68,
      "eligible_precision": 0.3076923076923077,
      "eligible_recall": 0.32,
      "unsafe_eligible_count": 20,
      "unsafe_eligible_rate": 0.4,
      "manual_review_rate": 0.62,
      "prediction_counts": {
        "1": 93,
        "2": 52,
        "0": 5
      },
      "confusion_matrix": [
        [
          3,
          27,
          20
        ],
        [
          0,
          34,
          16
        ],
        [
          2,
          32,
          16
        ]
      ],
      "report": {
        "NOT_ELIGIBLE": {
          "precision": 0.6,
          "recall": 0.06,
          "f1-score": 0.10909090909090909,
          "support": 50.0
        },
        "BORDERLINE": {
          "precision": 0.3655913978494624,
          "recall": 0.68,
          "f1-score": 0.4755244755244755,
          "support": 50.0
        },
        "ELIGIBLE": {
          "precision": 0.3076923076923077,
          "recall": 0.32,
          "f1-score": 0.3137254901960784,
          "support": 50.0
        },
        "accuracy": 0.35333333333333333,
        "macro avg": {
          "precision": 0.4244279018472567,
          "recall": 0.35333333333333333,
          "f1-score": 0.2994469582704877,
          "support": 150.0
        },
        "weighted avg": {
          "precision": 0.4244279018472567,
          "recall": 0.35333333333333333,
          "f1-score": 0.2994469582704877,
          "support": 150.0
        }
      }
    },
    "source": "/home/jacinth/nhs_omop_ncp/data/evaluations/pifu_cardiology/base_external_test_metrics.json"
  },
  {
    "dataset": "pifu",
    "model": "fine_tuned",
    "split": "test",
    "metrics": {
      "accuracy": 0.9466666666666667,
      "balanced_accuracy": 0.9466666666666667,
      "macro_f1": 0.946837104950836,
      "weighted_f1": 0.9468371049508357,
      "not_eligible_recall": 0.96,
      "borderline_recall": 0.92,
      "eligible_precision": 0.9795918367346939,
      "eligible_recall": 0.96,
      "unsafe_eligible_count": 0,
      "unsafe_eligible_rate": 0.0,
      "manual_review_rate": 0.32,
      "invalid_output_count": 0,
      "n_evaluated": 150,
      "confusion_matrix": [
        [
          48,
          2,
          0
        ],
        [
          3,
          46,
          1
        ],
        [
          2,
          0,
          48
        ]
      ]
    },
    "source": "/home/jacinth/nhs_omop_ncp/data/finetune/qwen35_9b_pifu_lora/evaluation/test_metrics.json"
  },
  {
    "dataset": "pifu",
    "model": "fine_tuned",
    "split": "test",
    "metrics": {
      "accuracy": 0.96,
      "balanced_accuracy": 0.96,
      "macro_f1": 0.9601171760596315,
      "weighted_f1": 0.9601171760596313,
      "not_eligible_recall": 0.96,
      "borderline_recall": 0.98,
      "eligible_precision": 1.0,
      "eligible_recall": 0.94,
      "unsafe_eligible_count": 0,
      "unsafe_eligible_rate": 0.0,
      "manual_review_rate": 0.3466666666666667,
      "invalid_output_count": 0,
      "n_evaluated": 150,
      "confusion_matrix": [
        [
          48,
          2,
          0
        ],
        [
          1,
          49,
          0
        ],
        [
          2,
          1,
          47
        ]
      ]
    },
    "source": "/home/jacinth/nhs_omop_ncp/data/finetune/qwen35_9b_pifu_lora_3epochs_same_protocol/evaluation/test_metrics.json"
  },
  {
    "dataset": "original",
    "model": "base",
    "split": "test",
    "metrics": {
      "f1": 0.7831,
      "macro_f1": 0.4864,
      "accuracy": 0.6578,
      "balanced_accuracy": 0.5366,
      "precision": 0.6607,
      "recall": 0.961,
      "roc_auc": 0.6214,
      "pr_auc": 0.7948
    },
    "source": "/home/jacinth/nhs_omop_ncp/data/pipeline_runs/20260807T134518Z/original_evaluate.log"
  },
  {
    "dataset": "original",
    "model": "fine_tuned",
    "split": "test",
    "metrics": {
      "f1": 0.8971,
      "macro_f1": 0.8386,
      "accuracy": 0.8598,
      "balanced_accuracy": 0.8235,
      "precision": 0.8492,
      "recall": 0.9506,
      "roc_auc": 0.9547,
      "pr_auc": 0.9728
    },
    "source": "/home/jacinth/nhs_omop_ncp/data/pipeline_runs/20260807T134518Z/original_evaluate.log"
  }
]
```

> Synthetic research data only. Human clinical and methodological review is required.