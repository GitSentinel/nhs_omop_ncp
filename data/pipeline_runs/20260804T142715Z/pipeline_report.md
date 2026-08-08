# Agentic Pipeline Report — 20260804T142715Z

- Target: `original`
- Mode: `post-finetune`
- Started: `2026-08-04T14:27:15.601681+00:00`
- Finished: `2026-08-04T14:32:03.628301+00:00`
- Training GPUs: `[]`
- Evaluation GPU: `0`
- Overall status: `success`

## Executive summary

Synthetic NHS research pipeline assessment (post-finetune run) for the original binary treatment-event task. The run executed validate on the shared dataset (succeeded) and produced original_evaluate results; subsequent steps original_generate, original_relabel, original_finetune were skipped due to target=original and mode=post-finetune. PIFU (three-class eligibility) steps were also skipped for the same reason, but PIFU artifacts remain in the evidence for completeness.

Original task evaluation results (test split):
- Base model (original): f1=0.7831, macro_f1=0.4864, accuracy=0.6578, balanced_accuracy=0.5366, precision=0.6607, recall=0.9610, ROC AUC=0.6214, PR AUC=0.7948.
- Fine-tuned model (original): f1=0.9010, macro_f1=0.8397, accuracy=0.8631, balanced_accuracy=0.8209, precision=0.8420, recall=0.9688, ROC AUC=0.9564, PR AUC=0.9727.

Comparison: Base vs Fine-Tuned (delta values from the evidence):
- f1: 0.7831 -> 0.9010, delta +0.1179
- macro_f1: 0.4864 -> 0.8397, delta +0.3533
- accuracy: 0.6578 -> 0.8631, delta +0.2053
- balanced_accuracy: 0.5366 -> 0.8209, delta +0.2843
- precision: 0.6607 -> 0.8420, delta +0.1813
- recall: 0.9610 -> 0.9688, delta +0.0078
- ROC AUC: 0.6214 -> 0.9564, delta +0.3350
- PR AUC: 0.7948 -> 0.9727, delta +0.1779

PIFU (three-class eligibility) results in the evidence are present but were not executed in this run due to target=original mode=post-finetune; the reported PIFU metrics cover external_test, challenge, and test splits across base and fine-tuned configurations and should be interpreted as separate artifacts. For completeness, notable PIFU values include [external_test, fine_tuned] accuracy 0.9667 with not_eligible_recall 1.0 and eligible_recall 0.98, and [external_test, base] accuracy 0.3533 with not_eligible_recall 0.06 and eligible_recall 0.32, among others. See the evidence for the full multi-split metrics and confusion matrices.

Outputs are provided for synthetic research and require human review before any clinical interpretation or deployment.


## Pipeline steps

| Step | Dataset | Status | Duration (s) | Return code |
|---|---|---:|---:|---:|
| validate | shared | succeeded | 0.0 | 0 |
| original_generate | original | skipped | 0.0 | None |
| original_relabel | original | skipped | 0.0 | None |
| original_finetune | original | skipped | 0.0 | None |
| original_evaluate | original | succeeded | 253.0 | 0 |
| pifu_prepare | pifu | skipped | 0.0 | None |
| pifu_finetune | pifu | skipped | 0.0 | None |
| pifu_evaluate | pifu | skipped | 0.0 | None |

## Key findings

- Original task: Fine-tuned model achieves substantially higher F1, macro F1, accuracy, balanced accuracy, precision, ROC AUC and PR AUC compared with the base model on the test split.
- Delta highlights: F1 +0.1179, macro F1 +0.3533, accuracy +0.2053, balanced_accuracy +0.2843, precision +0.1813, recall +0.0078, ROC AUC +0.3350, PR AUC +0.1779.

## Safety flags

- Synthetic dataset; all conclusions require human review before any clinical use.

## Recommended actions

- Flag for manual review of model performance on treatment-event binary task.
- If clinical deployment is contemplated, re-run with updated hyperparameters and include external validation splits.
- Document all limitations and ensure no over-interpretation of synthetic data.
- Review PIFU artifacts separately as they pertain to a different 3-class eligibility task and were not executed in this run.

## Comparison

Base vs Fine-Tuned deltas: f1 +0.1179, macro_f1 +0.3533, accuracy +0.2053, balanced_accuracy +0.2843, precision +0.1813, recall +0.0078, roc_auc +0.3350, pr_auc +0.1779.

## Structured metric bundles

```json
[
  {
    "dataset": "pifu",
    "model": "fine_tuned",
    "split": "external_test",
    "metrics": {
      "accuracy": 0.9666666666666667,
      "balanced_accuracy": 0.9666666666666667,
      "macro_f1": 0.9668710918710919,
      "weighted_f1": 0.9668710918710918,
      "not_eligible_precision": 0.9090909090909091,
      "not_eligible_recall": 1.0,
      "borderline_precision": 1.0,
      "borderline_recall": 0.92,
      "eligible_precision": 1.0,
      "eligible_recall": 0.98,
      "unsafe_eligible_count": 0,
      "unsafe_eligible_rate": 0.0,
      "manual_review_rate": 0.30666666666666664,
      "prediction_counts": {
        "2": 49,
        "0": 55,
        "1": 46
      },
      "confusion_matrix": [
        [
          50,
          0,
          0
        ],
        [
          4,
          46,
          0
        ],
        [
          1,
          0,
          49
        ]
      ],
      "report": {
        "NOT_ELIGIBLE": {
          "precision": 0.9090909090909091,
          "recall": 1.0,
          "f1-score": 0.9523809523809523,
          "support": 50.0
        },
        "BORDERLINE": {
          "precision": 1.0,
          "recall": 0.92,
          "f1-score": 0.9583333333333334,
          "support": 50.0
        },
        "ELIGIBLE": {
          "precision": 1.0,
          "recall": 0.98,
          "f1-score": 0.98989898989899,
          "support": 50.0
        },
        "accuracy": 0.9666666666666667,
        "macro avg": {
          "precision": 0.9696969696969697,
          "recall": 0.9666666666666667,
          "f1-score": 0.9668710918710919,
          "support": 150.0
        },
        "weighted avg": {
          "precision": 0.9696969696969696,
          "recall": 0.9666666666666667,
          "f1-score": 0.9668710918710918,
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
      "macro_f1": 0.9469745336596932,
      "weighted_f1": 0.9469745336596932,
      "not_eligible_recall": 0.96,
      "borderline_recall": 0.92,
      "eligible_precision": 1.0,
      "eligible_recall": 0.96,
      "unsafe_eligible_count": 0,
      "unsafe_eligible_rate": 0.0,
      "manual_review_rate": 0.32666666666666666,
      "invalid_output_count": 1,
      "n_evaluated": 150,
      "confusion_matrix": [
        [
          48,
          2,
          0
        ],
        [
          4,
          46,
          0
        ],
        [
          1,
          1,
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
    "source": "/home/jacinth/nhs_omop_ncp/data/pipeline_runs/20260804T142715Z/original_evaluate.log"
  },
  {
    "dataset": "original",
    "model": "fine_tuned",
    "split": "test",
    "metrics": {
      "f1": 0.901,
      "macro_f1": 0.8397,
      "accuracy": 0.8631,
      "balanced_accuracy": 0.8209,
      "precision": 0.842,
      "recall": 0.9688,
      "roc_auc": 0.9564,
      "pr_auc": 0.9727
    },
    "source": "/home/jacinth/nhs_omop_ncp/data/pipeline_runs/20260804T142715Z/original_evaluate.log"
  }
]
```

> Synthetic research data only. Human clinical and methodological review is required.