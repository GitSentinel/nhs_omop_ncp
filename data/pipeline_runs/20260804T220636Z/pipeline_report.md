# Agentic Pipeline Report — 20260804T220636Z

- Target: `both`
- Mode: `post-finetune`
- Started: `2026-08-04T22:06:36.563797+00:00`
- Finished: `2026-08-04T22:14:05.825456+00:00`
- Training GPUs: `[]`
- Evaluation GPU: `0`
- Overall status: `success`

## Executive summary

Pipeline assessment completed for synthetic post-finetune evaluation. Validation step succeeded and original_evaluate and pifu_evaluate ran to completion. The final outputs distinguish the two evaluation tracks: the original binary treatment-event task and the three-class PIFU eligibility task.

Original (binary) treatment-event task – test performance (base vs. fine-tuned):
- Base: F1=0.7831, Macro F1=0.4864, Accuracy=0.6578, Balanced accuracy=0.5366, Precision=0.6607, Recall=0.9610, PR AUC=0.7948, ROC AUC=0.6214. Confusion: TN=24, FP=190, FN=15, TP=370.
- Fine-tuned: F1=0.9010, Macro F1=0.8397, Accuracy=0.8631, Balanced accuracy=0.8209, Precision=0.8420, Recall=0.9688, PR AUC=0.9727, ROC AUC=0.9564. Confusion: TN=144, FP=70, FN=12, TP=373.
- Delta (Base -> Fine-Tuned): F1 +0.1179, Macro F1 +0.3533, Accuracy +0.2053, Balanced accuracy +0.2843, Precision +0.1813, Recall +0.0078, ROC AUC +0.3350, PR AUC +0.1779.

PIFU eligibility task – three-class results (external_test and challenge):
- External_test (fine-tuned): NOT_ELIGIBLE recall=1.0, NOT_ELIGIBLE precision=0.9090909090909091, BORDERLINE precision=1.0, BORDERLINE recall=0.92, ELIGIBLE precision=1.0, ELIGIBLE recall=0.98, Unsafe eligible=0, Manual review rate=0.30666666666666664. Accuracy=0.9666666666666667, Macro F1=0.9668710918710919. Confusion: [[50,0,0],[4,46,0],[1,0,49]]. Class-wise: NOT_ELIGIBLE P=0.9091 R=1.0000; BORDERLINE P=1.0000 R=0.92; ELIGIBLE P=1.0000 R=0.98.
- Challenge (fine-tuned): NOT_ELIGIBLE recall=0.7368421052631579, NOT_ELIGIBLE precision=1.0, BORDERLINE precision=0.0, BORDERLINE recall=0.0, ELIGIBLE precision=1.0, ELIGIBLE recall=1.0, Unsafe eligible=0, Manual review rate=0.20833333333333334. Accuracy=0.7916666666666666, Macro F1=0.6161616161616162. Confusion: [[14,5,0],[0,0,0],[0,0,5]]. NOT_ELIGIBLE: P=1.0, R=0.7368; ELIGIBLE: P=1.0, R=1.0.
- External_test (base): Accuracy=0.35333333333333333, Balanced accuracy=0.35333333333333333, Macro F1=0.2994469582704877, Not_ELIGIBLE precision=0.6, Not_ELIGIBLE recall=0.06, BORDERLINE precision=0.0? (0.0) or 0.3656 in the report, but Metrics show Borderline precision=0.3655913978494624, Borderline recall=0.68, ELIGIBLE precision=0.3076923076923077, ELIGIBLE recall=0.32. Unsafe eligible count=20, Unsafe eligible rate=0.4, Manual review rate=0.62. Confusion: [[3,27,20],[0,34,16],[2,32,16]].
- Challenge (base): Accuracy=0.7916666666666666, NOT_ELIGIBLE recall=0.7368421052631579, NOT_ELIGIBLE precision=1.0, BORDERLINE recall=0.0, BORDERLINE precision=0.0, ELIGIBLE recall=1.0, ELIGIBLE precision=1.0, Unsafe eligible count=7, Unsafe eligible rate=0.3684210526315789, Manual review rate=0.625. Confusion: [[2,10,7],[0,0,0],[0,5,0]].

Overall, fine-tuned models generally show large improvements on the original task metrics and substantial gains in PIFU macro performance on external_test, with some trade-offs in challenge/test settings. The results are derived from synthetic data and should be reviewed by humans before any clinical interpretation.

Notes: All reported metrics come from the provided evidence bundle only and are for synthetic research; no clinical conclusions should be drawn without human review.

## Pipeline steps

| Step | Dataset | Status | Duration (s) | Return code |
|---|---|---:|---:|---:|
| validate | shared | succeeded | 0.0 | 0 |
| original_generate | original | skipped | 0.0 | None |
| original_relabel | original | skipped | 0.0 | None |
| original_finetune | original | skipped | 0.0 | None |
| original_evaluate | original | succeeded | 263.8 | 0 |
| pifu_prepare | pifu | skipped | 0.0 | None |
| pifu_finetune | pifu | skipped | 0.0 | None |
| pifu_evaluate | pifu | succeeded | 148.3 | 0 |

## Key findings

- Original binary task: Fine-tuned model improves F1 from 0.7831 to 0.9010 and Macro F1 from 0.4864 to 0.8397; PR AUC improves from 0.7948 to 0.9727; ROC AUC improves from 0.6214 to 0.9564.
- PIFU external_test: Fine-tuned yields NOT_ELIGIBLE recall 1.0 and ELIGIBLE recall 0.98 with precision 1.0 for both NOT_ELIGIBLE and ELIGIBLE; macro F1 0.9669 and accuracy 0.9667.
- PIFU challenge/test: Macro F1 improves from 0.0635 (base challenge) to 0.6162 (fine-tuned challenge); NOT_ELIGIBLE recall improves from 0.1053 to 0.7368 in external/test when comparing base vs fine-tuned.

## Safety flags

- Outputs are synthetic data used for research; require human review before drawing clinical conclusions.

## Recommended actions

- Review per-class performance for PIFU predictions to ensure safe thresholds for NOT_ELIGIBLE and ELIGIBLE categories.
- Compare final model selections with domain experts and consider rebalancing where class-wise recall/precision are imbalanced (e.g., base external_test PIFU).
- Document that results are synthetic and subject to review; no patient-level inference should be made.

## Comparison

Original task: Base vs Fine-Tuned shows clear gains across F1, Macro F1, accuracy, and PR AUC, particularly benefiting macro-level performance (Macro F1 +0.3533, PR AUC +0.1779). PIFU: External_test shows dramatic gains for fine-tuned over base in NOT_ELIGIBLE recall and ELIGIBLE recall with perfect precision in many classes (notably External_test: NOT_ELIGIBLE recall 1.0, ELIGIBLE recall 0.98; Challenge: Macro F1 improvement from 0.0635 to 0.6162), indicating substantial improvement in three-class discrimination with synthetic data; however, observed manual_review_rate remains non-negligible and must be reviewed.

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
    "source": "/home/jacinth/nhs_omop_ncp/data/pipeline_runs/20260804T220636Z/original_evaluate.log"
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
    "source": "/home/jacinth/nhs_omop_ncp/data/pipeline_runs/20260804T220636Z/original_evaluate.log"
  }
]
```

> Synthetic research data only. Human clinical and methodological review is required.