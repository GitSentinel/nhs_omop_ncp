# Agentic Pipeline Report — 20260805T225843Z

- Target: `both`
- Mode: `post-finetune`
- Started: `2026-08-05T22:58:43.528685+00:00`
- Finished: `2026-08-05T23:05:51.060446+00:00`
- Training GPUs: `[]`
- Evaluation GPU: `0`
- Overall status: `success`

## Executive summary

This is a synthetic-research pipeline assessment (post-finetune) for a dual-target evaluation: the original binary treatment-event task and the PIFU three-class eligibility task. The report collates results from the original-evaluate run and the pifu_evaluate runs across the base and finetuned models, using the provided evidence.

Original task (binary treatment-event): base vs finetuned comparison shows substantial gains after finetuning on the test set. Key numbers (finetuned vs baseline):
- F1: 0.7831 -> 0.9010 (+0.1179)
- Macro F1: 0.4864 -> 0.8397 (+0.3533)
- Accuracy: 0.6578 -> 0.8631 (+0.2053)
- Balanced accuracy: 0.5366 -> 0.8209 (+0.2843)
- Precision: 0.6607 -> 0.8420 (+0.1813)
- Recall: 0.9610 -> 0.9688 (+0.0078)
- ROC AUC: 0.6214 -> 0.9564 (+0.3350)
- PR AUC: 0.7948 -> 0.9727 (+0.1779)

Finetuned model test metrics (original_evaluate): F1 0.9010, Macro F1 0.8397, Accuracy 0.8631, Balanced accuracy 0.8209, Precision 0.8420, Recall 0.9688, ROC AUC 0.9564, PR AUC 0.9727. Confusion matrix (FN/TN/FP/TP): TN=144, FP=70, FN=12, TP=373.

PIFU (three-class) results show distinct improvements with finetuning on external and challenge tests:
- External test (finetuned): NOT_ELIGIBLE recall 1.0000; ELIGIBLE precision 1.0000; Unsafe eligible 0 (0.0000); Manual review rate 0.3067; confusion matrix: [[50, 0, 0], [4, 46, 0], [1, 0, 49]]; Macro F1 0.9669. Per-class reports include NOT_ELIGIBLE: precision 0.9090909090909091, recall 1.0; BORDERLINE: precision 1.0, recall 0.92; ELIGIBLE: precision 1.0, recall 0.98.
- Challenge test (finetuned): NOT_ELIGIBLE recall 0.7368; ELIGIBLE precision 1.0000; Unsafe eligible 0 (0.0000); Manual review rate 0.2083; confusion matrix: [[14, 5, 0], [0, 0, 0], [0, 0, 5]]; Macro F1 0.6162; NOT_ELIGIBLE precision 1.0, recall 0.7368; ELIGIBLE precision 1.0, recall 1.0.

Baseline (base) PIFU external/test metrics remain far lower, illustrating the magnitude of improvement after finetuning:
- External test (base): accuracy 0.3533; balanced_accuracy 0.3533; macro_f1 0.2994469582704877; NOT_ELIGIBLE precision 0.6; NOT_ELIGIBLE recall 0.06; Borderline precision 0.0; Borderline recall 0.68; ELIGIBLE precision 0.3076923076923077; ELIGIBLE recall 0.32; Unsafe eligible count 20; manual review rate 0.62.
- Challenge (base): accuracy 0.0833; balanced_accuracy 0.0526; macro_f1 0.06349; NOT_ELIGIBLE precision 1.0; NOT_ELIGIBLE recall 0.1053; ELIGIBLE precision 0.0; ELIGIBLE recall 0.0; Unsafe eligible count 7; manual review rate 0.625.

Notes on model evaluation: metrics include accuracy, balanced accuracy, macro F1, PR AUC for the original task, and NOT_ELIGIBLE/ELIGIBLE/BORDERLINE per-class metrics for PIFU. The evidence also includes confusion matrices and per-class F1 scores. All results shown here are derived from synthetic data for research purposes and require human review before any clinical interpretation.

Comparison across models indicates clear performance gains with finetuning for both tasks, though PIFU results highlight differing behavior by test set (external vs challenge). The post-finetune PIFU results show strong NOT_ELIGIBLE recall and ELIGIBLE precision on external tests, with some manual-review overhead; this pattern is less pronounced on the challenge split. For the original task, finetuning yields broad improvements across F1, macro F1, accuracy, and PR AUC, with only modest gains in recall.

Outputs are for synthetic research and require human review.

## Pipeline steps

| Step | Dataset | Status | Duration (s) | Return code |
|---|---|---:|---:|---:|
| validate | shared | succeeded | 0.0 | 0 |
| original_generate | original | skipped | 0.0 | None |
| original_relabel | original | skipped | 0.0 | None |
| original_finetune | original | skipped | 0.0 | None |
| original_evaluate | original | succeeded | 251.2 | 0 |
| pifu_prepare | pifu | skipped | 0.0 | None |
| pifu_finetune | pifu | skipped | 0.0 | None |
| pifu_evaluate | pifu | succeeded | 138.5 | 0 |

## Key findings

- Original task: large uplift from baseline to finetuned across F1, macro F1, accuracy, balanced accuracy, precision and PR AUC (e.g., F1 0.7831→0.9010; Macro F1 0.4864→0.8397; PR AUC 0.7948→0.9727).
- Original task: finetuned test metrics (original_evaluate) achieved F1 0.9010, Macro F1 0.8397, Accuracy 0.8631, Balanced accuracy 0.8209, Precision 0.8420, Recall 0.9688, ROC AUC 0.9564, PR AUC 0.9727; Confusion matrix: TN=144, FP=70, FN=12, TP=373.
- PIFU external_test (finetuned): NOT_ELIGIBLE recall 1.0000; ELIGIBLE precision 1.0000; Unsafe eligible 0 (0.0000); Manual review rate 0.3067; Macro F1 0.9669; confusion_matrix [[50,0,0],[4,46,0],[1,0,49]].
- PIFU challenge (finetuned): NOT_ELIGIBLE recall 0.7368; ELIGIBLE precision 1.0000; Manual review rate 0.2083; Unsafe eligible 0; Macro F1 0.6162; confusion_matrix [[14,5,0],[0,0,0],[0,0,5]].
- Baseline PIFU external_test (base): NOT_ELIGIBLE recall 0.06; NOT_ELIGIBLE precision 0.6; Borderline recall 0.68; ELIGIBLE recall 0.32; Unsafe eligible count 20; manual_review_rate 0.62.

## Safety flags

- Synthetic data; results are for synthetic NHS research and require human review before any clinical interpretation.

## Recommended actions

- Flag the synthetic nature of dataset when reporting findings.
- Proceed to external validation on real or additional datasets.
- Perform error analysis focusing on FN cases in the original task and NOT_ELIGIBLE/ELIGIBLE misclassifications in PIFU.
- Review manual-review rate implications and potential threshold calibration for external_test vs challenge_test splits.
- Document all deltas between base and finetuned models and ensure transparent reporting of uncertainty.

## Comparison

Original task shows substantial performance uplift after finetuning across F1, Macro F1, accuracy, balanced accuracy, precision, recall, ROC AUC and PR AUC (Base vs Finetuned). PIFU shows dramatic improvements on external_test after finetuning (NOT_ELIGIBLE recall to 1.0000; ELIGIBLE precision to 1.0000; manual-review rate ~0.3067) but more modest gains on the challenge_test (NOT_ELIGIBLE recall 0.7368; manual-review rate ~0.2083; macro F1 0.6162).

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
    "source": "/home/jacinth/nhs_omop_ncp/data/pipeline_runs/20260805T225843Z/original_evaluate.log"
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
    "source": "/home/jacinth/nhs_omop_ncp/data/pipeline_runs/20260805T225843Z/original_evaluate.log"
  }
]
```

> Synthetic research data only. Human clinical and methodological review is required.