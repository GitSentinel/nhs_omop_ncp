# Agentic Pipeline Report — 20260804T125504Z

- Target: `pifu`
- Mode: `post-finetune`
- Started: `2026-08-04T12:55:04.683709+00:00`
- Finished: `2026-08-04T12:57:23.428120+00:00`
- Training GPUs: `[]`
- Evaluation GPU: `0`
- Overall status: `success`

## Executive summary

The deterministic pipeline completed with 2 successful step(s) and 0 failed step(s). LLM assessment fallback used: LLM reporting was disabled.

## Pipeline steps

| Step | Dataset | Status | Duration (s) | Return code |
|---|---|---:|---:|---:|
| validate | shared | succeeded | 0.0 | 0 |
| original_generate | original | skipped | 0.0 | None |
| original_relabel | original | skipped | 0.0 | None |
| original_finetune | original | skipped | 0.0 | None |
| original_evaluate | original | skipped | 0.0 | None |
| pifu_prepare | pifu | skipped | 0.0 | None |
| pifu_finetune | pifu | skipped | 0.0 | None |
| pifu_evaluate | pifu | succeeded | 138.6 | 0 |

## Key findings

- Collected 6 structured metric bundle(s).

## Safety flags

- All data and model outputs are synthetic/research outputs.
- PIFU eligibility predictions require clinician review and must not be used autonomously.

## Recommended actions

- Review the saved metrics, confusion matrices, and false predictions before drawing conclusions.

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
  }
]
```

> Synthetic research data only. Human clinical and methodological review is required.