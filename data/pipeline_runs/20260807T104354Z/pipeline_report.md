# Agentic Pipeline Report — 20260807T104354Z

- Target: `pifu`
- Mode: `train-and-evaluate`
- Started: `2026-08-07T10:43:54.263237+00:00`
- Finished: `2026-08-07T13:30:05.232123+00:00`
- Training GPUs: `[0, 1, 2]`
- Evaluation GPU: `0`
- Overall status: `success`

## Executive summary

Pipeline assessment for synthetic PIFU (three-class eligibility) in target mode: train-and-evaluate.
- Validation step: succeeded on dataset 'shared' (stdout indicates project ready with required steps).
- Original data steps (generate, relabel, finetune, evaluate) were skipped as Not required for target=pifu mode=train-and-evaluate.
- PIFU finetuning: succeeded on dataset 'pifu'. Duration: 9806.602244818583 seconds. Key artifacts produced: adapter files at the finetune path. This step completed with return_code 0.
- PIFU evaluation: succeeded on dataset 'pifu'. Evaluated model using finetuned adapter. Runtime: 140.37518893554807 seconds. External evidence includes external/test and challenge/test metrics.

Key results for PIFU three-class eligibility (NOT_ELIGIBLE, BORDERLINE, ELIGIBLE):
- External test (fine-tuned model):
  • Accuracy: 0.9466666666666667; Balanced accuracy: 0.9466666666666667; Macro F1: 0.946837104950836; Weighted F1: 0.9468371049508357
  • NOT_ELIGIBLE: precision 0.9056603773584906; recall 0.96; F1 0.9320388349514563; support 50
  • BORDERLINE: precision 0.9583333333333334; recall 0.92; F1 0.9387755102040817; support 50
  • ELIGIBLE: precision 0.9795918367346939; recall 0.96; F1 0.9696969696969697; support 50
  • NOT_ELIGIBLE recall emphasized at 0.96; ELIGIBLE precision emphasized at 0.9796; Unsafe eligible count: 0; Unsafe eligible rate: 0.0; Manual review rate: 0.32
- Challenge set (finetuned model):
  • Accuracy: 0.7916666666666666; Balanced accuracy: 0.868421052631579; Macro F1: 0.6161616161616162; Weighted F1: 0.8800505050505051
  • NOT_ELIGIBLE: precision 1.0; recall 0.7368421052631579; F1 0.8484848484848485; support 19
  • BORDERLINE: precision 0.0; recall 0.0; F1 0.0; support 0
  • ELIGIBLE: precision 1.0; recall 1.0; F1 1.0; support 5
  • Unsafe eligible count: 0; Manual review rate: 0.20833333333333334
- Test split (finetuned model):
  • Accuracy: 0.9466666666666667; Balanced accuracy: 0.9466666666666667; Macro F1: 0.946837104950836; Weights identical to external test; NOT_ELIGIBLE recall: 0.96; Borderline recall: 0.92; ELIGIBLE precision: 0.9795918367346939; ELIGIBLE recall: 0.96; Unsafe eligible count: 0; Manual review rate: 0.32; Confusion matrix: [[48,2,0],[3,46,1],[2,0,48]]

Notable comparisons (BASE vs FINE-TUNED PIFU model, Qwen/Qwen3.5-9B):
- External test macro F1: 0.2994 -> 0.9468 (+0.6474)
- External test balanced accuracy: 0.3533 -> 0.9467 (+0.5933)
- External test NOT_ELIGIBLE recall: 0.0600 -> 0.9600 (+0.9000)
- External test ELIGIBLE precision: 0.3077 -> 0.9796 (+0.6719)
- External test unsafe eligible count: 20 -> 0
- Challenge set macro F1: 0.0635 -> 0.6162 (+0.5527)
- Challenge set NOT_ELIGIBLE recall: 0.1053 -> 0.7368 (+0.6316)
- Challenge set unsafe eligible count: 7 -> 0

Overall interpretation: The finetuned PIFU model shows strong external/test performance in macro F1 and recall for NOT_ELIGIBLE, with high ELIGIBLE precision. However, challenge-set results are more variable, with some classes failing to meet performance in BORDERLINE. The manual review rate remains a consideration for deployment. These results are derived from synthetic evidence and must be reviewed by humans before any real-world conclusions.

Note: outputs are for synthetic research and require human review.

## Pipeline steps

| Step | Dataset | Status | Duration (s) | Return code |
|---|---|---:|---:|---:|
| validate | shared | succeeded | 0.0 | 0 |
| original_generate | original | skipped | 0.0 | None |
| original_relabel | original | skipped | 0.0 | None |
| original_finetune | original | skipped | 0.0 | None |
| original_evaluate | original | skipped | 0.0 | None |
| pifu_prepare | pifu | skipped | 0.0 | None |
| pifu_finetune | pifu | succeeded | 9806.6 | 0 |
| pifu_evaluate | pifu | succeeded | 140.4 | 0 |

## Key findings

- PIFU finetuning completed; validation and pifu_evaluate steps succeeded.
- External_test: Macro F1 ~0.9468; NOT_ELIGIBLE recall 0.96; ELIGIBLE precision 0.9796; Manual review rate 0.32; Unsafe eligible 0.
- Challenge set: Macro F1 ~0.6162; NOT_ELIGIBLE recall ~0.7368; ELIGIBLE precision/recall = 1.0; Manual review rate ~0.2083; Unsafe eligible 0.
- Test split: Macro F1 ~0.9468; NOT_ELIGIBLE recall 0.96; ELIGIBLE precision 0.9796; Manual review rate 0.32; Unsafe eligible 0.

## Safety flags

- Synthetic data; results require human review; performance varies across challenge set; potential class imbalance across three classes.

## Recommended actions

- Flag results for expert review due to synthetic-data provenance.
- Inspect BORDERLINE class performance on challenge set; consider targeted data augmentation if deployed.
- Monitor manual_review_rate implications in real deployments; confirm not_elible recall and eligible precision thresholds meet domain requirements.

## Comparison

BASE vs FINE-TUNED PIFU: External test Macro F1 improved from 0.2994 to 0.9468; Balanced accuracy from 0.3533 to 0.9467; NOT_ELIGIBLE recall from 0.0600 to 0.9600; ELIGIBLE precision from 0.3077 to 0.9796; Unsafe eligible count from 20 to 0. Challenge set Macro F1 improved from 0.0635 to 0.6162; NOT_ELIGIBLE recall from 0.1053 to 0.7368; Unsafe eligible count from 7 to 0.

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
  }
]
```

> Synthetic research data only. Human clinical and methodological review is required.