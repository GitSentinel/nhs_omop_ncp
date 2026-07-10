---
name: fastpifu-gastroenterology
description: >
  Use when deciding what should happen next
  with a gastroenterology patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers stable cirrhosis, hyperferritinaemia/haemochromatosis, hepatitis B, auto-immune liver disease, MASLD, alcohol-related liver disease, inflammatory bowel disease and iron-deficiency anaemia.
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is gastroenterology-specific only. -->

# FastPIFU — Gastroenterology

Gastroenterology PIFU assessment.
Routing, cautions, and hard rules specific to gastroenterology; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Stable cirrhosis | `references/stable-cirrhosis.md` |
| Hyperferritinaemia / haemochromatosis (HH) | `references/hyperferritinaemia-haemochromatosis.md` |
| Hepatitis B | `references/hepatitis-b.md` |
| Auto-immune liver disease | `references/autoimmune-liver-disease.md` |
| MASLD (metabolic dysfunction-associated steatotic liver disease) | `references/masld.md` |
| Alcohol-related liver disease (ArLD) | `references/alcohol-related-liver-disease.md` |
| Inflammatory bowel disease (IBD) | `references/ibd.md` |
| Iron deficiency anaemia (IDA) | `references/iron-deficiency-anaemia.md` |

## Specialty-specific cautions (data-gap traps)

- A chronic liver patient who 'feels well' but is due surveillance (6-monthly USS/AFP, fibrosis re-assessment) — surveillance is independent of clinic and must continue.
- Any patient found to be cirrhotic on assessment — moves to the stable-cirrhosis surveillance pathway regardless of how they present.
- A stable patient who lacks a safe/robust remote-monitoring or DNA-tracking system — without it, PIFU/remote monitoring is not safe.

## Specialty-specific hard rules

- PIFU/remote monitoring for chronic liver disease is only safe where a robust surveillance and DNA-tracking system exists — rationale: surveillance results, not clinic attendance, drive escalation.
- Any patient shown to be cirrhotic enters the stable-cirrhosis surveillance pathway — rationale: HCC and portal-hypertension risk require mandated surveillance.
