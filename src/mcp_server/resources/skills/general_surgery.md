---
name: fastpifu-general-surgery
description: >
  Use when deciding what should happen next
  with a general surgery patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers general-surgery day-case pathways (inguinal and paraumbilical hernia, laparoscopic cholecystectomy, pilonidal sinus) and colorectal pathways (haemorrhoids, anal fissure, anal fistula, colorectal resection).
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is general surgery-specific only. -->

# FastPIFU — General surgery

General surgery PIFU assessment.
Routing, cautions, and hard rules specific to general surgery; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Inguinal hernia repair | `references/inguinal-hernia.md` |
| Paraumbilical hernia repair | `references/paraumbilical-hernia.md` |
| Laparoscopic cholecystectomy | `references/laparoscopic-cholecystectomy.md` |
| Pilonidal sinus surgery | `references/pilonidal-sinus.md` |
| Haemorrhoids (operative treatment) | `references/haemorrhoids-operative.md` |
| Haemorrhoids (outpatient banding/injection) | `references/haemorrhoids-outpatient.md` |
| Anal fissure (medical management) | `references/anal-fissure-medical.md` |
| Anal fissure (operative treatment) | `references/anal-fissure-operative.md` |
| Anal fistula | `references/anal-fistula.md` |
| Colorectal resection | `references/colorectal-resection.md` |

## Specialty-specific cautions (data-gap traps)

- Rectal bleeding not yet investigated before a haemorrhoid/fissure procedure — straight-to-test flexible sigmoidoscopy is required first.
- A high-complication-risk hernia patient — needs a targeted 6–8 week face-to-face review, not no-follow-up.
- A colorectal-resection specimen showing cancer/dysplasia — moves to active cancer surveillance, not discharge.

## Specialty-specific hard rules

- Ensure rectal bleeding has been investigated (straight-to-test flexible sigmoidoscopy) before any haemorrhoid/fissure pathway — rationale: exclude malignancy.
- A resection/excision specimen with dysplasia or carcinoma triggers targeted oncological follow-up, never routine discharge.
