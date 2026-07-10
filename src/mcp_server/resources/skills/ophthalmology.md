---
name: fastpifu-ophthalmology
description: >
  Use when deciding what should happen next
  with a ophthalmology patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers cataract surgery, age-related macular degeneration (AMD), diabetic macular oedema (DMO) and glaucoma, including virtual/asynchronous review pathways.
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is ophthalmology-specific only. -->

# FastPIFU — Ophthalmology

Ophthalmology PIFU assessment.
Routing, cautions, and hard rules specific to ophthalmology; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Cataract surgery | `references/cataract-surgery.md` |
| Age-related macular degeneration (AMD) | `references/amd.md` |
| Diabetic macular oedema (DMO) | `references/diabetic-macular-oedema.md` |
| Glaucoma | `references/glaucoma.md` |

## Specialty-specific cautions (data-gap traps)

- A treated macular patient who looks inactive but is in the strict treat-and-extend window — monitoring must continue at interval before discharge.
- A glaucoma patient whose follow-up interval and discharge eligibility are defined by GIRFT Best Practice for Glaucoma Services, not by clinic convenience.
- PDR/R3a or new wet AMD needs urgent face-to-face assessment, never PIFU.

## Specialty-specific hard rules

- Glaucoma follow-up intervals and discharge are governed by GIRFT Best Practice for Glaucoma Services — rationale: risk-stratified intervals, not ad hoc review.
- Suspected wet AMD requires urgent referral/assessment within one working day — never a PIFU disposition.
