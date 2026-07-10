---
name: fastpifu-gynaecology
description: >
  Use when deciding what should happen next
  with a gynaecology patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers pelvic organ prolapse (conservative, pessary and surgical pathways).
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is gynaecology-specific only. -->

# FastPIFU — Gynaecology

Gynaecology PIFU assessment.
Routing, cautions, and hard rules specific to gynaecology; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Pelvic organ prolapse (POP) | `references/pelvic-organ-prolapse.md` |

## Specialty-specific cautions (data-gap traps)

- A patient close to the BMI surgical cut-off — may be listed conditionally to allow time for weight loss; not a straightforward PIFU.
- A post-operative patient where early recurrence or complication is the concern — direct access back to the surgical team is needed.

## Specialty-specific hard rules

- BMI > 35 may make a patient ineligible for surgery — ensure this is communicated and conservative options (physio, pessary) are offered.
