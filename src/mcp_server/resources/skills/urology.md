---
name: fastpifu-urology
description: >
  Use when deciding what should happen next
  with a urology patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers benign urology conditions (stones, bladder-outlet obstruction, erectile dysfunction, LUTS/overactive bladder, bladder pain syndrome, pelvic floor disorders, long-term ISC, post-Peyronie's, recurrent UTI) and urological cancer remote-monitoring (bladder, prostate, kidney).
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is urology-specific only. -->

# FastPIFU — Urology

Urology PIFU assessment.
Routing, cautions, and hard rules specific to urology; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Benign urology conditions (PIFU table) | `references/benign-conditions.md` |
| Urological cancer follow-up and remote monitoring (bladder, prostate, kidney) | `references/cancer-remote-monitoring.md` |

## Specialty-specific cautions (data-gap traps)

- A recurrent-UTI patient with pain on voiding plus back/loin pain or temperature — needs urgent medical attention (NHS 111/GP), not PIFU.
- A cancer-surveillance patient who feels well but is due interval imaging/PSA — surveillance is the point of the pathway and must continue.

## Specialty-specific hard rules

- Recurrent UTI with pain on voiding plus back/loin pain or fever requires urgent medical attention (NHS 111/GP) — never a PIFU disposition.
