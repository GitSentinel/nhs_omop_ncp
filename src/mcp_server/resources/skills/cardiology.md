---
name: fastpifu-cardiology
description: >
  Assess a cardiology patient for PIFU (Patient-Initiated Follow-Up) suitability from their clinical
  documents and return a follow-up disposition. Use when deciding what should happen next with a
  cardiology patient's follow-up — whether they can move to patient-initiated follow-up, be
  discharged, or stay in timed follow-up. Covers arrhythmia, heart valve disease, heart failure,
  chest pain, LBBB, POTS, and other cardiology conditions via general SOP reasoning.
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is cardiology-specific only. -->

# FastPIFU — Cardiology

Cardiology PIFU assessment.
Routing, cautions, and hard rules specific to cardiology; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Arrhythmia — AF, flutter, SVT, VT, ectopy, pre-excitation, devices | `references/arrhythmia.md` |
| Heart valve disease — AS, AR, MR, MS, MVP, BAV, prosthetic | `references/valve-disease.md` |
| Heart failure — HFrEF, HFmrEF, HFpEF | `references/heart-failure.md` |
| Chest pain (suspected cardiac, new presentation) | `references/chest-pain.md` |
| Left bundle branch block (LBBB) | `references/lbbb.md` |
| POTS | `references/pots.md` |
| Stable angina, post-MI, cardiomyopathy, syncope, other | `references/non-pathway.md` |

## Specialty-specific cautions (data-gap traps)

- A patient who feels well but has a falling eGFR after ARNI, rising LFTs on amiodarone, or hyperkalaemia on an MRA/ARNI — the bloods, not the narrative, decide.
- An echo that reads "stable" but whose rate of change between studies is the red flag.
- A suitable primary condition with a co-medication needing secondary-care monitoring (e.g. variable INR on a mechanical valve).
- A deferred plan ("will reassess after uptitration") — PIFU is premature.

## Specialty-specific hard rules

- Recent hospitalisation < 6 months (heart failure) or cardioversion < 3 months (arrhythmia) excludes PIFU — rationale: high early-deterioration risk regardless of current appearance.
- ICD with recent therapies, or any cardiac device without remote monitoring in place, excludes PIFU — rationale: requires scheduled device review.
- Newly initiated class I/III antiarrhythmic (amiodarone, flecainide, sotalol) excludes PIFU — rationale: active secondary-care monitoring requirement.
- Low/sub-therapeutic GDMT doses are not "optimised" without a documented clinical ceiling reason.
