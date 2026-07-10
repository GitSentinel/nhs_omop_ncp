---
name: fastpifu-spinal
description: >
  Use when deciding what should happen next
  with a spinal patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers lumbar facet-joint (medial branch block) and nerve-root-block injections, posterior lumbar decompression/discectomy, and the adolescent idiopathic scoliosis pathway.
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is spinal-specific only. -->

# FastPIFU — Spinal

Spinal PIFU assessment.
Routing, cautions, and hard rules specific to spinal; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Lumbar medial branch block (facet joint injections) | `references/facet-joint-injections.md` |
| Lumbar nerve root block (epidural) | `references/lumbar-nerve-root-block.md` |
| Posterior lumbar decompression / discectomy | `references/lumbar-decompression-discectomy.md` |
| Adolescent idiopathic scoliosis (AIS) | `references/adolescent-idiopathic-scoliosis.md` |

## Specialty-specific cautions (data-gap traps)

- An injection patient with a diagnostic response but returning pain — offer denervation (facet) or consider surgery (nerve root); injections should NOT be repeated.
- An AIS patient where PIFU is being used as a substitute for discharge — it must run alongside planned monitoring, not replace appropriate discharge.

## Specialty-specific hard rules

- Spinal injections should NOT be repeated — rationale: a diagnostic response with recurrent pain warrants denervation or surgical consideration, not repeat injection.
- PIFU is used whenever clinically appropriate but never as an alternative to discharge.
