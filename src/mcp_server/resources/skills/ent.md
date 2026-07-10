---
name: fastpifu-ent
description: >
  Use when deciding what should happen next
  with a ent patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers endoscopic sinus surgery, nasal septoplasty, functional septorhinoplasty, paediatric otitis media with effusion (grommets) and paediatric day-case adenotonsillectomy.
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is ent-specific only. -->

# FastPIFU — ENT

ENT PIFU assessment.
Routing, cautions, and hard rules specific to ent; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Endoscopic sinus surgery | `references/endoscopic-sinus-surgery.md` |
| Nasal septoplasty | `references/nasal-septoplasty.md` |
| Functional septorhinoplasty | `references/functional-septorhinoplasty.md` |
| Paediatric otitis media with effusion (OME) | `references/paediatric-ome.md` |
| Paediatric day-case adenotonsillectomy | `references/paediatric-adenotonsillectomy.md` |

## Specialty-specific cautions (data-gap traps)

- A post-nasal-surgery patient with splints still in — first review is for splint/wound, not yet a discharge point.
- A paediatric OME child whose hearing has NOT improved at the six-week test, or who has concerning social/neurodevelopmental factors — not routine PIFU.

## Specialty-specific hard rules

- Paediatric pathways (OME, adenotonsillectomy) use nurse-led discharge against BADS criteria; PIFU is conditional on the six-week hearing test and absence of concerning social/neurodevelopmental factors.
