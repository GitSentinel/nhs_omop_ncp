---
name: fastpifu-omfs
description: >
  Use when deciding what should happen next
  with a oral and maxillofacial surgery patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers OMFS surgical pathways (dentoalveolar surgery, jaw cysts, facial fractures, orthognathic osteotomy, skin lesions) and non-surgical oral-mucosal conditions, TMJ problems and myofascial pain.
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is oral and maxillofacial surgery-specific only. -->

# FastPIFU — Oral and maxillofacial surgery

Oral and maxillofacial surgery PIFU assessment.
Routing, cautions, and hard rules specific to oral and maxillofacial surgery; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Dentoalveolar surgery (incl. MRONJ-risk and soft-tissue procedures) | `references/dentoalveolar-surgery.md` |
| Jaw cysts and odontogenic keratocysts | `references/jaw-cysts.md` |
| Facial fractures (mandibular, condyle, zygomatic, orbital) | `references/facial-fractures.md` |
| Orthognathic osteotomy (single-jaw and bimaxillary) | `references/orthognathic-osteotomy.md` |
| Head and neck skin lesions (OMFS) | `references/skin-lesions.md` |
| Oral mucosal conditions (dry mouth, geographic tongue, recurrent oral ulceration, oral lichen planus) | `references/oral-mucosal-conditions.md` |
| Temporomandibular joint problems and myofascial pain | `references/tmj-and-myofascial-pain.md` |

## Specialty-specific cautions (data-gap traps)

- An irradiated patient or one on bisphosphonates/antiresorptive/anti-angiogenic drugs — needs an 8-week MRONJ-aware review, not the no-follow-up default.
- A large jaw cyst or odontogenic keratocyst — needs radiographic follow-up (and annual review to 5 years for keratocysts), unlike small cysts which discharge.

## Specialty-specific hard rules

- Dentoalveolar surgery in irradiated patients or those on bisphosphonate/antiresorptive/anti-angiogenic drugs follows the SDCEP MRONJ-risk pathway with 8-week review — never the routine no-follow-up default.
