---
name: fastpifu-dermatology
description: >
  Use when deciding what should happen next
  with a dermatology patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers skin cancers and pre-cancers (melanoma, cSCC, BCC, actinic keratosis, SCC in situ), multiple atypical moles, post-skin-lesion surgery, inflammatory skin disease, acne, isotretinoin and patch tests.
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is dermatology-specific only. -->

# FastPIFU — Dermatology

Dermatology PIFU assessment.
Routing, cautions, and hard rules specific to dermatology; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| General principles (Dermatology follow-up) | `references/general-principles.md` |
| Melanoma | `references/melanoma.md` |
| Cutaneous squamous cell carcinoma (cSCC) | `references/cutaneous-scc.md` |
| Basal cell carcinoma (BCC) | `references/basal-cell-carcinoma.md` |
| Actinic keratosis (AK) | `references/actinic-keratosis.md` |
| SCC in situ (Bowen's disease) | `references/scc-in-situ.md` |
| Multiple atypical moles | `references/multiple-atypical-moles.md` |
| Post-skin-lesion surgery | `references/post-skin-lesion-surgery.md` |
| Inflammatory skin disease (psoriasis, eczema) | `references/inflammatory-skin-disease.md` |
| Acne | `references/acne.md` |
| Isotretinoin | `references/isotretinoin.md` |
| Patch tests | `references/patch-tests.md` |

## Specialty-specific cautions (data-gap traps)

- A skin-cancer patient who could self-examine scar sites and nodes — PSFU/PIFU may replace routine review (MelFo model), but frailty may make even that unnecessary.
- An inflammatory-disease patient on systemics who is stable: annual nurse monitoring or PIFU may suffice; do not assume guideline-frequency bloods are required (check the SMPC).
- A condition excluded by the local threshold policy — do not offer PIFU; discharge.

## Specialty-specific hard rules

- Do not offer PIFU for conditions excluded by the local threshold policy — rationale: these should be discharged, not retained on a pathway.
