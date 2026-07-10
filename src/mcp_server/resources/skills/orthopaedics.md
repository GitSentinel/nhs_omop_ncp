---
name: fastpifu-orthopaedics
description: >
  Use when deciding what should happen next
  with a orthopaedics patient's follow-up — whether they can move to patient-initiated follow-up,
  be discharged, or stay in timed follow-up. Covers elective joint replacement (hip, knee), ACL reconstruction, foot-and-ankle (bunion, injections), shoulder arthroscopy, and hand pathways (Dupuytren's, ganglion, trigger digit, carpal tunnel, hand trauma).
---

<!-- Universal FastPIFU logic (dispositions, safety asymmetry, SOP checklist, output format) lives
in the orchestrator system prompt. This file is orthopaedics-specific only. -->

# FastPIFU — Orthopaedics

Orthopaedics PIFU assessment.
Routing, cautions, and hard rules specific to orthopaedics; apply under the FastPIFU orchestrator's universal operating instructions.

## Condition routing

| Condition | Reference file |
|---|---|
| Elective primary total hip replacement | `references/total-hip-replacement.md` |
| Elective primary total and uni knee replacement | `references/total-knee-replacement.md` |
| Anterior cruciate ligament reconstruction | `references/acl-reconstruction.md` |
| Hallux valgus/rigidus (bunion) surgery | `references/hallux-valgus-rigidus.md` |
| Foot and ankle injections | `references/foot-ankle-injections.md` |
| Shoulder arthroscopy (rotator cuff repair / subacromial decompression) | `references/shoulder-arthroscopy.md` |
| Dupuytren's disease | `references/dupuytrens.md` |
| Ganglion | `references/ganglion.md` |
| Adult trigger digit | `references/trigger-digit.md` |
| Hand lacerations / cuts / trauma / bite wounds | `references/hand-trauma.md` |
| Carpal tunnel syndrome | `references/carpal-tunnel.md` |

## Specialty-specific cautions (data-gap traps)

- A post-operative patient NOT meeting expected recovery milestones — keep in timed follow-up rather than moving to PIFU/discharge.
- A diagnostic (rather than therapeutic) injection — needs telephone follow-up to act on the result, not a therapeutic PIFU.
- A complex/skin-graft hand case — needs face-to-face consultant follow-up, unlike simple cases.

## Specialty-specific hard rules

- Post-operative PIFU/discharge applies only when the patient is meeting expected recovery milestones — rationale: deviation needs scheduled review.
