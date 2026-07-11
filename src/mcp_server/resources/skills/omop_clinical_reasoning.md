# OMOP Clinical Reasoning Guide

## Purpose

This guide governs how the clinical AI agent should reason over patient data retrieved dataset. It defines data quality rules, clinical reasoning flags, and PIFU suitability indicators that are specific to the OMOP data model and to the characteristics of this dataset discovered during exploratory data analysis and cohort evaluation.

This guide must be loaded at the start of every patient assessment before any clinical conclusions are drawn.

---

## Section 1 — Data Quality Rules

These rules must be applied before any clinical reasoning begins. They govern how to handle known data quality issues in OMOP CDM v5.4 data.

### 1.1 Concept Mapping

- If `condition_concept_id = 0` → the condition is unmapped to a standard OMOP concept. Treat the `condition_source_value` as the only available label. Do NOT infer clinical meaning from a zero concept ID.
- If `drug_concept_id = 0` → the drug is unmapped. Use `drug_source_value` only. Do NOT infer drug class or mechanism from a zero concept ID.
- If `measurement_concept_id = 0` → the measurement type is unmapped. Report the raw value only without clinical interpretation.
- If `race_concept_id = 0` or `ethnicity_concept_id = 0` → race or ethnicity is unmapped. Do NOT infer or assume demographic background.
- If `visit_concept_id = 0` → visit type is unmapped. Do NOT classify the encounter as inpatient, outpatient, or emergency.

### 1.2 Known Dataset Artefacts (delphi-100k specific)

- `note_type_concept_id` maps ALL 1,499,925 clinical notes to the concept "Emergency contraception declined". This is a known synthetic dataset defect. IGNORE the note type label entirely. The note TEXT is clinically valid and should be used for reasoning.
- `visit_concept_id` maps all visit occurrences to the same erroneous concept. Do NOT use visit_concept_id for visit type classification. Use `visit_source_value` if available.
- `range_low` and `range_high` are NULL across all 13,714,401 measurement records. Reference ranges are NOT available in this dataset. Do NOT state that a lab value is normal or abnormal based on reference ranges. Use clinical knowledge only for interpretation.
- Median patient age is 30 years. This does NOT reflect NHS outpatient population age profile and is a synthetic generation artefact. Age-based clinical reasoning should acknowledge this limitation.

### 1.3 Null and Sparse Field Handling

- If `condition_end_date` is NULL → the condition may still be active. Do NOT assume the condition has resolved. State "ongoing (end date not recorded)".
- If `drug_exposure_end_date` is NULL → the medication course end is unknown. Do NOT infer current prescribing status from a null end date.
- If `days_supply` is NULL → medication duration cannot be determined. This affects 90.3% of drug exposure records in delphi-100k. Do NOT infer treatment duration from missing days_supply.
- If `value_as_number` is NULL for a measurement → the result is qualitative only or was not recorded numerically. Use `value_as_concept_id` or `value_source_value` if available. Do NOT trend or compare null numeric values.
- If `quantity` is NULL for a drug exposure → dispensing quantity is unknown. Do NOT infer dose from a missing quantity field.

---

## Section 2 — Clinical Reasoning Rules

These rules govern clinical inference from OMOP structured data fields.

### 2.1 Condition Reasoning

- Absence of conditions in `condition_occurrence` does NOT mean the patient is healthy. 76.9% of patients in this dataset have zero coded conditions, reflecting episodic primary care coding, not a chronic disease registry. Always check clinical notes and observations for additional clinical context when conditions are absent or sparse.
- A patient with zero conditions but multiple visits should be flagged as a potential coding gap. Review `note` table for clinical context.
- A patient with zero conditions AND zero visits has insufficient clinical history for a PIFU assessment. State this explicitly.
- Multiple condition occurrences with the same concept name represent separate episodes, not duplicates. Each entry is a distinct coded encounter.

### 2.2 Medication Reasoning

- A drug exposure with `drug_exposure_start_date = drug_exposure_end_date` is a single administration, likely a vaccination or one-off injection. Do NOT classify as a chronic prescription.
- Influenza vaccine appears in 79% of patients in this dataset. Its presence does NOT indicate chronic medication burden.
- Multiple short drug courses of the same drug over time indicate repeated acute prescribing, not a single chronic prescription. State each course separately.
- Insulin-related exposures in combination with Type 1 diabetes mellitus indicate insulin-dependent diabetes management. Note this explicitly.

### 2.3 Measurement Reasoning

- The dataset contains 13,714,401 measurements across 211 measurement types. Use `get_patient_measurements(limit=50)` to retrieve the most recent values. Explicitly note that historical trends are truncated to the most recent 50.
- Vital signs (blood pressure, BMI, heart rate) are the most commonly recorded measurement types. Lab chemistry values (HbA1c, creatinine, lipids) are present for patients with relevant conditions.
- Without reference ranges (`range_low`/`range_high` are null), interpret numeric values using clinical knowledge only. State the basis of interpretation explicitly.
- A single measurement value cannot establish a trend. Multiple values for the same measurement type over time are required for trend analysis.

### 2.4 Visit Reasoning

- The median number of visits per patient in this dataset is approximately 19. A patient with significantly fewer visits may have an incomplete follow-up record.
- Visit dates span 1926–2012 in this synthetic dataset. This date range is a synthetic generation artefact and does NOT reflect a real NHS follow-up timeline.
- Do NOT use visit dates to infer current clinical status. The dataset represents a historical synthetic record, not a live patient record.

### 2.5 Clinical Notes Reasoning

- Clinical notes are in SOAP format (Subjective, Objective, Assessment, Plan). They contain the richest narrative clinical information in the dataset.
- When structured data (conditions, medications) is sparse or absent, the `note_text` field is the primary source of clinical context.
- Note titles may be null in this dataset. Do NOT rely on note titles for classification, use note text content only.
- The median note length is 52 words. Notes are brief encounter summaries, not full discharge letters.

---

## Section 3 — PIFU Suitability Flags

These flags are cross-cutting indicators that apply regardless of specialty. They must be evaluated before applying any specialty-specific FastPIFU skill.

### 3.1 Flags That Require Caution Before PIFU

| Flag | Condition | Recommended Action |
|------|-----------|-------------------|
| Coding gap suspected | 0 conditions but multiple visits or rich notes | Review note_text before PIFU decision |
| Insufficient history | 0 visits in record | State that follow-up history is absent |
| Unmapped demographics | race_concept_id = 0 | Do not use demographic factors in PIFU reasoning |
| Active comorbidity unknown | condition_end_date null | Treat all conditions as potentially active |
| Medication status unknown | drug_exposure_end_date null | Do not assume current prescribing from historical records |
| Multi-morbidity present | 3+ active conditions | Flag for clinician review — PIFU may not be appropriate |
| Measurement interpretation limited | range_low/range_high null | Do not state normal/abnormal without clinical knowledge basis |

### 3.2 Flags That Support PIFU Consideration

| Flag | Condition |
|------|-----------|
| Stable condition history | Conditions present with distant start dates and no recent acute episodes in visits |
| Low visit frequency | Fewer than 5 visits per year suggests stable outpatient pattern |
| No acute medications | Drug exposures limited to vaccinations and single administrations |
| Rich note record | Substantial note_text available confirming stable clinical narrative |

### 3.3 Multi-Specialty Patients

- If a patient has conditions spanning multiple specialties, apply the FastPIFU skill for each relevant specialty separately.
- The most clinically urgent specialty takes precedence for PIFU routing.
- Cirrhosis always routes to the gastroenterology stable-cirrhosis surveillance pathway regardless of other conditions — this is a hard rule from FastPIFU.
- Neurological conditions (multiple sclerosis, epilepsy) require specialist review before PIFU regardless of stability.

---

## Section 4 — Agent Behaviour Rules

These rules govern how the agent must behave during a patient assessment.

### 4.1 Mandatory Disclosures

The agent MUST include the following in every response:
- Statement that data are synthetic (delphi-100k, OMOP CDM v5.4)
- List of tools called during the assessment
- Any fields that were null or missing and how they affected reasoning
- Explicit statement if conditions list is empty and what this implies

### 4.2 Prohibited Inferences

The agent MUST NOT:
- State that a patient is currently on a medication without checking drug_exposure_end_date
- State that a condition has resolved without a non-null condition_end_date
- Classify a lab value as normal or abnormal using range_low/range_high (these are null in this dataset)
- Use the note_type label ("Emergency contraception declined") for any clinical reasoning
- Infer demographic risk factors from unmapped race or ethnicity concept IDs
- State that a patient has no clinical history when conditions are absent but visits or notes are present

### 4.3 Uncertainty Handling

- If clinical data is insufficient for a PIFU recommendation, state this explicitly rather than guessing.
- If multiple specialties are relevant, list all and identify the primary.
- If the FastPIFU skill for a condition is not available, state which specialty would be relevant and that the protocol was not retrieved.

---

## Section 5 — Cohort-Level Findings

These findings were observed during the Sprint 2 cohort evaluation run across 10 synthetic patients and should inform agent reasoning.

| Finding | Implication |
|---------|-------------|
| 76.9% of patients have zero coded conditions | Absence of conditions is the norm, not the exception |
| 100% of patients have measurement records | Measurements are the most reliably populated clinical domain |
| 90.3% of drug exposures have null days_supply | Duration cannot be inferred from drug records |
| All notes have artefact note_type label | Note text is the only valid notes signal |
| Patients 6 and 10 returned no conditions | Agent correctly flagged coding gap — this is expected behaviour |
| Patient 5 correctly routed to Cardiology | Condition-to-specialty mapping is functioning correctly |
| Response length varied 864–3,146 chars | Query specificity drives response depth |
| 10/10 cohort patients handled successfully | No hard failures observed in the dual-server agent |

---