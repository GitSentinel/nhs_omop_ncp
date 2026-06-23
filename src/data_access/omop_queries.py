# Patient-level query functions for the OMOP CDM v5.4 database
from src.data_access.connection import get_table


def _validate_person_id(person_id: int) -> int:
    # Validate and return an OMOP person identifier
    person_id = int(person_id)

    if person_id <= 0:
        raise ValueError("person_id must be a positive integer.")

    return person_id


def _validate_limit(limit: int, maximum: int = 1000) -> int:
    # Restrict a query limit to a valid range
    return max(1, min(int(limit), maximum))


def _to_records(query) -> list[dict]:
    # Execute an Ibis query and return dictionary records
    return query.execute().to_dict(orient="records")


def get_person(person_id: int) -> dict:
    # Return the demographic record for one patient
    person_id = _validate_person_id(person_id)
    person = get_table("person")

    result = (
        person
        .filter(person.person_id == person_id)
        .execute()
    )

    if result.empty:
        raise ValueError(
            f"No patient found with person_id={person_id}."
        )

    return result.iloc[0].to_dict()


def get_conditions(person_id: int) -> list[dict]:
    # Return condition records for one patient
    person_id = _validate_person_id(person_id)
    conditions = get_table("condition_occurrence")
    concept = get_table("concept")

    query = (
        conditions
        .filter(conditions.person_id == person_id)
        .left_join(
            concept,
            conditions.condition_concept_id == concept.concept_id
        )
        .select(
            conditions.condition_occurrence_id,
            conditions.condition_concept_id,
            concept.concept_name.name("condition_name"),
            conditions.condition_start_date,
            conditions.condition_end_date,
            conditions.condition_type_concept_id
        )
        .order_by(conditions.condition_start_date.desc())
    )

    return _to_records(query)


def get_medications(person_id: int) -> list[dict]:
    # Return medication records for one patient
    person_id = _validate_person_id(person_id)
    medications = get_table("drug_exposure")
    concept = get_table("concept")

    query = (
        medications
        .filter(medications.person_id == person_id)
        .left_join(
            concept,
            medications.drug_concept_id == concept.concept_id
        )
        .select(
            medications.drug_exposure_id,
            medications.drug_concept_id,
            concept.concept_name.name("drug_name"),
            medications.drug_exposure_start_date,
            medications.drug_exposure_end_date,
            medications.quantity,
            medications.days_supply
        )
        .order_by(medications.drug_exposure_start_date.desc())
    )

    return _to_records(query)


def get_visits(person_id: int) -> list[dict]:
    # Return healthcare visit records for one patient
    person_id = _validate_person_id(person_id)
    visits = get_table("visit_occurrence")
    concept = get_table("concept")

    query = (
        visits
        .filter(visits.person_id == person_id)
        .left_join(
            concept,
            visits.visit_concept_id == concept.concept_id
        )
        .select(
            visits.visit_occurrence_id,
            visits.visit_concept_id,
            concept.concept_name.name("visit_type"),
            visits.visit_start_date,
            visits.visit_end_date,
            visits.care_site_id
        )
        .order_by(visits.visit_start_date.desc())
    )

    return _to_records(query)


def get_measurements(
    person_id: int,
    limit: int = 50
) -> list[dict]:
    # Return recent measurement records for one patient
    person_id = _validate_person_id(person_id)
    limit = _validate_limit(limit)

    measurements = get_table("measurement")
    concept = get_table("concept")

    query = (
        measurements
        .filter(measurements.person_id == person_id)
        .left_join(
            concept,
            measurements.measurement_concept_id == concept.concept_id
        )
        .select(
            measurements.measurement_id,
            measurements.measurement_concept_id,
            concept.concept_name.name("measurement_name"),
            measurements.measurement_date,
            measurements.value_as_number,
            measurements.unit_source_value,
            measurements.range_low,
            measurements.range_high
        )
        .order_by(measurements.measurement_date.desc())
        .limit(limit)
    )

    return _to_records(query)


def get_observations(person_id: int) -> list[dict]:
    # Return observation records for one patient
    person_id = _validate_person_id(person_id)
    observations = get_table("observation")
    concept = get_table("concept")

    query = (
        observations
        .filter(observations.person_id == person_id)
        .left_join(
            concept,
            observations.observation_concept_id == concept.concept_id
        )
        .select(
            observations.observation_id,
            observations.observation_concept_id,
            concept.concept_name.name("observation_name"),
            observations.observation_date,
            observations.value_as_string,
            observations.value_as_number
        )
        .order_by(observations.observation_date.desc())
    )

    return _to_records(query)


def get_notes(person_id: int) -> list[dict]:
    # Return clinical notes for one patient
    person_id = _validate_person_id(person_id)
    notes = get_table("note")

    query = (
        notes
        .filter(notes.person_id == person_id)
        .select(
            notes.note_id,
            notes.note_date,
            notes.note_type_concept_id,
            notes.note_class_concept_id,
            notes.note_title,
            notes.note_text
        )
        .order_by(notes.note_date.desc())
    )

    return _to_records(query)


def get_procedures(person_id: int) -> list[dict]:
    # Return procedure records for one patient
    person_id = _validate_person_id(person_id)
    procedures = get_table("procedure_occurrence")
    concept = get_table("concept")

    query = (
        procedures
        .filter(procedures.person_id == person_id)
        .left_join(
            concept,
            procedures.procedure_concept_id == concept.concept_id
        )
        .select(
            procedures.procedure_occurrence_id,
            procedures.procedure_concept_id,
            concept.concept_name.name("procedure_name"),
            procedures.procedure_date,
            procedures.procedure_type_concept_id
        )
        .order_by(procedures.procedure_date.desc())
    )

    return _to_records(query)