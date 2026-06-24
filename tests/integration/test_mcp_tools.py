# Integration tests for the FastMCP tool layer over OMOP DuckDB.
import json
import pytest
from src.mcp_server.server import (
    get_patient_conditions,
    get_patient_measurements,
    get_patient_medications,
    get_patient_notes,
    get_patient_observations,
    get_patient_procedures,
    get_patient_summary,
    get_patient_visits,
    list_available_patients,
)

# Patient with records across the main OMOP domains
TEST_PID = 17247
INVALID_PID = 999_999_999


def test_list_available_patients_returns_ids():
    patient_ids = list_available_patients(limit=10)

    assert isinstance(patient_ids, list)
    assert len(patient_ids) == 10
    assert all(isinstance(person_id, int) for person_id in patient_ids)


def test_list_available_patients_respects_maximum():
    patient_ids = list_available_patients(limit=999)

    assert len(patient_ids) <= 100


def test_get_patient_summary_returns_demographics():
    result = get_patient_summary(TEST_PID)

    assert isinstance(result, dict)
    assert result["person_id"] == TEST_PID
    assert "year_of_birth" in result
    assert "gender_concept_id" in result


def test_get_patient_summary_rejects_invalid_id():
    with pytest.raises(
        ValueError,
        match=r"No (person|patient) found"
    ):
        get_patient_summary(INVALID_PID)


@pytest.mark.parametrize(
    ("tool", "kwargs", "required_fields"),
    [
        (
            get_patient_conditions,
            {},
            ["condition_name", "condition_start_date"]
        ),
        (
            get_patient_medications,
            {},
            ["drug_name", "drug_exposure_start_date"]
        ),
        (
            get_patient_visits,
            {},
            ["visit_type", "visit_start_date"]
        ),
        (
            get_patient_measurements,
            {"limit": 10},
            ["measurement_name", "measurement_date"]
        ),
        (
            get_patient_observations,
            {},
            ["observation_name", "observation_date"]
        ),
        (
            get_patient_notes,
            {},
            ["note_text", "note_date"]
        ),
        (
            get_patient_procedures,
            {},
            ["procedure_name", "procedure_date"]
        ),
    ]
)
def test_patient_domain_tools_return_records(
    tool,
    kwargs,
    required_fields
):
    result = tool(TEST_PID, **kwargs)

    assert isinstance(result, list)
    assert len(result) > 0

    for field in required_fields:
        assert field in result[0]


def test_get_patient_conditions_sorted_newest_first():
    result = get_patient_conditions(TEST_PID)

    dates = [
        record["condition_start_date"]
        for record in result
        if record["condition_start_date"] is not None
    ]

    assert dates == sorted(dates, reverse=True)


@pytest.mark.parametrize("limit", [1, 5, 10])
def test_get_patient_measurements_respects_limit(limit):
    result = get_patient_measurements(TEST_PID, limit=limit)

    assert len(result) <= limit


def test_get_patient_notes_text_not_empty():
    result = get_patient_notes(TEST_PID)

    assert all(
        record.get("note_text")
        for record in result[:5]
    )


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (get_patient_summary, (TEST_PID,)),
        (get_patient_conditions, (TEST_PID,)),
        (get_patient_medications, (TEST_PID,)),
        (get_patient_visits, (TEST_PID,)),
        (get_patient_measurements, (TEST_PID,)),
        (get_patient_observations, (TEST_PID,)),
        (get_patient_notes, (TEST_PID,)),
        (get_patient_procedures, (TEST_PID,)),
    ]
)
def test_tool_outputs_are_json_serialisable(tool, args):
    result = tool(*args)
    json.dumps(result)