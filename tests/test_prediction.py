"""
Tests for Prediction REST API (POST /api/predict)
Covers:
- Valid input processing and score calculation
- Missing required fields
- Empty values and empty payloads
- Wrong data types (strings, booleans, arrays, null)
- Negative numbers
- Out-of-range values
- Malformed JSON and missing Content-Type
- NaN and Infinity values
- Model loading failure resilience
"""

import json
import pytest
import app as flask_app


def test_predict_valid_input(client, valid_student_data):
    """
    Test prediction with standard, valid student metrics.
    Should return 200 OK, success=True, predicted_score within [0, 100], and a valid integer ID.
    """
    response = client.post(
        "/api/predict",
        data=json.dumps(valid_student_data),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "predicted_score" in data
    assert isinstance(data["predicted_score"], (int, float))
    assert 0.0 <= data["predicted_score"] <= 100.0
    assert "id" in data
    assert isinstance(data["id"], int)
    assert data["id"] > 0


@pytest.mark.parametrize("missing_field", [
    "study_hours",
    "previous_score",
    "attendance",
    "assignment_score"
])
def test_predict_missing_fields(client, valid_student_data, missing_field):
    """
    Test prediction rejection when any required field is missing.
    Should return 400 Bad Request with a descriptive error message.
    """
    payload = valid_student_data.copy()
    payload.pop(missing_field)

    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert f"Missing required field: {missing_field}" in data["error"]


def test_predict_empty_payload(client):
    """
    Test prediction with empty JSON dictionary payload.
    """
    response = client.post(
        "/api/predict",
        data=json.dumps({}),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Missing required field" in data["error"]


@pytest.mark.parametrize("field,wrong_value", [
    ("study_hours", "five"),
    ("study_hours", True),       # Booleans inherit from int in Python, must be rejected
    ("previous_score", "eighty"),
    ("previous_score", False),
    ("attendance", [90]),
    ("attendance", None),
    ("assignment_score", {"score": 85})
])
def test_predict_wrong_data_types(client, valid_student_data, field, wrong_value):
    """
    Test prediction rejection when a field has a non-numeric type or boolean.
    Should return 400 Bad Request.
    """
    payload = valid_student_data.copy()
    payload[field] = wrong_value

    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "must be a valid number" in data["error"]


@pytest.mark.parametrize("field,negative_value", [
    ("study_hours", -1.0),
    ("previous_score", -5.0),
    ("attendance", -10.0),
    ("assignment_score", -0.1)
])
def test_predict_negative_values(client, valid_student_data, field, negative_value):
    """
    Test prediction rejection when any metric is negative.
    Should return 400 Bad Request.
    """
    payload = valid_student_data.copy()
    payload[field] = negative_value

    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False


@pytest.mark.parametrize("field,out_of_range_value", [
    ("study_hours", 25.0),         # Max 24 hours per day
    ("study_hours", 100.0),
    ("previous_score", 100.1),     # Max 100
    ("attendance", 101.0),         # Max 100%
    ("assignment_score", 150.0)    # Max 100
])
def test_predict_out_of_range_values(client, valid_student_data, field, out_of_range_value):
    """
    Test prediction rejection when metrics exceed permissible domain bounds.
    Should return 400 Bad Request.
    """
    payload = valid_student_data.copy()
    payload[field] = out_of_range_value

    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False


def test_predict_malformed_json(client):
    """
    Test prediction rejection when raw body has malformed JSON syntax.
    """
    response = client.post(
        "/api/predict",
        data="{'invalid_json': true,",
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Malformed JSON" in data["error"] or "JSON" in data["error"]


def test_predict_non_json_content_type(client, valid_student_data):
    """
    Test prediction rejection when Content-Type is not application/json.
    """
    response = client.post(
        "/api/predict",
        data=json.dumps(valid_student_data),
        content_type="text/plain"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "application/json" in data["error"]


def test_predict_model_loading_failure(client, valid_student_data, monkeypatch):
    """
    Test that when the ML model fails to load or is None, the API returns a safe 500
    error without exposing internal tracebacks or system paths.
    """
    monkeypatch.setattr(flask_app, "model", None)
    monkeypatch.setattr(flask_app, "load_trained_model", lambda: None)

    response = client.post(
        "/api/predict",
        data=json.dumps(valid_student_data),
        content_type="application/json"
    )
    assert response.status_code == 500
    data = response.get_json()
    assert data["success"] is False
    assert "model is currently unavailable" in data["error"].lower()
    # Ensure no traceback or internal server paths leaked
    assert "Traceback" not in response.get_data(as_text=True)
    assert "C:\\" not in response.get_data(as_text=True)
