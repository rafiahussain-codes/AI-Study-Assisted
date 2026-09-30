"""
Tests for SQLite Database Operations & Prediction History API
Covers:
- Database schema initialization
- Prediction record persistence via parameterized queries
- Safe handling of SQL injection payloads in input fields
- Retrieval of prediction history (GET /api/predictions)
- Ordering of history (newest first)
- Empty database history response
- Query parameter limit handling (?limit=N)
- Safe error handling when database fails
"""

import json
import sqlite3
import pytest
import app as flask_app


def test_prediction_saved_correctly(temp_db):
    """
    Verify that save_prediction inserts records accurately with expected columns and values.
    """
    record_id = flask_app.save_prediction(
        study_hours=5.5,
        previous_score=78.0,
        attendance=90.0,
        assignment_score=85.0,
        predicted_score=82.4
    )
    assert isinstance(record_id, int)
    assert record_id > 0

    with flask_app.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["study_hours"] == 5.5
        assert row["previous_score"] == 78.0
        assert row["attendance"] == 90.0
        assert row["assignment_score"] == 85.0
        assert row["predicted_score"] == 82.4
        assert row["created_at"] is not None


def test_sql_injection_attempt_prevented(client):
    """
    Verify that SQL injection strings in JSON inputs are safely rejected by validation
    and cannot reach SQLite concatenation.
    """
    injection_payload = {
        "study_hours": "5; DROP TABLE predictions; --",
        "previous_score": 75.0,
        "attendance": 85.0,
        "assignment_score": 80.0
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(injection_payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False

    # Verify table is still intact and queryable
    with flask_app.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        assert cursor.fetchone()[0] == 0


def test_empty_database_history(client):
    """
    When no predictions exist in the database, GET /api/predictions should return
    an empty list with success=True.
    """
    response = client.get("/api/predictions")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["predictions"] == []


def test_prediction_history_returned_newest_first(client, temp_db):
    """
    Verify that multiple predictions are retrieved in reverse chronological order (newest first).
    """
    flask_app.save_prediction(2.0, 50.0, 60.0, 55.0, 52.0)
    flask_app.save_prediction(6.0, 80.0, 90.0, 85.0, 84.5)
    flask_app.save_prediction(8.0, 95.0, 98.0, 92.0, 94.0)

    response = client.get("/api/predictions")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    predictions = data["predictions"]
    assert len(predictions) == 3

    # Newest ID should be first
    assert predictions[0]["study_hours"] == 8.0
    assert predictions[1]["study_hours"] == 6.0
    assert predictions[2]["study_hours"] == 2.0


def test_prediction_history_limit_parameter(client, temp_db):
    """
    Verify that GET /api/predictions respects the ?limit=N query parameter.
    """
    for i in range(10):
        flask_app.save_prediction(float(i + 1), 70.0, 80.0, 75.0, 72.0)

    response = client.get("/api/predictions?limit=3")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["predictions"]) == 3

    # Negative or non-numeric limit falls back safely to default
    response_invalid = client.get("/api/predictions?limit=invalid")
    assert response_invalid.status_code == 200
    assert len(response_invalid.get_json()["predictions"]) == 10


def test_database_error_handling(client, monkeypatch):
    """
    Verify that if the database encounters an operational error, the API returns
    a safe 500 error response without leaking tracebacks or paths.
    """
    def broken_get_predictions(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated disk I/O failure")

    monkeypatch.setattr(flask_app, "get_predictions", broken_get_predictions)

    response = client.get("/api/predictions")
    assert response.status_code == 500
    data = response.get_json()
    assert data["success"] is False
    assert "Unable to load prediction history" in data["error"]
    assert "Traceback" not in response.get_data(as_text=True)
