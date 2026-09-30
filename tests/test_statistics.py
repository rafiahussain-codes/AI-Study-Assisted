"""
Tests for Dashboard Summary Statistics API (GET /api/statistics)
Covers:
- Empty database aggregation state
- Accurate computation of total, average, highest, and lowest scores
- Rounding behavior of aggregated metrics
- Safe error handling without revealing server internals
"""

import sqlite3
import pytest
import app as flask_app


def test_empty_database_statistics(client):
    """
    When no predictions exist, statistics endpoint should return total_predictions: 0
    and None for average, highest, and lowest scores.
    """
    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["total_predictions"] == 0
    assert data["average_score"] is None
    assert data["highest_score"] is None
    assert data["lowest_score"] is None


def test_statistics_calculated_correctly(client, temp_db):
    """
    Verify aggregate calculations:
    Scores: 60.0, 80.0, 100.0
    Total: 3
    Average: (60 + 80 + 100) / 3 = 80.0
    Highest: 100.0
    Lowest: 60.0
    """
    flask_app.save_prediction(2.0, 55.0, 60.0, 58.0, 60.0)
    flask_app.save_prediction(5.0, 75.0, 80.0, 78.0, 80.0)
    flask_app.save_prediction(8.0, 95.0, 100.0, 95.0, 100.0)

    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["total_predictions"] == 3
    assert data["average_score"] == 80.0
    assert data["highest_score"] == 100.0
    assert data["lowest_score"] == 60.0


def test_single_prediction_statistics(client, temp_db):
    """
    Verify statistics when exactly one prediction exists.
    Average, highest, and lowest should all equal the single predicted score.
    """
    flask_app.save_prediction(4.0, 70.0, 80.0, 75.0, 74.2)

    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["total_predictions"] == 1
    assert data["average_score"] == 74.2
    assert data["highest_score"] == 74.2
    assert data["lowest_score"] == 74.2


def test_statistics_error_handling(client, monkeypatch):
    """
    Verify safe 500 error response if statistics calculation fails.
    """
    def broken_statistics():
        raise sqlite3.OperationalError("Database locked")

    monkeypatch.setattr(flask_app, "get_prediction_statistics", broken_statistics)

    response = client.get("/api/statistics")
    assert response.status_code == 500
    data = response.get_json()
    assert data["success"] is False
    assert "Unable to load dashboard statistics" in data["error"]
    assert "Traceback" not in response.get_data(as_text=True)
