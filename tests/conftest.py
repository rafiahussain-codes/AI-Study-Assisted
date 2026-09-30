"""
Pytest configuration and shared fixtures for Student Score Predictor tests.
Ensures tests run in complete isolation with a temporary SQLite database.
"""

import os
import tempfile
import sqlite3
import pytest
import app as flask_app


@pytest.fixture
def temp_db(tmp_path):
    """
    Create a clean, isolated temporary SQLite database for test execution.
    """
    db_file = tmp_path / "test_student_scores.db"
    old_db = os.environ.get("STUDENT_SCORES_DB_PATH")
    os.environ["STUDENT_SCORES_DB_PATH"] = str(db_file)

    # Initialize tables
    flask_app.init_db()

    yield str(db_file)

    # Teardown
    if old_db:
        os.environ["STUDENT_SCORES_DB_PATH"] = old_db
    else:
        os.environ.pop("STUDENT_SCORES_DB_PATH", None)


@pytest.fixture
def client(temp_db):
    """
    Flask test client configured with isolated database.
    """
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def valid_student_data():
    """
    Standard valid academic inputs for student prediction testing.
    """
    return {
        "study_hours": 6.0,
        "previous_score": 75.0,
        "attendance": 85.0,
        "assignment_score": 80.0
    }
