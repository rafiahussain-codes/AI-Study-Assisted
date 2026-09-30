"""
Pytest configuration and shared fixtures for AI Study Assistant tests.
"""

import pytest
import app as flask_app


@pytest.fixture
def client():
    """
    Flask test client configured for testing.
    """
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as test_client:
        yield test_client
