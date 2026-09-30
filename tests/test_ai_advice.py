"""
Tests for AI Study Advice Feature (POST /api/ai-advice) & LLM Service
Covers:
- Valid AI advice generation (mocked provider success)
- Missing API key handling (503 Service Unavailable)
- Invalid API key handling (502 Bad Gateway with safe message)
- Network / SSL / Connection failure handling (502 Bad Gateway)
- Request timeout handling (504 Gateway Timeout)
- Empty or malformed provider response (502 Bad Gateway)
- Predicted score integrity check (preventing tampered scores)
- Verification that no API keys, internal paths, or stack traces are leaked
- Backend health check endpoint (GET /api/health)
"""

import json
import pytest
import requests
import app as flask_app
import llm_service


def test_ai_advice_valid_request(client, monkeypatch):
    """
    Test successful study advice retrieval with mocked provider.
    Should return 200 OK with advice text and model identifier.
    """
    mock_advice = "Your study hours and attendance show good commitment. Focus on practicing complex assignment problems."

    def mock_generate_advice(student_data, predicted_score):
        return True, {
            "success": True,
            "advice": mock_advice,
            "model": "mock-model"
        }, 200

    monkeypatch.setattr(flask_app, "generate_study_advice", mock_generate_advice)

    # First generate a real prediction to get matching predicted_score
    pred_res = client.post("/api/predict", json={
        "study_hours": 6.0,
        "previous_score": 75.0,
        "attendance": 85.0,
        "assignment_score": 80.0
    })
    predicted_score = pred_res.get_json()["predicted_score"]

    payload = {
        "study_hours": 6.0,
        "previous_score": 75.0,
        "attendance": 85.0,
        "assignment_score": 80.0,
        "predicted_score": predicted_score
    }

    response = client.post(
        "/api/ai-advice",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["advice"] == mock_advice
    assert data["model"] == "mock-model"


def test_ai_advice_missing_api_key(client, monkeypatch):
    """
    When no API key is provided, the API must return 503 Service Unavailable
    with a clear, user-friendly error message.
    """
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")

    # Calculate valid predicted score
    pred_res = client.post("/api/predict", json={
        "study_hours": 5.0,
        "previous_score": 70.0,
        "attendance": 80.0,
        "assignment_score": 75.0
    })
    predicted_score = pred_res.get_json()["predicted_score"]

    payload = {
        "study_hours": 5.0,
        "previous_score": 70.0,
        "attendance": 80.0,
        "assignment_score": 75.0,
        "predicted_score": predicted_score
    }

    response = client.post("/api/ai-advice", json=payload)
    assert response.status_code == 503
    data = response.get_json()
    assert data["success"] is False
    assert "LLM_API_KEY" in data["error"] or "not configured" in data["error"]


def test_ai_advice_invalid_api_key(monkeypatch):
    """
    Test that an invalid API key results in a clean 502 error and safe message.
    """
    class MockResponse:
        status_code = 401
        text = '{"error": {"message": "Invalid API Key"}}'
        def json(self):
            return {"error": {"message": "Invalid API Key"}}

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr(requests, "post", mock_post)

    success, payload, code = llm_service._generate_with_groq(
        api_key="gsk_invalid_test_key",
        model_name="openai/gpt-oss-20b",
        system_instruction="sys",
        user_content="user"
    )

    assert success is False
    assert code == 502
    assert "authentication failed" in payload["error"].lower()
    # Ensure raw key is not included in the payload
    assert "gsk_invalid_test_key" not in str(payload)


def test_ai_advice_network_and_ssl_failure(monkeypatch):
    """
    Test that connection or SSL EOF errors are safely caught and returned as 502
    without leaking internal socket information or stack traces.
    """
    def mock_post_conn_err(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Connection reset by peer / SSL UNEXPECTED_EOF")

    monkeypatch.setattr(requests, "post", mock_post_conn_err)

    success, payload, code = llm_service._generate_with_groq(
        api_key="gsk_test_key",
        model_name="openai/gpt-oss-20b",
        system_instruction="sys",
        user_content="user"
    )

    assert success is False
    assert code == 502
    assert "network connection issue" in payload["error"].lower()
    assert "Traceback" not in str(payload)


def test_ai_advice_timeout(monkeypatch):
    """
    Test timeout handling (returns 504 Gateway Timeout with safe message).
    """
    def mock_post_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("Request timed out")

    monkeypatch.setattr(requests, "post", mock_post_timeout)

    success, payload, code = llm_service._generate_with_groq(
        api_key="gsk_test_key",
        model_name="openai/gpt-oss-20b",
        system_instruction="sys",
        user_content="user"
    )

    assert success is False
    assert code == 504
    assert "timed out" in payload["error"].lower()


def test_ai_advice_empty_provider_response(monkeypatch):
    """
    Test handling when provider returns an empty string or empty choices.
    """
    class MockEmptyResponse:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": ""}}]}

    monkeypatch.setattr(requests, "post", lambda *a, **kw: MockEmptyResponse())

    success, payload, code = llm_service._generate_with_groq(
        api_key="gsk_test_key",
        model_name="openai/gpt-oss-20b",
        system_instruction="sys",
        user_content="user"
    )

    assert success is False
    assert code == 502
    assert "empty response" in payload["error"].lower()


def test_ai_advice_tampered_score_rejected(client):
    """
    Verify that if the client sends a predicted_score that does not match the ML model's
    actual calculation for those inputs, the request is rejected with 400 Bad Request.
    """
    payload = {
        "study_hours": 2.0,
        "previous_score": 40.0,
        "attendance": 50.0,
        "assignment_score": 45.0,
        "predicted_score": 99.0  # Inconsistent with low academic inputs!
    }

    response = client.post("/api/ai-advice", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "inconsistent" in data["error"].lower()


def test_health_check_endpoint(client):
    """
    Verify that GET /api/health returns 200 OK and expected status fields.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "database_connected" in data
    assert "llm_configured" in data
    assert data["database_connected"] is True
    assert data["model_loaded"] is True
