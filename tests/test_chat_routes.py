"""
Integration Tests for Chat and RAG Flask REST API Routes
"""

import io
import pytest
from unittest.mock import patch
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_chat_message_validation(client):
    """Test message endpoint validation."""
    # Non-JSON
    res = client.post("/api/chat/message", data="not json")
    assert res.status_code == 400

    # Empty message
    res = client.post("/api/chat/message", json={"message": "   "})
    assert res.status_code == 400
    assert "required" in res.get_json()["error"]


@patch("chat_service.generate_chat_completion")
def test_chat_message_success(mock_chat, client):
    mock_chat.return_value = (
        True,
        {"success": True, "reply": "Sure! Pipelining is like an assembly line.", "model": "test-model"},
        200
    )

    res = client.post("/api/chat/message", json={
        "session_id": "route-test-session",
        "message": "Explain pipelining like I'm a beginner"
    })

    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "assembly line" in data["reply"]
    assert data["session_id"] == "route-test-session"


def test_chat_upload_text_notes(client):
    """Test uploading a text notes file."""
    sample_notes = (
        "Operating Systems Notes\n\n"
        "Virtual memory maps virtual addresses used by an application onto physical addresses in computer memory.\n"
        "Paging avoids external fragmentation by dividing memory into fixed-size frames."
    )
    data = {
        "file": (io.BytesIO(sample_notes.encode("utf-8")), "os_notes.txt"),
        "session_id": "upload-test-session"
    }

    res = client.post(
        "/api/chat/upload",
        data=data,
        content_type="multipart/form-data"
    )

    assert res.status_code == 200
    res_data = res.get_json()
    assert res_data["success"] is True
    assert res_data["document"]["filename"] == "os_notes.txt"
    assert res_data["document"]["chunk_count"] >= 1
    doc_id = res_data["document"]["doc_id"]

    # List documents
    docs_res = client.get(f"/api/chat/documents?session_id=upload-test-session")
    assert docs_res.status_code == 200
    docs = docs_res.get_json()["documents"]
    assert any(d["filename"] == "os_notes.txt" for d in docs)

    # Delete document
    del_res = client.delete(f"/api/chat/documents/{doc_id}?session_id=upload-test-session")
    assert del_res.status_code == 200

    # Verify deleted
    docs_res2 = client.get(f"/api/chat/documents?session_id=upload-test-session")
    assert len(docs_res2.get_json()["documents"]) == 0


def test_chat_upload_invalid_extension(client):
    data = {
        "file": (io.BytesIO(b"binary data"), "notes.exe"),
        "session_id": "invalid-session"
    }
    res = client.post(
        "/api/chat/upload",
        data=data,
        content_type="multipart/form-data"
    )
    assert res.status_code == 400
    assert "Unsupported file type" in res.get_json()["error"]


def test_chat_history_and_reset(client):
    """Test history retrieval and session reset."""
    res = client.get("/api/chat/history?session_id=history-test")
    assert res.status_code == 200
    assert "messages" in res.get_json()

    reset_res = client.post("/api/chat/reset", json={"session_id": "history-test"})
    assert reset_res.status_code == 200
    assert reset_res.get_json()["success"] is True


def test_chat_sessions_crud(client):
    """Test session creation, listing, and deletion."""
    create_res = client.post("/api/chat/sessions", json={"title": "Biology Notes"})
    assert create_res.status_code == 201
    new_sess = create_res.get_json()["session"]
    sess_id = new_sess["session_id"]
    assert new_sess["title"] == "Biology Notes"

    list_res = client.get("/api/chat/sessions")
    assert list_res.status_code == 200
    sessions = list_res.get_json()["sessions"]
    assert any(s["session_id"] == sess_id for s in sessions)

    del_res = client.delete(f"/api/chat/sessions/{sess_id}")
    assert del_res.status_code == 200
    assert del_res.get_json()["success"] is True
