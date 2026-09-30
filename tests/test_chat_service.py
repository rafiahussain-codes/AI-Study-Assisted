"""
Unit Tests for ChatService and Conversational Logic
"""

import pytest
from unittest.mock import patch
from chat_service import ChatService


def test_session_creation_and_history():
    service = ChatService()
    session = service.get_or_create_session("test-session-1")
    assert session.session_id == "test-session-1"

    session.add_message("user", "Hello study assistant")
    session.add_message("assistant", "Hello! How can I help you study today?")

    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["role"] == "assistant"

    # Reset history
    service.reset_session("test-session-1")
    assert len(session.messages) == 0


def test_contextual_query_resolution():
    service = ChatService()
    session = service.get_or_create_session("session-followup")

    # When no history, query remains unchanged
    resolved_1 = service._resolve_contextual_query("Explain pipelining", session.messages)
    assert resolved_1 == "Explain pipelining"

    # Add message
    session.add_message("user", "Explain pipelining")
    session.add_message("assistant", "Pipelining is a technique...")

    # Follow-up query with pronoun "its"
    resolved_2 = service._resolve_contextual_query("What are its advantages?", session.messages)
    assert "pipelining" in resolved_2.lower()
    assert "advantages" in resolved_2.lower()


@patch("chat_service.generate_chat_completion")
def test_handle_message_success(mock_generate):
    mock_generate.return_value = (
        True,
        {"success": True, "reply": "Pipelining increases throughput.", "model": "mock-llm"},
        200
    )

    service = ChatService()
    res = service.handle_message(
        session_id="session-test",
        user_message="Explain pipelining in simple words"
    )

    assert res["success"] is True
    assert res["reply"] == "Pipelining increases throughput."
    assert res["session_id"] == "session-test"

    # Check session history
    session = service.get_or_create_session("session-test")
    assert len(session.messages) == 2
    assert session.messages[0]["content"] == "Explain pipelining in simple words"
    assert session.messages[1]["content"] == "Pipelining increases throughput."


@patch("chat_service.generate_chat_completion")
def test_handle_message_study_tools(mock_generate):
    mock_generate.return_value = (
        True,
        {"success": True, "reply": "1. What is pipelining? A) ... B) ...", "model": "mock-llm"},
        200
    )

    service = ChatService()
    res = service.handle_message(
        session_id="session-study-tools",
        user_message="Create MCQs from these notes"
    )

    assert res["success"] is True
    assert "MCQs" in res["reply"] or "pipelining" in res["reply"]
    args, kwargs = mock_generate.call_args
    system_inst = kwargs.get("system_instruction") or args[1]
    assert "Edvora" in system_inst

