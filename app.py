"""
AI Study Assistant - Flask Backend & REST API
A dedicated AI-powered study platform with Document Understanding & RAG.

Endpoints:
- GET  /                        : Render AI Study Assistant Web UI
- GET  /api/health              : Check backend status & LLM readiness
- POST /api/chat/message        : Process conversational queries with RAG retrieval
- POST /api/chat/upload         : Ingest & index study notes (PDF, DOCX, TXT, MD)
- GET  /api/chat/documents      : List active study documents for a session
- DELETE /api/chat/documents/<id>: Remove indexed document from session store
- GET  /api/chat/history        : Retrieve session conversation history
- POST /api/chat/reset          : Clear conversation session history
"""

import os
import io
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Load local environment variables from .env
load_dotenv()

from llm_service import is_llm_configured, get_llm_config
from chat_service import chat_service
from rag.document_loader import DocumentLoadError, SUPPORTED_EXTENSIONS

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Maximum file upload size: 15 MB
MAX_FILE_SIZE = 15 * 1024 * 1024


# ============================================================================
# CORE APPLICATION ROUTES
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    """
    Serve the main AI Study Assistant frontend application.
    """
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify backend service readiness and LLM configuration.
    """
    config = get_llm_config()
    return jsonify({
        "status": "ok",
        "service": "Edvora - AI Study Assistant API",
        "app_name": "Edvora",
        "version": "2.0.0",
        "llm_configured": is_llm_configured(),
        "model": config.get("model", ""),
        "active_sessions": len(chat_service.sessions)
    }), 200


# ============================================================================
# AI STUDY ASSISTANT CHATBOT & RAG API
# ============================================================================

@app.route("/api/chat/message", methods=["POST"])
def chat_message():
    """
    Process incoming study query with RAG retrieval and multi-turn context.
    Expects JSON: { "session_id": str (optional), "message": str }
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request must be JSON with Content-Type: application/json"
        }), 400

    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "").strip() or None

    if not message:
        return jsonify({
            "success": False,
            "error": "Message text is required."
        }), 400

    result = chat_service.handle_message(
        session_id=session_id,
        user_message=message
    )

    status_code = 200 if result.get("success") else 500
    if not result.get("success") and "configured" in result.get("error", "").lower():
        status_code = 503

    return jsonify(result), status_code


@app.route("/api/chat/upload", methods=["POST"])
def chat_upload():
    """
    Upload study materials (PDF, DOCX, TXT, MD) and index them in the session RAG vector store.
    """
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file part in the upload request."
        }), 400

    uploaded_file = request.files["file"]
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    raw_filename = secure_filename(uploaded_file.filename) or "study_material.txt"
    ext = os.path.splitext(raw_filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return jsonify({
            "success": False,
            "error": f"Unsupported file type '{ext}'. Supported formats: {supported_list}"
        }), 400

    session_id = request.form.get("session_id", "").strip() or None
    session = chat_service.get_or_create_session(session_id)

    # Read file stream into memory
    file_bytes = uploaded_file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE // (1024 * 1024)
        return jsonify({
            "success": False,
            "error": f"File exceeds maximum allowed size of {max_mb}MB."
        }), 400

    if len(file_bytes) == 0:
        return jsonify({
            "success": False,
            "error": "Uploaded file is empty."
        }), 400

    try:
        stream = io.BytesIO(file_bytes)
        result = session.rag_pipeline.process_and_index_document(
            file_source=stream,
            filename=raw_filename
        )
        return jsonify({
            "success": True,
            "session_id": session.session_id,
            "document": result
        }), 200

    except DocumentLoadError as err:
        return jsonify({"success": False, "error": str(err)}), 400

    except Exception as exc:
        app.logger.error("Error processing document '%s': %s", raw_filename, exc)
        return jsonify({
            "success": False,
            "error": f"Failed to process document: {str(exc)}"
        }), 500


@app.route("/api/chat/documents", methods=["GET"])
def chat_list_documents():
    """
    List active uploaded documents for the specified chat session.
    """
    session_id = request.args.get("session_id", "").strip() or None
    session = chat_service.get_or_create_session(session_id)
    docs = session.rag_pipeline.list_documents()
    return jsonify({
        "success": True,
        "session_id": session.session_id,
        "documents": docs
    }), 200


@app.route("/api/chat/documents/<doc_id>", methods=["DELETE"])
def chat_delete_document(doc_id):
    """
    Remove an uploaded document and its chunks from the session vector store.
    """
    session_id = request.args.get("session_id", "").strip() or None
    session = chat_service.get_or_create_session(session_id)
    deleted = session.rag_pipeline.delete_document(doc_id)
    if deleted:
        return jsonify({
            "success": True,
            "session_id": session.session_id,
            "deleted": doc_id
        }), 200
    return jsonify({
        "success": False,
        "error": f"Document with ID '{doc_id}' not found."
    }), 404


@app.route("/api/chat/history", methods=["GET"])
def chat_history():
    """
    Retrieve message history and active documents for a session.
    """
    session_id = request.args.get("session_id", "").strip() or None
    session = chat_service.get_or_create_session(session_id)
    return jsonify({
        "success": True,
        "session_id": session.session_id,
        "messages": session.messages,
        "documents": session.rag_pipeline.list_documents()
    }), 200


@app.route("/api/chat/reset", methods=["POST"])
def chat_reset():
    """
    Clear message history for a session (keeps indexed documents intact).
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip() or request.args.get("session_id", "").strip() or None
    if session_id:
        chat_service.reset_session(session_id)
    return jsonify({
        "success": True,
        "session_id": session_id
    }), 200


@app.route("/api/chat/sessions", methods=["GET"])
def chat_list_sessions():
    """
    List all active study chat sessions.
    """
    return jsonify({
        "success": True,
        "sessions": chat_service.list_sessions()
    }), 200


@app.route("/api/chat/sessions", methods=["POST"])
def chat_create_session():
    """
    Explicitly create a new study chat session.
    """
    data = request.get_json(silent=True) or {}
    title = data.get("title", "New Study Session").strip() or "New Study Session"
    session = chat_service.create_new_session(title=title)
    return jsonify({
        "success": True,
        "session": session.to_dict()
    }), 201


@app.route("/api/chat/sessions/<session_id>", methods=["DELETE"])
def chat_delete_session(session_id):
    """
    Delete a study chat session and its vector knowledge base.
    """
    deleted = chat_service.remove_session(session_id)
    return jsonify({
        "success": deleted,
        "session_id": session_id
    }), (200 if deleted else 404)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "API endpoint not found"
        }), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "Internal server error. Please try again later."
        }), 500
    return "An internal server error occurred.", 500


if __name__ == "__main__":
    flask_env = os.environ.get("FLASK_ENV", "development").lower()
    flask_debug = os.environ.get("FLASK_DEBUG", "").lower() in ("true", "1", "yes")
    debug_mode = flask_debug or (flask_env == "development")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=debug_mode)
