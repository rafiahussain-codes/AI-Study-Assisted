"""
Student Score Predictor - Flask Backend & REST API
Phase 8: Dashboard & Application Integration

This module serves the frontend dashboard and provides the REST API endpoints:
- GET  /                : Render frontend dashboard HTML
- GET  /api/health      : Check backend status, database, and model availability
- POST /api/predict     : Real-time ML inference + SQLite persistence
- GET  /api/predictions : Retrieve recent prediction history from SQLite
- GET  /api/statistics  : Retrieve aggregate summary statistics from SQLite
"""

import os
import math
import sqlite3
import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# Load local environment variables from .env
load_dotenv()

# Import isolated LLM service
from llm_service import generate_study_advice, is_llm_configured


app = Flask(__name__)

# Base directory and asset path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("STUDENT_SCORES_MODEL_PATH") or os.path.join(BASE_DIR, "models", "student_score_model.pkl")
DB_PATH = os.environ.get("STUDENT_SCORES_DB_PATH") or os.path.join(BASE_DIR, "database", "student_scores.db")


# Required input fields and their sensible range constraints
VALIDATION_RULES = {
    "study_hours": {
        "min": 0,
        "max": 24,
        "name": "Study hours",
        "error_msg": "Study hours cannot be negative and must be between 0 and 24 hours per day."
    },
    "previous_score": {
        "min": 0,
        "max": 100,
        "name": "Previous score",
        "error_msg": "Previous score must be between 0 and 100."
    },
    "attendance": {
        "min": 0,
        "max": 100,
        "name": "Attendance",
        "error_msg": "Attendance must be between 0 and 100%."
    },
    "assignment_score": {
        "min": 0,
        "max": 100,
        "name": "Assignment score",
        "error_msg": "Assignment score must be between 0 and 100."
    }
}


# ============================================================================
# DATABASE FUNCTIONS (sqlite3)
# ============================================================================

def get_db_connection():
    """
    Open and configure a connection to the local SQLite database.
    row_factory is set to sqlite3.Row for dictionary-like column access.
    """
    target_db = os.environ.get("STUDENT_SCORES_DB_PATH") or DB_PATH
    conn = sqlite3.connect(target_db)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize SQLite database and create the predictions table if it doesn't exist.
    Creates the database directory automatically if missing.
    """
    target_db = os.environ.get("STUDENT_SCORES_DB_PATH") or DB_PATH
    db_dir = os.path.dirname(os.path.abspath(target_db))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_hours REAL NOT NULL,
                previous_score REAL NOT NULL,
                attendance REAL NOT NULL,
                assignment_score REAL NOT NULL,
                predicted_score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    app.logger.info("Database initialized successfully at: %s", target_db)



def save_prediction(study_hours, previous_score, attendance, assignment_score, predicted_score):
    """
    Persist a successful prediction record into SQLite using a parameterized query.
    Returns the auto-generated integer record ID.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions
                (study_hours, previous_score, attendance, assignment_score, predicted_score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (study_hours, previous_score, attendance, assignment_score, predicted_score)
        )
        conn.commit()
        return cursor.lastrowid


def get_predictions(limit=50):
    """
    Retrieve previous prediction records ordered newest first (DESC).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, study_hours, previous_score, attendance, assignment_score, predicted_score, created_at
            FROM predictions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        # Convert sqlite3.Row objects into standard serializable Python dictionaries
        return [dict(row) for row in rows]


def get_prediction_statistics():
    """
    Compute aggregate statistics across stored predictions using SQL aggregation.
    Returns total count, average, max, and min predicted scores.
    Returns None for numerical statistics if no predictions exist yet.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) AS total_predictions,
                AVG(predicted_score) AS average_score,
                MAX(predicted_score) AS highest_score,
                MIN(predicted_score) AS lowest_score
            FROM predictions;
        """)
        row = cursor.fetchone()

        total = row["total_predictions"] if row and row["total_predictions"] is not None else 0
        if total == 0:
            return {
                "total_predictions": 0,
                "average_score": None,
                "highest_score": None,
                "lowest_score": None
            }

        return {
            "total_predictions": total,
            "average_score": round(float(row["average_score"]), 1) if row["average_score"] is not None else None,
            "highest_score": round(float(row["highest_score"]), 1) if row["highest_score"] is not None else None,
            "lowest_score": round(float(row["lowest_score"]), 1) if row["lowest_score"] is not None else None
        }


# Initialize database table upon startup
init_db()


# ============================================================================
# MACHINE LEARNING MODEL LOADER
# ============================================================================

def load_trained_model():
    """
    Load the trained Scikit-learn model serialized during Phase 5.
    Returns the loaded model instance or None if loading fails.
    """
    target_model = os.environ.get("STUDENT_SCORES_MODEL_PATH") or MODEL_PATH
    if not os.path.exists(target_model):
        app.logger.error("Trained model file not found at: %s", target_model)
        return None
    try:
        loaded_model = joblib.load(target_model)
        app.logger.info("Successfully loaded ML model from: %s", target_model)
        return loaded_model
    except Exception as exc:
        app.logger.error("Error loading model from %s: %s", target_model, exc)
        return None



# Load model at application startup to avoid redundant disk I/O on requests
model = load_trained_model()


# ============================================================================
# APPLICATION ROUTES
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    """
    Serve the main Student Score Predictor frontend dashboard application.
    """
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify backend service, database, and model readiness.
    """
    db_connected = False
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1")
            db_connected = True
    except Exception:
        db_connected = False

    return jsonify({
        "status": "ok",
        "service": "Student Score Predictor API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "database_connected": db_connected,
        "llm_configured": is_llm_configured()
    }), 200



@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Prediction REST API endpoint.
    
    Order of execution:
    1. Check and parse incoming JSON payload
    2. Validate input fields, types, and allowable ranges
    3. Feed inputs to trained Scikit-learn model
    4. Save successful prediction to SQLite database
    5. Return prediction result and database record ID as JSON
    """
    global model

    # Ensure model is available; attempt reloading if initially unready
    if model is None:
        model = load_trained_model()
        if model is None:
            return jsonify({
                "success": False,
                "error": "Prediction model is currently unavailable."
            }), 500

    # 1. Validate JSON presence and Content-Type header
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request must be JSON with Content-Type: application/json"
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Malformed JSON payload in request body"
        }), 400

    # 2. Check for missing required fields
    for field in VALIDATION_RULES.keys():
        if field not in data:
            return jsonify({
                "success": False,
                "error": f"Missing required field: {field}"
            }), 400

    # 3. Validate data types and ranges
    validated_values = {}
    for field, rules in VALIDATION_RULES.items():
        val = data[field]

        # Booleans inherit from int in Python, so exclude them explicitly
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return jsonify({
                "success": False,
                "error": f"Field '{field}' must be a valid number"
            }), 400

        # Reject NaN or Infinity
        if math.isnan(val) or math.isinf(val):
            return jsonify({
                "success": False,
                "error": f"Field '{field}' contains an invalid numerical value (NaN/Inf)"
            }), 400

        # Validate range constraints
        if val < rules["min"] or val > rules["max"]:
            return jsonify({
                "success": False,
                "error": rules["error_msg"]
            }), 400

        validated_values[field] = float(val)

    # 4. Prepare feature DataFrame in exact training order and column names
    input_features = pd.DataFrame([{
        "study_hours": validated_values["study_hours"],
        "previous_score": validated_values["previous_score"],
        "attendance": validated_values["attendance"],
        "assignment_score": validated_values["assignment_score"]
    }])

    # 5. Run real ML model inference
    try:
        raw_prediction = float(model.predict(input_features)[0])
        # Apply sensible boundary clamping for academic scores (0.0 to 100.0)
        predicted_score = round(max(0.0, min(100.0, raw_prediction)), 1)
    except Exception as exc:
        app.logger.error("Inference execution failed: %s", exc)
        return jsonify({
            "success": False,
            "error": "Failed to generate prediction from the model."
        }), 500

    # 6. Save successful prediction to SQLite database
    try:
        record_id = save_prediction(
            study_hours=validated_values["study_hours"],
            previous_score=validated_values["previous_score"],
            attendance=validated_values["attendance"],
            assignment_score=validated_values["assignment_score"],
            predicted_score=predicted_score
        )
    except Exception as exc:
        app.logger.error("Database save failed for prediction: %s", exc)
        return jsonify({
            "success": False,
            "error": "Unable to save prediction to database."
        }), 500

    # 7. Return standardized JSON response
    return jsonify({
        "success": True,
        "predicted_score": predicted_score,
        "id": record_id
    }), 200


@app.route("/api/predictions", methods=["GET"])
def list_predictions():
    """
    Retrieve previous prediction records from SQLite, ordered newest first.
    Optional query parameter: ?limit=N (defaults to 50, maximum 100).
    """
    limit_param = request.args.get("limit", default=50)
    try:
        limit = int(limit_param)
        if limit <= 0:
            limit = 50
        limit = min(limit, 100)
    except (ValueError, TypeError):
        limit = 50

    try:
        history = get_predictions(limit=limit)
        return jsonify({
            "success": True,
            "predictions": history
        }), 200
    except Exception as exc:
        app.logger.error("Failed to fetch prediction history: %s", exc)
        return jsonify({
            "success": False,
            "error": "Unable to load prediction history."
        }), 500


@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    """
    Retrieve summary statistics calculated over all saved predictions from SQLite.
    Returns total_predictions, average_score, highest_score, lowest_score.
    """
    try:
        stats = get_prediction_statistics()
        return jsonify({
            "success": True,
            **stats
        }), 200
    except Exception as exc:
        app.logger.error("Failed to fetch prediction statistics: %s", exc)
        return jsonify({
            "success": False,
            "error": "Unable to load dashboard statistics."
        }), 500


@app.route("/api/ai-advice", methods=["POST"])
def get_ai_advice():
    """
    Generate personalized study advice and explanation via LLM.
    
    Order of execution:
    1. Validate Content-Type and parse JSON
    2. Check presence of all required fields:
       study_hours, previous_score, attendance, assignment_score, predicted_score
    3. Validate numerical types and ranges
    4. Verify predicted_score against ML model output to prevent tampered scores
    5. Call isolated LLM service
    6. Return clean JSON response
    """
    # 1. Validate JSON presence and Content-Type header
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request must be JSON with Content-Type: application/json"
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Malformed JSON payload in request body"
        }), 400

    # 2. Check for missing required fields
    required_fields = ["study_hours", "previous_score", "attendance", "assignment_score", "predicted_score"]
    for field in required_fields:
        if field not in data:
            return jsonify({
                "success": False,
                "error": f"Missing required field: {field}"
            }), 400

    # 3. Validate types and ranges
    validated_values = {}
    for field in ["study_hours", "previous_score", "attendance", "assignment_score"]:
        rules = VALIDATION_RULES[field]
        val = data[field]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return jsonify({
                "success": False,
                "error": f"Field '{field}' must be a valid number"
            }), 400
        if math.isnan(val) or math.isinf(val):
            return jsonify({
                "success": False,
                "error": f"Field '{field}' contains an invalid numerical value (NaN/Inf)"
            }), 400
        if val < rules["min"] or val > rules["max"]:
            return jsonify({
                "success": False,
                "error": rules["error_msg"]
            }), 400
        validated_values[field] = float(val)

    # Validate predicted_score specifically
    raw_pred = data["predicted_score"]
    if isinstance(raw_pred, bool) or not isinstance(raw_pred, (int, float)):
        return jsonify({
            "success": False,
            "error": "Field 'predicted_score' must be a valid number"
        }), 400
    if math.isnan(raw_pred) or math.isinf(raw_pred):
        return jsonify({
            "success": False,
            "error": "Field 'predicted_score' contains an invalid numerical value (NaN/Inf)"
        }), 400
    if raw_pred < 0.0 or raw_pred > 100.0:
        return jsonify({
            "success": False,
            "error": "Predicted score must be between 0 and 100."
        }), 400

    predicted_score = float(raw_pred)

    # 4. Validate predicted_score consistency against ML model
    global model
    if model is None:
        model = load_trained_model()
    if model is not None:
        try:
            input_df = pd.DataFrame([{
                "study_hours": validated_values["study_hours"],
                "previous_score": validated_values["previous_score"],
                "attendance": validated_values["attendance"],
                "assignment_score": validated_values["assignment_score"]
            }])
            expected_score = round(max(0.0, min(100.0, float(model.predict(input_df)[0]))), 1)
            # Allow a small numerical tolerance (+/- 1.5 points)
            if abs(expected_score - predicted_score) > 1.5:
                return jsonify({
                    "success": False,
                    "error": f"Predicted score {predicted_score} is inconsistent with the model calculation ({expected_score}) for the given inputs."
                }), 400
        except Exception as exc:
            app.logger.warning("Could not cross-check prediction against model: %s", exc)

    # 5. Call LLM service to generate personalized study advice
    success, result_payload, status_code = generate_study_advice(validated_values, predicted_score)
    return jsonify(result_payload), status_code



@app.errorhandler(404)
def not_found_error(error):
    """
    Handle 404 errors with JSON for API routes and template fallback for others.
    """
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "API endpoint not found"
        }), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 internal errors safely without exposing Python stack traces.
    """
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "Internal server error. Please try again later."
        }), 500
    return "An internal server error occurred.", 500


if __name__ == "__main__":
    # Ensure debug mode is enabled only in development environments, never in production
    flask_env = os.environ.get("FLASK_ENV", "development").lower()
    flask_debug = os.environ.get("FLASK_DEBUG", "").lower() in ("true", "1", "yes")
    debug_mode = flask_debug or (flask_env == "development")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=debug_mode)

