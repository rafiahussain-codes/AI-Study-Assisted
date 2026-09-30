# Student Score Predictor

An end-to-end educational web application that predicts expected student exam performance using a trained Machine Learning regression model, stores prediction history and summary metrics in SQLite, and provides personalized AI-driven study recommendations.

---

## Table of Contents

- [Overview & Problem Statement](#overview--problem-statement)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture & Application Flow](#architecture--application-flow)
- [Machine Learning Approach](#machine-learning-approach)
- [API Endpoints](#api-endpoints)
- [Database (SQLite)](#database-sqlite)
- [AI / LLM Integration](#ai--llm-integration)
- [Environment Variables](#environment-variables)
- [Local Setup & Installation](#local-setup--installation)
- [Running the Application](#running-the-application)
- [Running Automated Tests](#running-automated-tests)
- [Security Notes](#security-notes)
- [Deployment Instructions (Render / Railway)](#deployment-instructions-render--railway)
- [Future Improvements](#future-improvements)

---

## Overview & Problem Statement

### Problem Statement
Students often struggle to know whether their daily study habits, class attendance, and past exam results are sufficient to meet their academic goals. Without clear, data-informed feedback, students can miss early warnings of academic risk or lack actionable advice on how to improve.

### Solution
The **Student Score Predictor** bridges this gap by combining:
1. An empirical Machine Learning regression model that estimates exam performance based on concrete academic inputs.
2. A lightweight SQLite database to track historical predictions and compute summary statistics.
3. An AI study advisor that explains the prediction and provides concrete, realistic study advice.

---

## Features

- **Real-Time Score Prediction:** Instant numerical score prediction (0–100) based on 4 student metrics:
  - Daily Study Hours (0–24 hours/day)
  - Previous Exam Score (0–100)
  - Class Attendance Rate (0–100%)
  - Assignment Score (0–100)
- **Visual Score Bar:** Smoothly animated score progress bar reflecting predicted performance.
- **SQLite History & Analytics:** Automatically logs all predictions with timestamps and calculates live performance summaries (total predictions, class average, highest score, lowest score).
- **Personalized AI Study Advice:** On-demand AI explanation and study recommendations powered by Groq or Google Gemini.
- **Dual-Layer Validation:** Form inputs are validated on the client side for instant feedback and strictly re-validated on the backend for complete security.
- **Fail-Safe Graceful Degradation:** The application functions fully even if the LLM service is offline or unconfigured.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | Responsive user interface, form controls, dynamic dashboard hydration |
| **Backend** | Python 3.10+, Flask 3 | REST API routing, input validation, orchestration, error handling |
| **Machine Learning** | Scikit-learn, Pandas, NumPy, Joblib | Linear regression & Random Forest modeling, prediction inference |
| **Database** | SQLite3 (built-in Python) | Zero-configuration persistence for prediction history and summary statistics |
| **AI / LLM** | Groq API / Google Gemini API | Natural-language explanation and personalized study recommendations |
| **Testing** | Pytest | Automated unit, validation, integration, and security test suite |

---

## Architecture & Application Flow

### High-Level Architecture

```
Student (Browser)
       ↕ [HTML/CSS/JS]
Flask REST API (app.py)
       ├── Input Validation (Server-side)
       ├── ML Inference Engine (models/student_score_model.pkl)
       ├── SQLite Database (database/student_scores.db)
       └── Isolated LLM Service (llm_service.py) -> [Groq / Gemini API]
```

### Application Request Flow

```
1. Student enters metrics in frontend form
2. Client runs local boundary checks & sends JSON:
   POST /api/predict
3. Flask verifies Content-Type, checks ranges & types
4. Scikit-learn model calculates predicted exam score
5. SQLite persists the record (INSERT with parameterized query)
6. Flask returns JSON response { success: true, predicted_score: 82.5, id: 1 }
7. Frontend updates score display, history table, and summary cards
8. (Optional) Student clicks "Explain & Get Study Advice"
9. Frontend sends: POST /api/ai-advice
10. Flask cross-checks prediction consistency, calls LLM service
11. LLM returns encouragement & study recommendations
12. Frontend renders advice safely as text
```

---

## Machine Learning Approach

The machine learning workflow is implemented in `train_model.py`:

1. **Dataset:** A 1,000-sample synthetic academic dataset (`data/student_scores.csv`) simulating realistic correlation between study hours, attendance, assignments, previous scores, and final exam performance.
2. **Feature Engineering:** 4 normalized features: `study_hours`, `previous_score`, `attendance`, `assignment_score`.
3. **Data Splitting:** 80% training set and 20% held-out test set (`random_state=42`).
4. **Model Comparison:** Evaluated `LinearRegression` vs. `RandomForestRegressor`.
5. **Evaluation Metrics:**
   - **MAE (Mean Absolute Error):** Measures average deviation in score points.
   - **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors.
   - **R² Score:** Measures proportion of variance explained by features.
6. **Persistence:** The top-performing model is serialized to `models/student_score_model.pkl` with full metadata in `models/model_metadata.json`.

---

## API Endpoints

| Method | Endpoint | Description | Request Body | Response Codes |
|---|---|---|---|---|
| `GET` | `/` | Serves dashboard HTML interface | None | `200` |
| `GET` | `/api/health` | Health check (DB, model, LLM status) | None | `200` |
| `POST` | `/api/predict` | Runs ML inference & saves prediction | JSON with 4 metrics | `200`, `400`, `500` |
| `GET` | `/api/predictions` | Retrieves recent prediction records | Optional `?limit=N` | `200`, `500` |
| `GET` | `/api/statistics` | Aggregate summary statistics | None | `200`, `500` |
| `POST` | `/api/ai-advice` | Generates personalized study advice | JSON with metrics + score | `200`, `400`, `429`, `502`, `503`, `504` |

---

## Database (SQLite)

The application uses an embedded SQLite database stored at `database/student_scores.db`:

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    study_hours REAL NOT NULL,
    previous_score REAL NOT NULL,
    attendance REAL NOT NULL,
    assignment_score REAL NOT NULL,
    predicted_score REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Safety Features
- **Parameterized Queries:** All insertions and queries use parameter placeholders (`?`) preventing SQL injection.
- **Dynamic Override:** Supports `STUDENT_SCORES_DB_PATH` environment variable for isolated automated testing.

---

## AI / LLM Integration

The LLM integration (`llm_service.py`) provides qualitative academic guidance:

- **Separation of Concerns:** The ML model generates the numerical prediction; the LLM explains it and suggests study strategies. The LLM is explicitly forbidden from recalculating scores.
- **Provider Support:**
  - **Groq Cloud API** (Default model: `openai/gpt-oss-20b`, keys start with `gsk_`)
  - **Google Gemini API** (Default model: `gemini-2.5-flash`, keys start with `AIza`)
- **Resilience:**
  - Handles missing keys (`503 Service Unavailable`)
  - Handles auth failure (`502 Bad Gateway`)
  - Handles rate limits (`429 Too Many Requests`)
  - Handles timeouts (`504 Gateway Timeout`)
  - Handles network/SSL connection failures (`502 Bad Gateway`)
  - Cross-validates incoming score with the ML model to prevent tampered requests.

---

## Environment Variables

Copy `.env.example` to `.env` to configure your environment:

```ini
# Flask environment ('development' or 'production')
FLASK_ENV=development
FLASK_DEBUG=false
PORT=5000

# LLM API Configuration
# Option A: Groq API (keys start with gsk_)
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-20b

# Option B: Google Gemini API (keys start with AIza)
# LLM_API_KEY=your_gemini_api_key_here
# LLM_MODEL=gemini-2.5-flash
```

> **Security Rule:** Never commit the `.env` file to Git. It is automatically ignored in `.gitignore`.

---

## Local Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/student-score-predictor.git
cd student-score-predictor
```

### 2. Create and activate a virtual environment
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env with your LLM_API_KEY
```

### 5. Train the ML model (optional, pre-trained model included)
```bash
python train_model.py
```

---

## Running the Application

Start the Flask development server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Running Automated Tests

Run the full automated test suite with `pytest`:

```bash
pytest -v
```

All 43 unit and integration tests covering prediction boundaries, database persistence, aggregate statistics, LLM error handling, and security will run in isolated temporary databases.

---

## Security Notes

1. **Secret Isolation:** API keys exist only in `.env` and are read into memory at runtime. They are never sent to client browsers or printed in error responses.
2. **Input Validation:** Strict type casting, `NaN`/`Infinity` checks, and numerical domain bounds (e.g. study hours must be 0–24) run on every API request.
3. **SQL Injection Defense:** All SQLite interactions use parameterized queries (`?`). Unsafe string concatenation is prohibited.
4. **Information Disclosure Prevention:** Custom 404 and 500 error handlers return clean, generic JSON error messages on API routes without revealing Python tracebacks or filesystem paths.
5. **Debug Mode Protection:** In production (`FLASK_ENV=production`), Flask debug mode is disabled.
6. **XSS Protection:** Client-side JavaScript injects all LLM and server text via `textContent` rather than `innerHTML`.

---

## Deployment Instructions (Render / Railway)

This application is ready for deployment on platforms like **Render** or **Railway**.

### Deploying on Render (Free Tier Friendly):

1. **Push your code to GitHub** (ensure `.env` is NOT committed).
2. **Create a new Web Service** on [Render Dashboard](https://dashboard.render.com/).
3. Connect your GitHub repository.
4. Configure the service settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (or use the included `Procfile`)
5. **Add Environment Variables** under the "Environment" tab:
   - `FLASK_ENV`: `production`
   - `FLASK_DEBUG`: `false`
   - `LLM_API_KEY`: `<your_active_api_key>`
   - `LLM_MODEL`: `openai/gpt-oss-20b` (or `gemini-2.5-flash`)
6. Click **Deploy Web Service**.

---

## Future Improvements

- User authentication for teachers and students to manage multiple classes.
- Historical trend visualization charts using Chart.js.
- Multi-model ensemble incorporating additional academic and extracurricular features.
- Downloadable PDF performance reports with AI recommendations.
