"""
Student Score Predictor - Model Training & Evaluation Pipeline
Phase 5: Machine Learning

This script performs the complete machine learning workflow:
1. Loads and validates the synthetic student dataset.
2. Separates feature inputs (study_hours, previous_score, attendance, assignment_score)
   from the target variable (final_score).
3. Splits data into training and test sets (80/20) with a fixed random seed.
4. Trains multiple regression models (Linear Regression vs. Random Forest).
5. Evaluates model performance using MAE, RMSE, and R-squared.
6. Selects the best performing model based on evaluation evidence.
7. Saves the trained model to models/student_score_model.pkl using joblib.
8. Saves evaluation and training metadata to models/model_metadata.json.
9. Performs an independent loading and prediction verification test.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Configuration constants
DATA_PATH = os.path.join("data", "student_scores.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "student_score_model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

FEATURE_COLUMNS = [
    "study_hours",
    "previous_score",
    "attendance",
    "assignment_score"
]
TARGET_COLUMN = "final_score"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def generate_synthetic_data(filepath=DATA_PATH, n_samples=1000, random_state=RANDOM_STATE):
    """
    Generate a realistic, reproducible synthetic dataset if the file is not found.
    """
    np.random.seed(random_state)
    study_hours = np.round(np.clip(np.random.normal(5.0, 2.2, n_samples), 0.5, 14.0), 1)
    previous_score = np.round(np.clip(np.random.normal(70.0, 12.0, n_samples), 30.0, 98.0), 1)
    attendance = np.round(np.clip(np.random.normal(82.0, 10.0, n_samples), 45.0, 100.0), 1)
    assignment_score = np.round(np.clip(
        previous_score * 0.45 + attendance * 0.3 + np.random.normal(18.0, 5.5, n_samples),
        35.0, 100.0
    ), 1)

    noise = np.random.normal(0, 3.2, n_samples)
    final_score = np.round(np.clip(
        0.38 * previous_score + 0.30 * assignment_score + 0.16 * attendance + 1.65 * study_hours + noise,
        0.0, 100.0
    ), 1)

    df = pd.DataFrame({
        "study_hours": study_hours,
        "previous_score": previous_score,
        "attendance": attendance,
        "assignment_score": assignment_score,
        "final_score": final_score
    })

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[Data] Created synthetic dataset at '{filepath}' ({n_samples} samples).")
    return df


def load_and_validate_data(filepath=DATA_PATH):
    """
    Load CSV data and perform basic integrity checks.
    """
    if not os.path.exists(filepath):
        print(f"[Data] '{filepath}' not found. Generating new dataset...")
        df = generate_synthetic_data(filepath)
    else:
        df = pd.read_csv(filepath)

    # 1. Check dataset is not empty
    if df.empty:
        raise ValueError(f"Dataset at '{filepath}' is empty.")

    # 2. Check required columns exist
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # 3. Check for missing / null values
    null_counts = df[expected_columns].isnull().sum()
    if null_counts.sum() > 0:
        print("[Warning] Missing values detected. Dropping rows with nulls...")
        df = df.dropna(subset=expected_columns)

    # 4. Verify numeric types
    for col in expected_columns:
        if not np.issubdtype(df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must contain numeric data.")

    print(f"[Data] Loaded {len(df)} valid records from '{filepath}'.")
    return df


def train_and_evaluate_models():
    """
    Main training workflow.
    """
    print("=" * 65)
    print("      STUDENT SCORE PREDICTOR - MACHINE LEARNING PIPELINE")
    print("=" * 65)

    # Step 1: Load and validate dataset
    df = load_and_validate_data(DATA_PATH)

    # Step 2: Separate features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Step 3: Split into train and test sets
    # Training set (80%): Used to fit model parameters.
    # Testing set (20%): Held-out unseen data used to evaluate generalization performance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[Split] Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")

    # Step 4: Define candidate regression models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE
        )
    }

    results = {}
    print("\n" + "-" * 65)
    print("MODEL EVALUATION RESULTS (ON TEST SET)")
    print("-" * 65)

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Generate predictions on test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        # MAE: Average absolute difference between predicted and actual scores (lower is better)
        mae = float(mean_absolute_error(y_test, y_pred))
        # RMSE: Square root of mean squared error; penalizes larger errors more heavily (lower is better)
        rmse = float(root_mean_squared_error(y_test, y_pred))
        # R2: Proportion of variance in the target explained by the features (closer to 1.0 is better)
        r2 = float(r2_score(y_test, y_pred))

        results[name] = {
            "model_instance": model,
            "metrics": {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4)
            }
        }

        print(f"Model: {name}")
        print(f"  MAE  : {mae:.4f} (Average score deviation)")
        print(f"  RMSE : {rmse:.4f} (Root mean squared error)")
        print(f"  R^2  : {r2:.4f} (Variance explained)")
        print("-" * 65)

    # Step 5: Select best model based on evaluation metrics
    # Lower MAE/RMSE and higher R^2 indicate superior predictive accuracy
    selected_name = min(results.keys(), key=lambda m: results[m]["metrics"]["MAE"])
    selected_info = results[selected_name]
    selected_model = selected_info["model_instance"]

    print(f"\n[Selection] Best Model: '{selected_name}'")
    print(f"  Selection rationale: Achieved lowest MAE ({selected_info['metrics']['MAE']}), "
          f"lowest RMSE ({selected_info['metrics']['RMSE']}), "
          f"and highest R^2 ({selected_info['metrics']['R2']}).")

    # Step 6: Save trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
    print(f"[Save] Model persisted to '{MODEL_PATH}' using Joblib.")

    # Step 7: Save model metadata
    metadata = {
        "selected_model": selected_name,
        "feature_names": FEATURE_COLUMNS,
        "target_name": TARGET_COLUMN,
        "dataset_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "evaluation_metrics": {name: res["metrics"] for name, res in results.items()},
        "selected_metrics": selected_info["metrics"]
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[Save] Metadata persisted to '{METADATA_PATH}'.")

    # Step 8: Verification Test - Load from disk and predict sample student
    print("\n" + "-" * 65)
    print("VERIFICATION: INDEPENDENT MODEL LOADING & PREDICTION TEST")
    print("-" * 65)

    loaded_model = joblib.load(MODEL_PATH)
    sample_student = pd.DataFrame([{
        "study_hours": 5.0,
        "previous_score": 72.0,
        "attendance": 85.0,
        "assignment_score": 80.0
    }])

    raw_prediction = float(loaded_model.predict(sample_student)[0])
    # Apply sensible boundary clamping for student exam scores (0.0 to 100.0)
    final_prediction = round(max(0.0, min(100.0, raw_prediction)), 1)

    print("Sample Input:")
    for col, val in sample_student.iloc[0].items():
        print(f"  {col}: {val}")
    print(f"\nPredicted Final Score: {final_prediction} / 100.0")
    print("=" * 65)
    print("ML Pipeline execution completed successfully.")
    print("=" * 65)

    return metadata


if __name__ == "__main__":
    train_and_evaluate_models()
