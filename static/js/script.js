/**
 * Student Score Predictor - Client-side Dashboard Logic
 * Phase 9: AI / LLM Integration
 *
 * Responsibilities:
 * - Capture and validate student academic details from the form
 * - Dispatch POST /api/predict to obtain real ML predictions
 * - Update the Latest Prediction card and animated score bar
 * - Fetch and display aggregate statistics from GET /api/statistics
 * - Fetch and render prediction history from GET /api/predictions
 * - Provide on-demand "Explain & Get Study Advice" via POST /api/ai-advice
 * - Maintain separation of responsibilities:
 *   * ML Model: Numerical score prediction
 *   * LLM: Prediction explanation and personalized study recommendations
 * - Safely render all text (textContent) without raw HTML injection
 */

document.addEventListener("DOMContentLoaded", () => {
    // ========================================================================
    // 1. DOM Element Selections
    // ========================================================================

    // Prediction Form
    const form = document.getElementById("predictionForm");
    const studyHoursInput = document.getElementById("study_hours");
    const previousScoreInput = document.getElementById("previous_score");
    const attendanceInput = document.getElementById("attendance");
    const assignmentScoreInput = document.getElementById("assignment_score");

    // Form Alert & Submit Controls
    const errorArea = document.getElementById("errorArea");
    const errorMessage = document.getElementById("errorMessage");
    const predictButton = document.getElementById("predictButton");
    const btnSpinner = predictButton.querySelector(".btn-spinner");
    const btnText = document.getElementById("btnText");

    // Latest Prediction Card Elements
    const resultArea = document.getElementById("resultArea");
    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const scoreDisplay = document.getElementById("scoreDisplay");
    const scoreValue = document.getElementById("scoreValue");
    const scoreBarWrapper = document.getElementById("scoreBarWrapper");
    const scoreProgressBar = document.getElementById("scoreProgressBar");

    // Phase 9: AI Study Advice Elements
    const getAiAdviceBtn = document.getElementById("getAiAdviceBtn");
    const aiBtnSpinner = document.getElementById("aiBtnSpinner");
    const aiBtnText = document.getElementById("aiBtnText");
    const aiAdviceArea = document.getElementById("aiAdviceArea");
    const aiAdviceText = document.getElementById("aiAdviceText");
    const aiAdviceError = document.getElementById("aiAdviceError");
    const aiModelBadge = document.getElementById("aiModelBadge");

    // Context tracking for the latest valid prediction
    let currentPredictionContext = null;

    // Dashboard Statistics Elements
    const statsStatus = document.getElementById("statsStatus");
    const statTotal = document.getElementById("statTotal");
    const statAverage = document.getElementById("statAverage");
    const statHighest = document.getElementById("statHighest");
    const statLowest = document.getElementById("statLowest");

    // Prediction History Elements
    const historyLoading = document.getElementById("historyLoading");
    const historyTableWrapper = document.getElementById("historyTableWrapper");
    const historyTableBody = document.getElementById("historyTableBody");
    const historyEmpty = document.getElementById("historyEmpty");
    const historyError = document.getElementById("historyError");
    const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");

    // Input fields descriptor array for reusable validation and styling
    const inputFields = [
        { el: studyHoursInput, name: "Daily Study Hours" },
        { el: previousScoreInput, name: "Previous Exam Score" },
        { el: attendanceInput, name: "Attendance Percentage" },
        { el: assignmentScoreInput, name: "Assignment Score" }
    ];

    // ========================================================================
    // 2. UI State Helpers (Errors & Loading)
    // ========================================================================

    /**
     * Display a validation or operational error message
     * @param {string} message - User-friendly error message
     * @param {HTMLInputElement|null} targetInput - Input field that caused the error
     */
    function showError(message, targetInput = null) {
        errorMessage.textContent = message;
        errorArea.classList.remove("hidden");

        // Clear previous input error borders
        inputFields.forEach(item => item.el.classList.remove("input-error"));

        if (targetInput) {
            targetInput.classList.add("input-error");
            targetInput.focus();
        }
    }

    /**
     * Clear any active error message and input highlighting
     */
    function clearError() {
        errorMessage.textContent = "";
        errorArea.classList.add("hidden");
        inputFields.forEach(item => item.el.classList.remove("input-error"));
    }

    /**
     * Toggle the prediction button loading state
     * @param {boolean} isLoading
     */
    function setFormLoading(isLoading) {
        if (isLoading) {
            predictButton.disabled = true;
            btnSpinner.classList.remove("hidden");
            btnText.textContent = "Predicting...";
        } else {
            predictButton.disabled = false;
            btnSpinner.classList.add("hidden");
            btnText.textContent = "Predict Score";
        }
    }

    /**
     * Validate client-side input fields before dispatching API request
     * @returns {Object|null} Validated data payload or null if validation fails
     */
    function validateForm() {
        clearError();

        // 1. Check for empty inputs
        for (const { el, name } of inputFields) {
            if (el.value.trim() === "") {
                showError(`Please enter a value for ${name}.`, el);
                return null;
            }
        }

        // 2. Parse numbers
        const studyHours = parseFloat(studyHoursInput.value);
        const previousScore = parseFloat(previousScoreInput.value);
        const attendance = parseFloat(attendanceInput.value);
        const assignmentScore = parseFloat(assignmentScoreInput.value);

        // 3. Validate Study Hours (0 to 24)
        if (isNaN(studyHours) || studyHours < 0 || studyHours > 24) {
            showError("Study hours cannot be negative and must be between 0 and 24 hours per day.", studyHoursInput);
            return null;
        }

        // 4. Validate Previous Exam Score (0 to 100)
        if (isNaN(previousScore) || previousScore < 0 || previousScore > 100) {
            showError("Previous score must be between 0 and 100.", previousScoreInput);
            return null;
        }

        // 5. Validate Attendance Percentage (0 to 100)
        if (isNaN(attendance) || attendance < 0 || attendance > 100) {
            showError("Attendance must be between 0 and 100%.", attendanceInput);
            return null;
        }

        // 6. Validate Assignment Score (0 to 100)
        if (isNaN(assignmentScore) || assignmentScore < 0 || assignmentScore > 100) {
            showError("Assignment score must be between 0 and 100.", assignmentScoreInput);
            return null;
        }

        return {
            study_hours: studyHours,
            previous_score: previousScore,
            attendance: attendance,
            assignment_score: assignmentScore
        };
    }

    // Clear error style as soon as user types into an invalid field
    inputFields.forEach(({ el }) => {
        el.addEventListener("input", () => {
            if (el.classList.contains("input-error")) {
                clearError();
            }
        });
    });

    // ========================================================================
    // 3. Latest Prediction Display
    // ========================================================================

    /**
     * Render the latest prediction score and animate the score progress bar
     * @param {number} predictedScore
     * @param {Object} studentPayload
     */
    function displayPrediction(predictedScore, studentPayload) {
        const formattedScore = Number(predictedScore).toFixed(1);

        // Update score text
        scoreValue.textContent = formattedScore;

        // Calculate clamped percentage for the visual bar (0 - 100%)
        const clampedPercentage = Math.max(0, Math.min(100, Number(predictedScore)));
        scoreProgressBar.style.width = `${clampedPercentage}%`;
        scoreBarWrapper.setAttribute("aria-valuenow", formattedScore);

        // Store context for on-demand AI explanation
        currentPredictionContext = {
            study_hours: studentPayload.study_hours,
            previous_score: studentPayload.previous_score,
            attendance: studentPayload.attendance,
            assignment_score: studentPayload.assignment_score,
            predicted_score: Number(predictedScore)
        };

        // Reset previous AI advice area for the new prediction
        if (aiAdviceArea) {
            aiAdviceArea.classList.add("hidden");
            aiAdviceText.textContent = "";
            if (aiAdviceError) {
                aiAdviceError.classList.add("hidden");
                aiAdviceError.textContent = "";
            }
        }
        if (getAiAdviceBtn) {
            getAiAdviceBtn.disabled = false;
        }

        // Transition card state
        resultArea.classList.add("has-result");
        resultPlaceholder.classList.add("hidden");
        scoreDisplay.classList.remove("hidden");
    }

    // ========================================================================
    // 4. Dashboard Statistics (GET /api/statistics)
    // ========================================================================

    /**
     * Fetch aggregated statistics from SQLite via GET /api/statistics
     */
    async function loadStatistics() {
        if (statsStatus) {
            statsStatus.textContent = "Loading statistics...";
        }

        try {
            const response = await fetch("/api/statistics");
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const data = await response.json();
            displayStatistics(data);

            if (statsStatus) {
                statsStatus.textContent = "SQLite Aggregates";
            }
        } catch (err) {
            console.error("Failed to load dashboard statistics:", err);
            if (statsStatus) {
                statsStatus.textContent = "Unable to load dashboard statistics.";
            }
        }
    }

    /**
     * Populate statistical summary cards with real database values
     * @param {Object} data - Payload containing total_predictions, average_score, highest_score, lowest_score
     */
    function displayStatistics(data) {
        if (!data || typeof data.total_predictions === "undefined") {
            return;
        }

        // Total count
        statTotal.textContent = data.total_predictions;

        // Average score
        statAverage.textContent = (data.average_score !== null && data.average_score !== undefined)
            ? Number(data.average_score).toFixed(1)
            : "--";

        // Highest score
        statHighest.textContent = (data.highest_score !== null && data.highest_score !== undefined)
            ? Number(data.highest_score).toFixed(1)
            : "--";

        // Lowest score
        statLowest.textContent = (data.lowest_score !== null && data.lowest_score !== undefined)
            ? Number(data.lowest_score).toFixed(1)
            : "--";
    }

    // ========================================================================
    // 5. Prediction History (GET /api/predictions)
    // ========================================================================

    /**
     * Fetch and render prediction history table from GET /api/predictions
     */
    async function loadPredictionHistory() {
        historyLoading.classList.remove("hidden");
        historyTableWrapper.classList.add("hidden");
        historyEmpty.classList.add("hidden");
        historyError.classList.add("hidden");

        try {
            const response = await fetch("/api/predictions");
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const data = await response.json();
            const predictions = (data && Array.isArray(data.predictions)) ? data.predictions : [];

            historyLoading.classList.add("hidden");
            displayPredictions(predictions);
        } catch (err) {
            console.error("Failed to load prediction history:", err);
            historyLoading.classList.add("hidden");
            historyError.classList.remove("hidden");
        }
    }

    /**
     * Render prediction items into the history table
     * @param {Array} predictions - List of prediction objects from SQLite
     */
    function displayPredictions(predictions) {
        if (!predictions || predictions.length === 0) {
            historyEmpty.classList.remove("hidden");
            historyTableWrapper.classList.add("hidden");
            return;
        }

        historyEmpty.classList.add("hidden");
        historyTableBody.innerHTML = "";

        predictions.forEach(item => {
            const row = document.createElement("tr");

            // Format timestamp (e.g., YYYY-MM-DD HH:MM)
            const formattedDate = item.created_at ? item.created_at.substring(0, 16) : "Just now";

            row.innerHTML = `
                <td>${formattedDate}</td>
                <td>${Number(item.study_hours).toFixed(1)}h</td>
                <td>${Number(item.previous_score).toFixed(1)}</td>
                <td>${Number(item.attendance).toFixed(1)}%</td>
                <td>${Number(item.assignment_score).toFixed(1)}</td>
                <td><span class="history-predicted-badge">${Number(item.predicted_score).toFixed(1)}</span></td>
            `;
            historyTableBody.appendChild(row);
        });

        historyTableWrapper.classList.remove("hidden");
    }

    // Manual refresh button for prediction history and statistics
    if (refreshHistoryBtn) {
        refreshHistoryBtn.addEventListener("click", () => {
            loadPredictionHistory();
            loadStatistics();
        });
    }

    // ========================================================================
    // 6. Form Submission (POST /api/predict)
    // ========================================================================

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        // 1. Run client-side validation
        const payload = validateForm();
        if (!payload) {
            return;
        }

        // 2. Set button loading state
        setFormLoading(true);

        try {
            // 3. Dispatch POST request to Flask backend
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json().catch(() => null);

            if (!response.ok) {
                // Show specific backend validation message if available
                const userMessage = (data && data.error)
                    ? data.error
                    : "Unable to generate prediction. Please try again.";

                showError(userMessage);
                return;
            }

            // 4. Handle successful prediction
            if (data && typeof data.predicted_score !== "undefined") {
                // Display the real ML predicted score and store context
                displayPrediction(data.predicted_score, payload);

                // Auto-refresh history and statistics from SQLite
                loadPredictionHistory();
                loadStatistics();
            } else {
                showError("Unable to generate prediction. Please try again.");
            }
        } catch (networkError) {
            // Network failure or backend unreachable
            console.error("Network error during prediction:", networkError);
            showError("Unable to connect to the prediction service. Please ensure the Flask server is running at http://127.0.0.1:5000.");
        } finally {
            // Restore button state
            setFormLoading(false);
        }
    });

    // ========================================================================
    // 7. Phase 9: AI Study Advice Action (POST /api/ai-advice)
    // ========================================================================

    if (getAiAdviceBtn) {
        getAiAdviceBtn.addEventListener("click", async () => {
            if (!currentPredictionContext) {
                return;
            }

            // Enter loading state
            getAiAdviceBtn.disabled = true;
            if (aiBtnSpinner) aiBtnSpinner.classList.remove("hidden");
            if (aiBtnText) aiBtnText.textContent = "Generating advice...";

            // Hide previous messages while loading
            if (aiAdviceArea) aiAdviceArea.classList.add("hidden");
            if (aiAdviceError) aiAdviceError.classList.add("hidden");

            try {
                const response = await fetch("/api/ai-advice", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(currentPredictionContext)
                });

                const data = await response.json().catch(() => null);

                if (!response.ok || !data || !data.success) {
                    const errorMsg = (data && data.error)
                        ? data.error
                        : "AI advice is currently unavailable. Please try again later.";

                    aiAdviceText.textContent = "";
                    aiAdviceError.textContent = errorMsg;
                    aiAdviceError.classList.remove("hidden");
                    aiAdviceArea.classList.remove("hidden");
                    return;
                }

                // Render LLM explanation safely as plain text (prevents HTML/script injection)
                aiAdviceText.textContent = data.advice;
                if (data.model && aiModelBadge) {
                    aiModelBadge.textContent = data.model;
                }

                aiAdviceError.classList.add("hidden");
                aiAdviceArea.classList.remove("hidden");

            } catch (networkError) {
                console.error("Network error while requesting AI advice:", networkError);
                aiAdviceText.textContent = "";
                aiAdviceError.textContent = "Unable to connect to the AI advice service. Please check your network and ensure the Flask server is running.";
                aiAdviceError.classList.remove("hidden");
                aiAdviceArea.classList.remove("hidden");
            } finally {
                // Restore button state
                getAiAdviceBtn.disabled = false;
                if (aiBtnSpinner) aiBtnSpinner.classList.add("hidden");
                if (aiBtnText) aiBtnText.innerHTML = "&#10024; Explain &amp; Get Study Advice";
            }
        });
    }

    // ========================================================================
    // 8. Initial Dashboard Hydration
    // ========================================================================
    loadStatistics();
    loadPredictionHistory();
});
