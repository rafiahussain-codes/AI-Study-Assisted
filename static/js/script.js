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

    // Phase 9 Enhancement: AI Study Assistant Chatbot & Notes RAG Elements
    const chatSection = document.getElementById("chatSection");
    const chatModelIndicator = document.getElementById("chatModelIndicator");
    const chatResetBtn = document.getElementById("chatResetBtn");
    const notesFileInput = document.getElementById("notesFileInput");
    const notesDropzone = document.getElementById("notesDropzone");
    const uploadProgress = document.getElementById("uploadProgress");
    const uploadProgressFill = document.getElementById("uploadProgressFill");
    const uploadStatusMsg = document.getElementById("uploadStatusMsg");
    const uploadedDocsContainer = document.getElementById("uploadedDocsContainer");
    const uploadedDocsList = document.getElementById("uploadedDocsList");
    const quickPromptsRow = document.getElementById("quickPromptsRow");
    const quickChipScore = document.getElementById("quickChipScore");
    const chatMessagesBox = document.getElementById("chatMessagesBox");
    const chatTypingIndicator = document.getElementById("chatTypingIndicator");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatSendBtn = document.getElementById("chatSendBtn");
    const chatSendIcon = document.getElementById("chatSendIcon");
    const chatSendSpinner = document.getElementById("chatSendSpinner");

    // Session management for Chat & RAG
    let currentChatSessionId = sessionStorage.getItem("study_assistant_session_id");
    if (!currentChatSessionId) {
        currentChatSessionId = "sess_" + Date.now().toString(36) + "_" + Math.random().toString(36).substring(2, 8);
        sessionStorage.setItem("study_assistant_session_id", currentChatSessionId);
    }

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
    // 7. Phase 9: AI Study Advice Action (Bridged with Interactive Chatbot)
    // ========================================================================

    if (getAiAdviceBtn) {
        getAiAdviceBtn.addEventListener("click", async () => {
            if (!currentPredictionContext) {
                return;
            }

            // Enter loading state on the button
            getAiAdviceBtn.disabled = true;
            if (aiBtnSpinner) aiBtnSpinner.classList.remove("hidden");
            if (aiBtnText) aiBtnText.textContent = "Generating advice...";

            // Hide previous messages while loading
            if (aiAdviceArea) aiAdviceArea.classList.add("hidden");
            if (aiAdviceError) aiAdviceError.classList.add("hidden");

            // Also smoothly scroll to the chat section so the student can converse
            if (chatSection) {
                chatSection.scrollIntoView({ behavior: "smooth", block: "start" });
            }

            try {
                // 1. First trigger the legacy endpoint for backward compatibility
                const response = await fetch("/api/ai-advice", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(currentPredictionContext)
                });

                const data = await response.json().catch(() => null);

                if (response.ok && data && data.success) {
                    aiAdviceText.textContent = data.advice;
                    if (data.model && aiModelBadge) {
                        aiModelBadge.textContent = data.model;
                    }
                    aiAdviceError.classList.add("hidden");
                    aiAdviceArea.classList.remove("hidden");

                    // 2. Also inject this into the interactive chatbot so the student can follow up!
                    const advicePrompt = `Please explain my predicted score of ${currentPredictionContext.predicted_score}/100 in detail and give me personalized study recommendations based on my daily study hours (${currentPredictionContext.study_hours} hrs), previous score (${currentPredictionContext.previous_score}), attendance (${currentPredictionContext.attendance}%), and assignment score (${currentPredictionContext.assignment_score}).`;
                    appendChatMessage("user", advicePrompt);
                    appendChatMessage("assistant", data.advice, [], data.model);
                } else {
                    const errorMsg = (data && data.error)
                        ? data.error
                        : "AI advice is currently unavailable. Please try again later.";

                    aiAdviceText.textContent = "";
                    aiAdviceError.textContent = errorMsg;
                    aiAdviceError.classList.remove("hidden");
                    aiAdviceArea.classList.remove("hidden");

                    appendChatMessage("assistant", `⚠️ **Could not generate score advice:** ${errorMsg}`);
                }

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
    // 8. Phase 9: AI Study Assistant Chatbot & Notes RAG Implementation
    // ========================================================================

    /**
     * Escape HTML special characters for XSS safety
     */
    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    /**
     * Convert markdown text to clean, structured HTML
     */
    function formatMarkdown(text) {
        if (!text) return "";

        // First escape HTML entities
        let escaped = escapeHtml(text);

        // Code blocks: ```code```
        escaped = escaped.replace(/```([\s\S]*?)```/g, (match, code) => {
            return `<pre><code>${code.trim()}</code></pre>`;
        });

        // Inline code: `code`
        escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold: **bold** or __bold__
        escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic: *italic* or _italic_
        escaped = escaped.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Markdown headings
        escaped = escaped.replace(/^### (.*$)/gim, '<h4 style="margin: 0.5rem 0 0.25rem 0; font-weight: 600;">$1</h4>');
        escaped = escaped.replace(/^## (.*$)/gim, '<h3 style="margin: 0.6rem 0 0.3rem 0; font-weight: 700;">$1</h3>');

        // Lines and lists
        const lines = escaped.split('\n');
        let inList = false;
        let listType = 'ul';
        const formattedLines = [];

        for (let i = 0; i < lines.length; i++) {
            const rawLine = lines[i];
            const line = rawLine.trim();

            const bulletMatch = line.match(/^[-*•]\s+(.*)$/);
            const numberMatch = line.match(/^(\d+)\.\s+(.*)$/);

            if (bulletMatch) {
                if (!inList || listType !== 'ul') {
                    if (inList) formattedLines.push(`</${listType}>`);
                    formattedLines.push('<ul>');
                    inList = true;
                    listType = 'ul';
                }
                formattedLines.push(`<li>${bulletMatch[1]}</li>`);
            } else if (numberMatch) {
                if (!inList || listType !== 'ol') {
                    if (inList) formattedLines.push(`</${listType}>`);
                    formattedLines.push('<ol>');
                    inList = true;
                    listType = 'ol';
                }
                formattedLines.push(`<li>${numberMatch[2]}</li>`);
            } else {
                if (inList) {
                    formattedLines.push(`</${listType}>`);
                    inList = false;
                }
                if (line.length > 0) {
                    // Don't wrap already block-level tags like pre, h3, h4
                    if (line.startsWith('<pre') || line.startsWith('<h3') || line.startsWith('<h4')) {
                        formattedLines.push(line);
                    } else {
                        formattedLines.push(`<p>${line}</p>`);
                    }
                }
            }
        }

        if (inList) {
            formattedLines.push(`</${listType}>`);
        }

        return formattedLines.join('');
    }

    /**
     * Render expandable source citations box for RAG-derived responses
     */
    function renderCitations(citations) {
        if (!citations || !Array.isArray(citations) || citations.length === 0) {
            return "";
        }

        const count = citations.length;
        const itemsHtml = citations.map(c => `
            <div class="citation-item">
                <div class="citation-header">
                    <span class="citation-file">&#128196; ${escapeHtml(c.filename)}</span>
                    <span class="citation-page">Page/Sec: ${escapeHtml(String(c.page || 1))}</span>
                </div>
                <div class="citation-snippet">"${escapeHtml(c.snippet || '')}"</div>
            </div>
        `).join("");

        return `
            <div class="citations-box">
                <details class="citations-details">
                    <summary class="citations-summary">
                        <span>&#128218; Referenced ${count} section${count > 1 ? 's' : ''} from uploaded notes</span>
                    </summary>
                    <div class="citations-list">
                        ${itemsHtml}
                    </div>
                </details>
            </div>
        `;
    }

    /**
     * Append a message bubble to the chat conversation
     */
    function appendChatMessage(role, text, citations = [], model = "") {
        if (!chatMessagesBox) return;

        const isUser = (role === "user");
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-msg ${isUser ? 'user-msg' : 'assistant-msg'}`;

        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const avatar = isUser ? '&#128100;' : '&#129302;';
        const authorName = isUser ? 'You' : 'Study Assistant';

        const formattedContent = isUser ? `<p>${escapeHtml(text)}</p>` : formatMarkdown(text);
        const citationsHtml = (!isUser && citations && citations.length > 0) ? renderCitations(citations) : '';

        msgDiv.innerHTML = `
            <div class="msg-avatar" aria-hidden="true">${avatar}</div>
            <div class="msg-bubble">
                <div class="msg-author-row">
                    <span class="msg-author">${authorName}</span>
                    <span class="msg-time">${timeStr}</span>
                </div>
                <div class="msg-text">${formattedContent}</div>
                ${citationsHtml}
            </div>
        `;

        chatMessagesBox.appendChild(msgDiv);
        chatMessagesBox.scrollTop = chatMessagesBox.scrollHeight;
    }

    /**
     * Toggle chat loading state and typing indicator
     */
    function setChatLoading(isLoading) {
        if (chatSendBtn) chatSendBtn.disabled = isLoading;
        if (chatSendIcon) chatSendIcon.classList.toggle("hidden", isLoading);
        if (chatSendSpinner) chatSendSpinner.classList.toggle("hidden", !isLoading);
        if (chatTypingIndicator) chatTypingIndicator.classList.toggle("hidden", !isLoading);

        if (isLoading && chatMessagesBox) {
            chatMessagesBox.scrollTop = chatMessagesBox.scrollHeight;
        }
    }

    /**
     * Dispatch user question to /api/chat/message with multi-turn context
     */
    async function sendChatMessage(queryText) {
        if (!queryText || !queryText.trim()) return;
        const textToSend = queryText.trim();

        // 1. Show user message in chat log
        appendChatMessage("user", textToSend);

        // Clear input field and reset height
        if (chatInput) {
            chatInput.value = "";
            chatInput.style.height = "auto";
        }

        // 2. Set loading state
        setChatLoading(true);

        try {
            const payload = {
                session_id: currentChatSessionId,
                message: textToSend,
                prediction_context: currentPredictionContext
            };

            const response = await fetch("/api/chat/message", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json().catch(() => null);

            if (!response.ok || !data || !data.success) {
                const err = (data && data.error)
                    ? data.error
                    : "Unable to reach the AI Study Assistant. Please try again.";
                appendChatMessage("assistant", `⚠️ **Notice:** ${err}`);
                return;
            }

            // Update model badge if present
            if (data.model && chatModelIndicator) {
                chatModelIndicator.textContent = data.model;
            }

            // 3. Render Assistant response with citations
            appendChatMessage("assistant", data.reply, data.citations, data.model);

        } catch (networkError) {
            console.error("Chat request failed:", networkError);
            appendChatMessage("assistant", "⚠️ **Network Error:** Could not connect to the study service. Please verify the Flask server is running.");
        } finally {
            setChatLoading(false);
        }
    }

    /**
     * Display status message for document uploads
     */
    function showUploadStatus(msg, type = "info") {
        if (!uploadStatusMsg) return;
        uploadStatusMsg.textContent = msg;
        uploadStatusMsg.className = `upload-status-msg ${type}`;
        uploadStatusMsg.classList.remove("hidden");

        if (type === "success") {
            setTimeout(() => {
                uploadStatusMsg.classList.add("hidden");
            }, 5000);
        }
    }

    /**
     * Fetch and render active documents indexed for the session
     */
    async function loadUploadedDocuments() {
        if (!uploadedDocsList || !uploadedDocsContainer) return;

        try {
            const res = await fetch(`/api/chat/documents?session_id=${encodeURIComponent(currentChatSessionId)}`);
            const data = await res.json().catch(() => null);

            if (!res.ok || !data || !data.success || !Array.isArray(data.documents)) {
                return;
            }

            const docs = data.documents;
            if (docs.length === 0) {
                uploadedDocsContainer.classList.add("hidden");
                uploadedDocsList.innerHTML = "";
                return;
            }

            uploadedDocsContainer.classList.remove("hidden");
            uploadedDocsList.innerHTML = docs.map(d => `
                <div class="doc-pill" data-doc-id="${escapeHtml(d.doc_id)}">
                    <span class="doc-pill-name" title="${escapeHtml(d.filename)}">&#128196; ${escapeHtml(d.filename)}</span>
                    <span class="doc-pill-chunks">${d.chunk_count} chunk${d.chunk_count > 1 ? 's' : ''}</span>
                    <button type="button" class="doc-pill-del" data-doc-id="${escapeHtml(d.doc_id)}" title="Remove document">&times;</button>
                </div>
            `).join("");

            // Attach delete listeners
            uploadedDocsList.querySelectorAll(".doc-pill-del").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const docId = btn.getAttribute("data-doc-id");
                    if (docId) {
                        await deleteStudyDocument(docId);
                    }
                });
            });

        } catch (err) {
            console.error("Failed to load documents list:", err);
        }
    }

    /**
     * Delete a document from the session vector store
     */
    async function deleteStudyDocument(docId) {
        try {
            const res = await fetch(`/api/chat/documents/${encodeURIComponent(docId)}?session_id=${encodeURIComponent(currentChatSessionId)}`, {
                method: "DELETE"
            });
            const data = await res.json().catch(() => null);
            if (res.ok && data && data.success) {
                loadUploadedDocuments();
                appendChatMessage("assistant", `🗑️ Removed document from active study memory.`);
            }
        } catch (err) {
            console.error("Failed to delete document:", err);
        }
    }

    /**
     * Upload and index a study file (PDF, DOCX, TXT, MD)
     */
    async function uploadStudyFile(file) {
        if (!file) return;

        showUploadStatus(`Uploading and parsing "${file.name}"...`, "info");
        if (uploadProgress) uploadProgress.classList.remove("hidden");
        if (uploadProgressFill) uploadProgressFill.style.width = "35%";

        const formData = new FormData();
        formData.append("file", file);
        formData.append("session_id", currentChatSessionId);

        try {
            if (uploadProgressFill) uploadProgressFill.style.width = "70%";
            const res = await fetch("/api/chat/upload", {
                method: "POST",
                body: formData
            });

            const data = await res.json().catch(() => null);

            if (!res.ok || !data || !data.success) {
                const errMsg = (data && data.error) ? data.error : "Failed to upload file.";
                showUploadStatus(`❌ ${errMsg}`, "error");
                return;
            }

            if (uploadProgressFill) uploadProgressFill.style.width = "100%";
            const doc = data.document;
            showUploadStatus(`✅ Successfully indexed "${doc.filename}" (${doc.chunk_count} chunks, ${doc.char_count} chars)!`, "success");

            // Post notification in chat
            appendChatMessage("assistant", `📄 I have read and indexed **${escapeHtml(doc.filename)}** (${doc.chunk_count} chunks). Ask me questions about it, request a summary, or practice MCQs!`);

            // Refresh document pills
            loadUploadedDocuments();

        } catch (err) {
            console.error("File upload error:", err);
            showUploadStatus("❌ Network error while uploading notes.", "error");
        } finally {
            setTimeout(() => {
                if (uploadProgress) uploadProgress.classList.add("hidden");
                if (uploadProgressFill) uploadProgressFill.style.width = "0%";
            }, 1000);
        }
    }

    // ========================================================================
    // Chat & Dropzone Event Listeners
    // ========================================================================

    // Dropzone Drag & Drop
    if (notesDropzone) {
        notesDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            notesDropzone.classList.add("dragover");
        });

        notesDropzone.addEventListener("dragleave", () => {
            notesDropzone.classList.remove("dragover");
        });

        notesDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            notesDropzone.classList.remove("dragover");
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                uploadStudyFile(files[0]);
            }
        });

        notesDropzone.addEventListener("click", () => {
            if (notesFileInput) notesFileInput.click();
        });
    }

    // File Input change
    if (notesFileInput) {
        notesFileInput.addEventListener("change", () => {
            const files = notesFileInput.files;
            if (files && files.length > 0) {
                uploadStudyFile(files[0]);
                notesFileInput.value = ""; // Reset for re-selection
            }
        });
    }

    // Chat Form Submit
    if (chatForm) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            if (chatInput) {
                sendChatMessage(chatInput.value);
            }
        });
    }

    // Chat Textarea Enter-to-send (Shift+Enter for new line) & Auto-resize
    if (chatInput) {
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage(chatInput.value);
            }
        });

        chatInput.addEventListener("input", () => {
            chatInput.style.height = "auto";
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
        });
    }

    // Quick Action Study Chips
    if (quickPromptsRow) {
        quickPromptsRow.addEventListener("click", (e) => {
            const chip = e.target.closest(".quick-chip");
            if (!chip) return;

            let prompt = chip.getAttribute("data-prompt");
            if (chip.id === "quickChipScore") {
                if (currentPredictionContext) {
                    prompt = `Explain my predicted score of ${currentPredictionContext.predicted_score}/100 and give me personalized study recommendations based on my metrics (${currentPredictionContext.study_hours} hrs study/day, ${currentPredictionContext.previous_score} previous score, ${currentPredictionContext.attendance}% attendance, and ${currentPredictionContext.assignment_score} assignment score).`;
                } else {
                    prompt = "How can a student optimize their study hours, attendance, and assignment consistency to reach a top exam score?";
                }
            }

            if (prompt) {
                sendChatMessage(prompt);
            }
        });
    }

    // Reset Chat / New Conversation Button
    if (chatResetBtn) {
        chatResetBtn.addEventListener("click", async () => {
            try {
                await fetch("/api/chat/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: currentChatSessionId })
                });

                if (chatMessagesBox) {
                    chatMessagesBox.innerHTML = `
                        <div class="chat-msg assistant-msg">
                            <div class="msg-avatar" aria-hidden="true">&#129302;</div>
                            <div class="msg-bubble">
                                <div class="msg-author-row">
                                    <span class="msg-author">Study Assistant</span>
                                    <span class="msg-time">Just now</span>
                                </div>
                                <div class="msg-text">
                                    Conversation reset. How can I help you study next?
                                </div>
                            </div>
                        </div>
                    `;
                }
            } catch (err) {
                console.error("Failed to reset chat session:", err);
            }
        });
    }

    // ========================================================================
    // 9. Initial Dashboard Hydration
    // ========================================================================
    loadStatistics();
    loadPredictionHistory();
    loadUploadedDocuments();
});

