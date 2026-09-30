/**
 * EDVORA — AI Study Assistant
 * Modern Dark UI, Document Understanding, Vector RAG & Interactive Tutoring
 */

document.addEventListener("DOMContentLoaded", () => {
    // ========================================================================
    // DOM Element Selectors
    // ========================================================================

    // Navigation & Workspace
    const sidebar = document.getElementById("sidebar");
    const sidebarBackdrop = document.getElementById("sidebarBackdrop");
    const mobileMenuBtn = document.getElementById("mobileMenuBtn");
    const sidebarCloseBtn = document.getElementById("sidebarCloseBtn");
    const newChatBtn = document.getElementById("newChatBtn");
    const resetCurrentChatBtn = document.getElementById("resetCurrentChatBtn");
    const currentSessionTitle = document.getElementById("currentSessionTitle");
    const topbarRagStatus = document.getElementById("topbarRagStatus");
    const topbarRagText = document.getElementById("topbarRagText");
    const modelStatusLabel = document.getElementById("modelStatusLabel");
    const engineStatusPill = document.getElementById("engineStatusPill");

    // Right Context Panel (Inspector)
    const contextPanel = document.getElementById("contextPanel");
    const inspectorBackdrop = document.getElementById("inspectorBackdrop");
    const toggleInspectorBtn = document.getElementById("toggleInspectorBtn");
    const closeInspectorBtn = document.getElementById("closeInspectorBtn");
    const inspectorDocsCount = document.getElementById("inspectorDocsCount");
    const inspectorDocsList = document.getElementById("inspectorDocsList");
    const sourcesCardList = document.getElementById("sourcesCardList");
    const sourcesEmptyState = document.getElementById("sourcesEmptyState");
    const inspectorUploadBtn = document.getElementById("inspectorUploadBtn");

    // Upload Drawer & Dropzone
    const uploadDrawer = document.getElementById("uploadDrawer");
    const toggleUploadZoneBtn = document.getElementById("toggleUploadZoneBtn");
    const closeUploadDrawerBtn = document.getElementById("closeUploadDrawerBtn");
    const notesDropzone = document.getElementById("notesDropzone");
    const notesFileInput = document.getElementById("notesFileInput");
    const dropzoneIdle = document.getElementById("dropzoneIdle");
    const dropzoneProcessing = document.getElementById("dropzoneProcessing");
    const procStepIcon = document.getElementById("procStepIcon");
    const procStageTitle = document.getElementById("procStageTitle");
    const procFilename = document.getElementById("procFilename");
    const stepPillUpload = document.getElementById("stepPillUpload");
    const stepPillProcess = document.getElementById("stepPillProcess");
    const stepPillUnderstand = document.getElementById("stepPillUnderstand");
    const stepPillReady = document.getElementById("stepPillReady");
    const uploadProgressFill = document.getElementById("uploadProgressFill");
    const uploadAlert = document.getElementById("uploadAlert");
    const activeDocsShelf = document.getElementById("activeDocsShelf");
    const shelfPills = document.getElementById("shelfPills");

    // Sidebar Items
    const sidebarDocCount = document.getElementById("sidebarDocCount");
    const sidebarDocList = document.getElementById("sidebarDocList");
    const sidebarDocEmpty = document.getElementById("sidebarDocEmpty");
    const sidebarUploadTrigger = document.getElementById("sidebarUploadTrigger");
    const emptyDocUploadBtn = document.getElementById("emptyDocUploadBtn");
    const sessionsHistoryList = document.getElementById("sessionsHistoryList");
    const clearAllHistoryBtn = document.getElementById("clearAllHistoryBtn");

    // Welcome Screen & Cards
    const welcomeHero = document.getElementById("welcomeHero");
    const welcomeContextCard = document.getElementById("welcomeContextCard");
    const welcomeContextTitle = document.getElementById("welcomeContextTitle");
    const welcomeContextDesc = document.getElementById("welcomeContextDesc");
    const heroUploadTrigger = document.getElementById("heroUploadTrigger");

    // Chat Feed & Composer
    const chatViewport = document.getElementById("chatViewport");
    const messagesContainer = document.getElementById("messagesContainer");
    const aiTypingState = document.getElementById("aiTypingState");
    const typingStatusText = document.getElementById("typingStatusText");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatSendBtn = document.getElementById("chatSendBtn");
    const sendIcon = document.getElementById("sendIcon");
    const sendSpinner = document.getElementById("sendSpinner");
    const composerAttachBtn = document.getElementById("composerAttachBtn");
    const composerQuickChips = document.getElementById("composerQuickChips");

    // Settings Modal
    const openSettingsBtn = document.getElementById("openSettingsBtn");
    const settingsModal = document.getElementById("settingsModal");
    const closeSettingsModalBtn = document.getElementById("closeSettingsModalBtn");
    const modalModelInfo = document.getElementById("modalModelInfo");
    const clearLocalDataBtn = document.getElementById("clearLocalDataBtn");

    // ========================================================================
    // State Management
    // ========================================================================
    const SESSION_STORAGE_KEY = "edvora_active_session_id";
    const RECENT_SESSIONS_KEY = "edvora_recent_sessions";
    
    let currentSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
    if (!currentSessionId) {
        currentSessionId = "edvora-" + Math.random().toString(36).substring(2, 10);
        localStorage.setItem(SESSION_STORAGE_KEY, currentSessionId);
    }

    let activeDocuments = [];
    let latestRetrievedCitations = [];
    let isGenerating = false;
    let lastUserQuery = "";

    // ========================================================================
    // 1. Session Lifecycle & Persistence
    // ========================================================================

    function getLocalRecentSessions() {
        try {
            return JSON.parse(localStorage.getItem(RECENT_SESSIONS_KEY)) || [];
        } catch {
            return [];
        }
    }

    function saveLocalRecentSessions(sessions) {
        try {
            localStorage.setItem(RECENT_SESSIONS_KEY, JSON.stringify(sessions));
        } catch (e) {
            console.warn("Storage save error:", e);
        }
    }

    function recordCurrentSessionInRecents(title = "New Study Session") {
        let recents = getLocalRecentSessions();
        const existingIdx = recents.findIndex(s => s.id === currentSessionId);
        const item = {
            id: currentSessionId,
            title: title,
            updatedAt: Date.now()
        };

        if (existingIdx >= 0) {
            recents[existingIdx].title = title;
            recents[existingIdx].updatedAt = Date.now();
        } else {
            recents.unshift(item);
        }

        recents = recents.slice(0, 15);
        saveLocalRecentSessions(recents);
        renderRecentSessionsList();
    }

    function renderRecentSessionsList() {
        if (!sessionsHistoryList) return;
        const recents = getLocalRecentSessions();

        if (recents.length === 0) {
            sessionsHistoryList.innerHTML = `
                <div style="font-size:0.75rem; color:var(--text-muted); padding:6px 8px;">No previous sessions yet.</div>
            `;
            return;
        }

        sessionsHistoryList.innerHTML = recents.map(s => `
            <div class="session-history-item ${s.id === currentSessionId ? 'active' : ''}" data-session-id="${escapeHtml(s.id)}" title="${escapeHtml(s.title)}">
                <span class="session-item-title">${escapeHtml(s.title)}</span>
            </div>
        `).join("");

        sessionsHistoryList.querySelectorAll(".session-history-item").forEach(el => {
            el.addEventListener("click", () => {
                const targetId = el.getAttribute("data-session-id");
                if (targetId && targetId !== currentSessionId) {
                    switchToSession(targetId);
                }
            });
        });
    }

    async function switchToSession(targetId) {
        currentSessionId = targetId;
        localStorage.setItem(SESSION_STORAGE_KEY, targetId);
        messagesContainer.innerHTML = "";
        latestRetrievedCitations = [];
        renderInspectorSources();
        closeSidebar();

        const recents = getLocalRecentSessions();
        const found = recents.find(s => s.id === targetId);
        if (currentSessionTitle) {
            currentSessionTitle.textContent = found ? found.title : "Study Session";
        }

        await loadHistory();
        await loadUploadedDocuments();
        renderRecentSessionsList();
    }

    async function createNewSession(customTitle = "New Study Session") {
        const newId = "edvora-" + Math.random().toString(36).substring(2, 10);
        currentSessionId = newId;
        localStorage.setItem(SESSION_STORAGE_KEY, newId);

        messagesContainer.innerHTML = "";
        latestRetrievedCitations = [];
        renderInspectorSources();

        if (currentSessionTitle) {
            currentSessionTitle.textContent = customTitle;
        }
        if (welcomeHero) {
            welcomeHero.classList.remove("hidden");
        }

        recordCurrentSessionInRecents(customTitle);
        await loadUploadedDocuments();
        closeSidebar();

        if (chatInput) {
            chatInput.focus();
        }
    }

    // ========================================================================
    // 2. Chat History & Message Rendering
    // ========================================================================

    async function loadHistory() {
        try {
            const res = await fetch(`/api/chat/history?session_id=${encodeURIComponent(currentSessionId)}`);
            const data = await res.json().catch(() => null);

            if (!res.ok || !data || !Array.isArray(data.messages)) {
                return;
            }

            messagesContainer.innerHTML = "";
            if (data.messages.length > 0) {
                if (welcomeHero) welcomeHero.classList.add("hidden");
                data.messages.forEach(msg => {
                    renderMessageBubble(msg.role, msg.content, msg.citations || [], msg.timestamp);
                });
                chatViewport.scrollTop = chatViewport.scrollHeight;
            } else {
                if (welcomeHero) welcomeHero.classList.remove("hidden");
            }
        } catch (err) {
            console.warn("Failed to load history:", err);
        }
    }

    function renderMessageBubble(role, content, citations = [], timestamp = null) {
        if (!messagesContainer) return;
        if (welcomeHero) welcomeHero.classList.add("hidden");

        const isUser = (role === "user");
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-message ${isUser ? 'user-chat-turn' : 'ai-chat-turn'}`;

        const now = timestamp ? new Date(timestamp * 1000) : new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        if (isUser) {
            msgDiv.innerHTML = `
                <div class="user-avatar" aria-hidden="true">👤</div>
                <div class="message-wrapper">
                    <div class="message-header-bar">
                        <span class="message-sender">You</span>
                        <span class="message-time">${timeStr}</span>
                    </div>
                    <div class="user-message-card">
                        ${escapeHtml(content)}
                    </div>
                </div>
            `;
        } else {
            // Update latest retrieved citations for Inspector panel
            if (citations && citations.length > 0) {
                latestRetrievedCitations = citations;
                renderInspectorSources();
            }

            // RAG Banner
            let ragBannerHtml = "";
            if (citations && citations.length > 0) {
                const count = citations.length;
                const sourceTags = citations.map(c => `
                    <span class="rag-tag-item" title="${escapeHtml(c.filename)}">
                        📄 ${escapeHtml(c.filename)} (P.${c.page})
                    </span>
                `).join("");

                ragBannerHtml = `
                    <div class="rag-context-banner">
                        <div class="rag-banner-left">
                            <span>⚡ Using your knowledge base</span>
                            <span style="color:var(--text-muted); font-weight:normal;">• ${count} relevant section${count > 1 ? 's' : ''} found</span>
                        </div>
                        <div class="rag-source-tags">
                            ${sourceTags}
                        </div>
                    </div>
                `;
            }

            // Markdown formatting
            const formattedBody = formatMarkdown(content);

            // Citations accordion
            let citationsAccordionHtml = "";
            if (citations && citations.length > 0) {
                const count = citations.length;
                const items = citations.map(c => `
                    <div class="citation-item">
                        <div class="citation-header">
                            <span>📄 ${escapeHtml(c.filename)} — Page ${escapeHtml(String(c.page || 1))}</span>
                            <span style="color:var(--accent-cyan);">${Math.round((c.score || 0) * 100)}% match</span>
                        </div>
                        <div class="citation-snippet">"${escapeHtml(c.snippet || '')}"</div>
                    </div>
                `).join("");

                citationsAccordionHtml = `
                    <div class="citations-box">
                        <details class="citations-details">
                            <summary>
                                <span>📚 View ${count} cited source excerpt${count > 1 ? 's' : ''}</span>
                            </summary>
                            <div class="citations-list">
                                ${items}
                            </div>
                        </details>
                    </div>
                `;
            }

            msgDiv.innerHTML = `
                <div class="edvora-avatar" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none" width="16" height="16">
                        <path d="M12 2L14.2 9.8L22 12L14.2 14.2L12 22L9.8 14.2L2 12L9.8 9.8L12 2Z" fill="white"/>
                    </svg>
                </div>
                <div class="message-wrapper">
                    <div class="message-header-bar">
                        <span class="message-sender">Edvora</span>
                        <span class="message-model-tag">AI Tutor</span>
                        <span class="message-time">${timeStr}</span>
                    </div>
                    
                    <div class="ai-message-card">
                        ${ragBannerHtml}
                        <div class="ai-rendered-content">${formattedBody}</div>
                        ${citationsAccordionHtml}
                    </div>

                    <div class="message-actions-bar">
                        <button type="button" class="btn-msg-action btn-copy-msg" title="Copy answer">
                            <svg viewBox="0 0 20 20" fill="currentColor" width="13" height="13"><path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z"/><path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 2H9a3 3 0 01-3-2z"/></svg>
                            <span class="copy-lbl">Copy</span>
                        </button>
                        <button type="button" class="btn-msg-action btn-regen-msg" title="Regenerate this response">
                            <svg viewBox="0 0 20 20" fill="currentColor" width="13" height="13"><path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"/></svg>
                            <span>Regenerate</span>
                        </button>
                        ${citations && citations.length > 0 ? `
                        <button type="button" class="btn-msg-action btn-inspect-sources" title="Open source inspector">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>
                            <span>Inspect Sources</span>
                        </button>
                        ` : ''}
                    </div>
                </div>
            `;

            // Attach Copy Answer listener
            const copyBtn = msgDiv.querySelector(".btn-copy-msg");
            if (copyBtn) {
                copyBtn.addEventListener("click", () => {
                    navigator.clipboard.writeText(content).then(() => {
                        const lbl = copyBtn.querySelector(".copy-lbl");
                        if (lbl) lbl.textContent = "Copied!";
                        setTimeout(() => {
                            if (lbl) lbl.textContent = "Copy";
                        }, 2000);
                    }).catch(() => {});
                });
            }

            // Attach Regenerate listener
            const regenBtn = msgDiv.querySelector(".btn-regen-msg");
            if (regenBtn) {
                regenBtn.addEventListener("click", () => {
                    if (lastUserQuery && !isGenerating) {
                        sendQuery(lastUserQuery);
                    }
                });
            }

            // Attach Inspect Sources listener
            const inspectBtn = msgDiv.querySelector(".btn-inspect-sources");
            if (inspectBtn) {
                inspectBtn.addEventListener("click", () => {
                    openInspectorTab("tabSources");
                });
            }

            // Attach Copy Code listeners for any code blocks
            msgDiv.querySelectorAll(".btn-copy-code").forEach(btn => {
                btn.addEventListener("click", () => {
                    const code = btn.getAttribute("data-code") || "";
                    navigator.clipboard.writeText(decodeURIComponent(code)).then(() => {
                        btn.textContent = "Copied!";
                        setTimeout(() => { btn.textContent = "Copy"; }, 2000);
                    });
                });
            });
        }

        messagesContainer.appendChild(msgDiv);
        chatViewport.scrollTop = chatViewport.scrollHeight;
    }

    // ========================================================================
    // 3. User Message Dispatch & AI Generation
    // ========================================================================

    async function sendQuery(queryText) {
        if (!queryText || !queryText.trim() || isGenerating) return;
        const textToSend = queryText.trim();
        lastUserQuery = textToSend;

        // Render User Message
        renderMessageBubble("user", textToSend);

        // Update Session Title if fresh
        const recents = getLocalRecentSessions();
        const cur = recents.find(s => s.id === currentSessionId);
        if (!cur || cur.title === "New Study Session") {
            const derivedTitle = textToSend.length > 34 ? textToSend.substring(0, 34) + "..." : textToSend;
            if (currentSessionTitle) currentSessionTitle.textContent = derivedTitle;
            recordCurrentSessionInRecents(derivedTitle);
        }

        // Reset Input
        if (chatInput) {
            chatInput.value = "";
            chatInput.style.height = "auto";
        }

        // Set Loading State
        setGeneratingState(true);

        try {
            const res = await fetch("/api/chat/message", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    message: textToSend
                })
            });

            const data = await res.json().catch(() => null);

            if (!res.ok || !data || !data.success) {
                const errMsg = (data && data.error)
                    ? data.error
                    : "Unable to contact Edvora study service. Please verify your connection.";
                renderMessageBubble("assistant", `⚠️ **Notice:** ${errMsg}`);
                return;
            }

            renderMessageBubble("assistant", data.reply, data.citations || []);

        } catch (err) {
            console.error("Message send error:", err);
            renderMessageBubble("assistant", "⚠️ **Connection Error:** Could not communicate with Edvora's study service. Please ensure the local server is running.");
        } finally {
            setGeneratingState(false);
        }
    }

    function setGeneratingState(generating) {
        isGenerating = generating;
        if (chatSendBtn) chatSendBtn.disabled = generating;
        if (sendIcon) sendIcon.classList.toggle("hidden", generating);
        if (sendSpinner) sendSpinner.classList.toggle("hidden", !generating);
        if (aiTypingState) aiTypingState.classList.toggle("hidden", !generating);

        if (generating) {
            const statusPhrases = [
                "Edvora is analyzing knowledge base & drafting answer...",
                "Retrieving relevant document sections with vector RAG...",
                "Synthesizing concepts and formulating explanation...",
                "Formatting pedagogical breakdown..."
            ];
            if (typingStatusText) {
                typingStatusText.textContent = statusPhrases[Math.floor(Math.random() * statusPhrases.length)];
            }
            if (chatViewport) {
                chatViewport.scrollTop = chatViewport.scrollHeight;
            }
        }
    }

    // ========================================================================
    // 4. Document Upload & Multi-Stage Processing Workflow
    // ========================================================================

    async function uploadStudyFile(file) {
        if (!file) return;

        // Reset Dropzone UI to processing state
        if (dropzoneIdle) dropzoneIdle.classList.add("hidden");
        if (dropzoneProcessing) dropzoneProcessing.classList.remove("hidden");
        if (uploadAlert) uploadAlert.classList.add("hidden");

        if (procFilename) procFilename.textContent = file.name;
        updateUploadWorkflowStep(1, "Uploading File...", "30%");

        const formData = new FormData();
        formData.append("file", file);
        formData.append("session_id", currentSessionId);

        try {
            // Simulated stage progression for smooth visual feedback
            setTimeout(() => {
                updateUploadWorkflowStep(2, "Extracting Text & Chunking Sections...", "65%");
            }, 600);

            setTimeout(() => {
                updateUploadWorkflowStep(3, "Understanding & Generating Vector Index...", "85%");
            }, 1200);

            const res = await fetch("/api/chat/upload", {
                method: "POST",
                body: formData
            });

            const data = await res.json().catch(() => null);

            if (!res.ok || !data || !data.success) {
                const errMsg = (data && data.error) ? data.error : "Failed to parse study document.";
                showUploadAlert(`❌ ${errMsg}`, "error");
                resetDropzoneIdle();
                return;
            }

            // Stage 4: Ready
            updateUploadWorkflowStep(4, "Ready! Document Fully Indexed", "100%");

            const doc = data.document;
            setTimeout(() => {
                resetDropzoneIdle();
                showUploadAlert(`✅ Successfully indexed "${doc.filename}" (${doc.chunk_count} chunks, ${doc.char_count} chars)!`, "success");
            }, 1200);

            // Notify in chat feed
            renderMessageBubble("assistant", `📄 I have parsed and indexed **${escapeHtml(doc.filename)}** into your Edvora knowledge base (${doc.chunk_count} chunks). Ask me to summarize it, create practice MCQs, or explain any concept!`);

            // Refresh document list
            await loadUploadedDocuments();

        } catch (err) {
            console.error("Upload failed:", err);
            showUploadAlert("❌ Network error occurred while uploading document.", "error");
            resetDropzoneIdle();
        }
    }

    function updateUploadWorkflowStep(stepNum, title, progressWidth) {
        if (procStageTitle) procStageTitle.textContent = title;
        if (uploadProgressFill) uploadProgressFill.style.width = progressWidth;

        const steps = [stepPillUpload, stepPillProcess, stepPillUnderstand, stepPillReady];
        steps.forEach((step, idx) => {
            if (!step) return;
            const currentNum = idx + 1;
            if (currentNum < stepNum) {
                step.className = "step-pill completed";
            } else if (currentNum === stepNum) {
                step.className = "step-pill active";
            } else {
                step.className = "step-pill";
            }
        });

        const icons = ["⚡", "📄", "🧠", "✅"];
        if (procStepIcon) procStepIcon.textContent = icons[stepNum - 1] || "⚡";
    }

    function resetDropzoneIdle() {
        if (dropzoneProcessing) dropzoneProcessing.classList.add("hidden");
        if (dropzoneIdle) dropzoneIdle.classList.remove("hidden");
        if (uploadProgressFill) uploadProgressFill.style.width = "0%";
    }

    function showUploadAlert(msg, type = "info") {
        if (!uploadAlert) return;
        uploadAlert.textContent = msg;
        uploadAlert.className = `upload-alert ${type}`;
        uploadAlert.classList.remove("hidden");

        if (type === "success") {
            setTimeout(() => {
                uploadAlert.classList.add("hidden");
            }, 7000);
        }
    }

    // ========================================================================
    // 5. Knowledge Base Document List Sync & Management
    // ========================================================================

    async function loadUploadedDocuments() {
        try {
            const res = await fetch(`/api/chat/documents?session_id=${encodeURIComponent(currentSessionId)}`);
            const data = await res.json().catch(() => null);

            if (!res.ok || !data || !Array.isArray(data.documents)) {
                return;
            }

            activeDocuments = data.documents;
            const count = activeDocuments.length;

            // 1. Update Top Bar Status Tag
            if (topbarRagText) {
                topbarRagText.textContent = count > 0 
                    ? `Knowledge Base: ${count} Active (${getTotalChunks()} chunks)`
                    : "Upload notes to enable RAG";
            }
            if (topbarRagStatus) {
                topbarRagStatus.querySelector(".rag-dot").style.backgroundColor = count > 0 
                    ? "var(--accent-emerald)" 
                    : "var(--accent-cyan)";
            }

            // 2. Update Sidebar Knowledge Base Section
            if (sidebarDocCount) sidebarDocCount.textContent = String(count);
            if (sidebarDocEmpty) sidebarDocEmpty.classList.toggle("hidden", count > 0);

            if (sidebarDocList) {
                const pillsHtml = activeDocuments.map(d => `
                    <div class="sidebar-doc-pill" data-doc-id="${escapeHtml(d.doc_id)}">
                        <div class="doc-pill-left">
                            <span class="doc-file-icon">${getFileIcon(d.filename)}</span>
                            <span class="doc-file-name" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</span>
                        </div>
                        <div class="doc-pill-right">
                            <span class="doc-chunks-tag">${d.chunk_count}c</span>
                            <button type="button" class="doc-del-btn" data-doc-id="${escapeHtml(d.doc_id)}" title="Remove from knowledge base">&times;</button>
                        </div>
                    </div>
                `).join("");

                // Keep empty state in list container if 0
                if (count > 0) {
                    sidebarDocList.innerHTML = pillsHtml;
                } else if (sidebarDocEmpty) {
                    sidebarDocList.innerHTML = "";
                    sidebarDocList.appendChild(sidebarDocEmpty);
                }
            }

            // 3. Update Upload Drawer Shelf
            if (activeDocsShelf && shelfPills) {
                activeDocsShelf.classList.toggle("hidden", count === 0);
                shelfPills.innerHTML = activeDocuments.map(d => `
                    <div class="sidebar-doc-pill" style="font-size:0.75rem;">
                        <span class="doc-file-icon">${getFileIcon(d.filename)}</span>
                        <span class="doc-file-name" style="max-width:130px;" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</span>
                        <span class="doc-chunks-tag">${d.chunk_count}c</span>
                        <button type="button" class="doc-del-btn" data-doc-id="${escapeHtml(d.doc_id)}" title="Remove">&times;</button>
                    </div>
                `).join("");
            }

            // 4. Update Welcome Hero Context Banner
            if (welcomeContextTitle && welcomeContextDesc) {
                if (count > 0) {
                    welcomeContextTitle.textContent = "Your knowledge base is ready. Ask anything about your material.";
                    welcomeContextDesc.textContent = `${count} document${count > 1 ? 's' : ''} indexed (${getTotalChunks()} chunks). Edvora will retrieve relevant sections for every study question.`;
                } else {
                    welcomeContextTitle.textContent = "Upload your study material to give Edvora context.";
                    welcomeContextDesc.textContent = "Drop lecture slides, textbook chapters, or exam notes. Edvora will extract concepts, index vectors, and retrieve verified source citations.";
                }
            }

            // 5. Update Inspector Documents Tab
            if (inspectorDocsCount) inspectorDocsCount.textContent = `${count} document${count === 1 ? '' : 's'} attached`;
            if (inspectorDocsList) {
                if (count === 0) {
                    inspectorDocsList.innerHTML = `
                        <div class="sources-empty-state" style="padding:20px 10px;">
                            <p class="pane-empty-title" style="font-size:0.84rem;">No documents in memory</p>
                            <p class="pane-empty-desc">Upload study material to see chunks, pages, and token analytics here.</p>
                        </div>
                    `;
                } else {
                    inspectorDocsList.innerHTML = activeDocuments.map(d => `
                        <div class="inspector-doc-item">
                            <div class="inspector-doc-header">
                                <span class="inspector-doc-title" title="${escapeHtml(d.filename)}">${getFileIcon(d.filename)} ${escapeHtml(d.filename)}</span>
                                <button type="button" class="doc-del-btn" data-doc-id="${escapeHtml(d.doc_id)}" title="Remove">&times;</button>
                            </div>
                            <div class="inspector-doc-meta">
                                <span>${d.chunk_count} chunks</span>
                                <span>&bull;</span>
                                <span>${d.page_count || 1} page${(d.page_count || 1) > 1 ? 's' : ''}</span>
                                <span>&bull;</span>
                                <span>${d.char_count} chars</span>
                            </div>
                        </div>
                    `).join("");
                }
            }

            // Attach Delete Listeners across all containers
            document.querySelectorAll(".doc-del-btn").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const docId = btn.getAttribute("data-doc-id");
                    if (docId) {
                        await deleteDocument(docId);
                    }
                });
            });

        } catch (err) {
            console.error("Failed to load documents:", err);
        }
    }

    async function deleteDocument(docId) {
        try {
            const res = await fetch(`/api/chat/documents/${encodeURIComponent(docId)}?session_id=${encodeURIComponent(currentSessionId)}`, {
                method: "DELETE"
            });
            const data = await res.json().catch(() => null);

            if (res.ok && data && data.success) {
                renderMessageBubble("assistant", `🗑️ Removed document from active study memory.`);
                await loadUploadedDocuments();
            }
        } catch (err) {
            console.error("Failed to delete document:", err);
        }
    }

    function getTotalChunks() {
        return activeDocuments.reduce((sum, d) => sum + (d.chunk_count || 0), 0);
    }

    function getFileIcon(filename) {
        if (!filename) return "📄";
        const ext = filename.split(".").pop().toLowerCase();
        if (ext === "pdf") return "📕";
        if (ext === "docx" || ext === "doc") return "📘";
        if (ext === "txt") return "📄";
        if (ext === "md" || ext === "markdown") return "📝";
        return "📑";
    }

    // ========================================================================
    // 6. Right Context Panel: Sources Inspector Tab
    // ========================================================================

    function renderInspectorSources() {
        if (!sourcesCardList || !sourcesEmptyState) return;

        if (!latestRetrievedCitations || latestRetrievedCitations.length === 0) {
            sourcesEmptyState.classList.remove("hidden");
            sourcesCardList.innerHTML = "";
            return;
        }

        sourcesEmptyState.classList.add("hidden");
        sourcesCardList.innerHTML = latestRetrievedCitations.map(c => `
            <div class="source-inspector-card">
                <div class="source-card-top">
                    <span class="source-doc-name" title="${escapeHtml(c.filename)}">📄 ${escapeHtml(c.filename)}</span>
                    <span class="source-match-score">${Math.round((c.score || 0) * 100)}% match</span>
                </div>
                <span class="source-page-label">Page / Section: ${escapeHtml(String(c.page || 1))}</span>
                <div class="source-card-text">"${escapeHtml(c.snippet || '')}"</div>
            </div>
        `).join("");
    }

    function openInspectorTab(tabId) {
        if (contextPanel) {
            contextPanel.classList.remove("collapsed");
            if (window.innerWidth <= 900) {
                contextPanel.classList.add("open-mobile");
                if (inspectorBackdrop) inspectorBackdrop.classList.add("active");
            }
        }

        document.querySelectorAll(".tab-btn").forEach(btn => {
            btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
        });

        document.querySelectorAll(".tab-pane").forEach(pane => {
            pane.classList.toggle("active", pane.id === tabId);
        });
    }

    // ========================================================================
    // 7. Event Handlers & Micro-Interactions
    // ========================================================================

    // Composer Form Submit
    if (chatForm) {
        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            if (chatInput) sendQuery(chatInput.value);
        });
    }

    // Textarea Enter-to-send (Shift+Enter for newline)
    if (chatInput) {
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendQuery(chatInput.value);
            }
        });

        // Auto-resizing textarea
        chatInput.addEventListener("input", () => {
            chatInput.style.height = "auto";
            chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + "px";
        });
    }

    // Global Keyboard Shortcut: Ctrl+K or Cmd+K for New Chat
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
            e.preventDefault();
            createNewSession();
        }
    });

    // New Chat Buttons
    if (newChatBtn) {
        newChatBtn.addEventListener("click", () => createNewSession());
    }

    // Reset Chat Messages Button
    if (resetCurrentChatBtn) {
        resetCurrentChatBtn.addEventListener("click", async () => {
            try {
                await fetch("/api/chat/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: currentSessionId })
                });

                messagesContainer.innerHTML = "";
                latestRetrievedCitations = [];
                renderInspectorSources();
                if (welcomeHero) welcomeHero.classList.remove("hidden");
                if (chatInput) chatInput.focus();
            } catch (err) {
                console.error("Reset failed:", err);
            }
        });
    }

    // Interactive Suggestion Cards & Quick Chips Handlers
    document.addEventListener("click", (e) => {
        const suggestCard = e.target.closest(".suggest-card");
        if (suggestCard) {
            const prompt = suggestCard.getAttribute("data-prompt");
            if (prompt) sendQuery(prompt);
            return;
        }

        const miniChip = e.target.closest(".mini-chip");
        if (miniChip) {
            const prompt = miniChip.getAttribute("data-prompt");
            if (prompt) sendQuery(prompt);
            return;
        }

        const modeBtn = e.target.closest(".mode-item");
        if (modeBtn) {
            const prompt = modeBtn.getAttribute("data-mode-prompt");
            if (prompt) {
                closeSidebar();
                sendQuery(prompt);
            }
            return;
        }

        const toolActionBtn = e.target.closest(".tool-action-btn");
        if (toolActionBtn) {
            const prompt = toolActionBtn.getAttribute("data-prompt");
            if (prompt) {
                if (window.innerWidth <= 900) closeInspector();
                sendQuery(prompt);
            }
            return;
        }
    });

    // Dropzone Drag-and-Drop & File Selection
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
        notesDropzone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                if (notesFileInput) notesFileInput.click();
            }
        });
    }

    if (notesFileInput) {
        notesFileInput.addEventListener("change", () => {
            const files = notesFileInput.files;
            if (files && files.length > 0) {
                uploadStudyFile(files[0]);
                notesFileInput.value = "";
            }
        });
    }

    // Attach File Trigger Buttons (Composer 📎, Sidebar, Hero, Drawer)
    function triggerFileUpload() {
        if (uploadDrawer && uploadDrawer.classList.contains("hidden")) {
            uploadDrawer.classList.remove("hidden");
        }
        if (notesFileInput) notesFileInput.click();
    }

    if (composerAttachBtn) composerAttachBtn.addEventListener("click", triggerFileUpload);
    if (sidebarUploadTrigger) sidebarUploadTrigger.addEventListener("click", triggerFileUpload);
    if (emptyDocUploadBtn) emptyDocUploadBtn.addEventListener("click", triggerFileUpload);
    if (heroUploadTrigger) heroUploadTrigger.addEventListener("click", triggerFileUpload);
    if (inspectorUploadBtn) inspectorUploadBtn.addEventListener("click", triggerFileUpload);

    // Toggle Upload Drawer
    if (toggleUploadZoneBtn) {
        toggleUploadZoneBtn.addEventListener("click", () => {
            if (uploadDrawer) uploadDrawer.classList.toggle("hidden");
        });
    }
    if (closeUploadDrawerBtn) {
        closeUploadDrawerBtn.addEventListener("click", () => {
            if (uploadDrawer) uploadDrawer.classList.add("hidden");
        });
    }

    // Toggle Right Inspector Panel
    if (toggleInspectorBtn) {
        toggleInspectorBtn.addEventListener("click", () => {
            if (contextPanel) {
                if (window.innerWidth <= 900) {
                    contextPanel.classList.toggle("open-mobile");
                    if (inspectorBackdrop) inspectorBackdrop.classList.toggle("active");
                } else {
                    contextPanel.classList.toggle("collapsed");
                }
            }
        });
    }
    if (closeInspectorBtn) {
        closeInspectorBtn.addEventListener("click", closeInspector);
    }
    if (inspectorBackdrop) {
        inspectorBackdrop.addEventListener("click", closeInspector);
    }

    function closeInspector() {
        if (contextPanel) {
            contextPanel.classList.remove("open-mobile");
        }
        if (inspectorBackdrop) {
            inspectorBackdrop.classList.remove("active");
        }
    }

    // Inspector Tab Switching
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");
            openInspectorTab(tabId);
        });
    });

    // Mobile Sidebar Drawer
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener("click", () => {
            if (sidebar) sidebar.classList.add("open-mobile");
            if (sidebarBackdrop) sidebarBackdrop.classList.add("active");
        });
    }
    if (sidebarCloseBtn) {
        sidebarCloseBtn.addEventListener("click", closeSidebar);
    }
    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener("click", closeSidebar);
    }

    function closeSidebar() {
        if (sidebar) sidebar.classList.remove("open-mobile");
        if (sidebarBackdrop) sidebarBackdrop.classList.remove("active");
    }

    // Clear All Sessions Button
    if (clearAllHistoryBtn) {
        clearAllHistoryBtn.addEventListener("click", () => {
            if (confirm("Clear all recorded sessions from your sidebar?")) {
                localStorage.removeItem(RECENT_SESSIONS_KEY);
                renderRecentSessionsList();
            }
        });
    }

    // Settings Modal
    if (openSettingsBtn && settingsModal) {
        openSettingsBtn.addEventListener("click", () => {
            settingsModal.classList.remove("hidden");
        });
    }
    if (closeSettingsModalBtn && settingsModal) {
        closeSettingsModalBtn.addEventListener("click", () => {
            settingsModal.classList.add("hidden");
        });
    }
    if (settingsModal) {
        settingsModal.addEventListener("click", (e) => {
            if (e.target === settingsModal) settingsModal.classList.add("hidden");
        });
    }

    if (clearLocalDataBtn) {
        clearLocalDataBtn.addEventListener("click", () => {
            if (confirm("Reset all local Edvora study sessions, documents, and cache?")) {
                localStorage.clear();
                window.location.reload();
            }
        });
    }

    // ========================================================================
    // 8. Markdown Parsing Utility
    // ========================================================================

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatMarkdown(rawText) {
        if (!rawText) return "";

        // 1. Extract fenced code blocks first
        const codeBlocks = [];
        let text = rawText.replace(/```([a-zA-Z0-9_\-+]*)\n([\s\S]*?)```/g, (match, lang, code) => {
            const idx = codeBlocks.length;
            const safeLang = (lang || "code").toLowerCase();
            const safeCode = escapeHtml(code.trim());
            const rawEncoded = encodeURIComponent(code.trim());

            codeBlocks.push(`
                <div class="code-block-wrapper">
                    <div class="code-block-header">
                        <span class="code-lang-tag">${safeLang}</span>
                        <button type="button" class="btn-copy-code" data-code="${rawEncoded}">Copy</button>
                    </div>
                    <pre class="code-block-pre"><code>${safeCode}</code></pre>
                </div>
            `);
            return `__CODE_BLOCK_${idx}__`;
        });

        // 2. Escape general HTML
        text = escapeHtml(text);

        // 3. Headers
        text = text.replace(/^#### (.*$)/gim, "<h4>$1</h4>");
        text = text.replace(/^### (.*$)/gim, "<h3>$1</h3>");
        text = text.replace(/^## (.*$)/gim, "<h2>$1</h2>");
        text = text.replace(/^# (.*$)/gim, "<h1>$1</h1>");

        // 4. Tables
        text = text.replace(/((?:\|[^\n]+\|\r?\n)+)/g, (tableBlock) => {
            const lines = tableBlock.trim().split("\n").map(l => l.trim()).filter(Boolean);
            if (lines.length < 2) return tableBlock;

            const parseCells = (row) => {
                return row
                    .replace(/^\|/, "")
                    .replace(/\|$/, "")
                    .split("|")
                    .map(c => c.trim());
            };

            const headerCells = parseCells(lines[0]);
            // check if line 1 is separator
            const isSeparator = /^\|?(\s*:?-+:?\s*\|)+\s*$/.test(lines[1]);
            const bodyLines = isSeparator ? lines.slice(2) : lines.slice(1);

            const thead = `<thead><tr>${headerCells.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
            const tbodyRows = bodyLines.map(row => {
                const cells = parseCells(row);
                return `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
            }).join("");

            return `<table>${thead}<tbody>${tbodyRows}</tbody></table>`;
        });

        // 5. Blockquotes
        text = text.replace(/^\> (.*$)/gim, "<blockquote>$1</blockquote>");

        // 6. Bold & Italics
        text = text.replace(/\*\*\*(.*?)\*\*\*/g, "<strong><em>$1</em></strong>");
        text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        text = text.replace(/\*(.*?)\*/g, "<em>$1</em>");
        text = text.replace(/`([^`]+)`/g, "<code style=\"font-family:var(--font-mono); background:rgba(0,0,0,0.3); padding:1px 5px; border-radius:4px; font-size:0.86em;\">$1</code>");

        // 7. Bullet Lists & Numbered Lists
        text = text.replace(/^(\s*)-\s+(.*$)/gim, "<li>$2</li>");
        text = text.replace(/^(\s*)\*\s+(.*$)/gim, "<li>$2</li>");
        text = text.replace(/(<li>.*<\/li>)/gis, "<ul>$1</ul>");

        // 8. Paragraphs
        const paragraphs = text.split(/\n{2,}/);
        text = paragraphs.map(p => {
            const trimmed = p.trim();
            if (!trimmed) return "";
            if (trimmed.startsWith("<h") || trimmed.startsWith("<ul") || trimmed.startsWith("<table") || trimmed.startsWith("<blockquote") || trimmed.startsWith("__CODE_BLOCK_")) {
                return trimmed;
            }
            return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
        }).join("");

        // 9. Restore code blocks
        codeBlocks.forEach((block, idx) => {
            text = text.replace(`__CODE_BLOCK_${idx}__`, block);
        });

        return text;
    }

    // ========================================================================
    // 9. System Initialization & Health Check
    // ========================================================================

    async function checkHealth() {
        try {
            const res = await fetch("/api/health");
            const data = await res.json().catch(() => null);
            if (res.ok && data) {
                const model = data.model || "Llama 3.3 • 70B";
                if (modelStatusLabel) {
                    modelStatusLabel.textContent = `Edvora • ${model}`;
                }
                if (modalModelInfo) {
                    modalModelInfo.innerHTML = `<strong>${escapeHtml(model)}</strong> (RAG Vector Store Active)`;
                }
            }
        } catch (e) {
            console.warn("Health check error:", e);
        }
    }

    // Hydrate
    renderRecentSessionsList();
    loadHistory();
    loadUploadedDocuments();
    checkHealth();
});
