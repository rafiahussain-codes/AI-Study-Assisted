# Edvora — Your AI-powered study companion

An interactive, modern AI-powered academic workspace equipped with **Document Understanding** and **Retrieval-Augmented Generation (RAG)**.

Students can upload lecture notes, slides, textbook chapters, or syllabi (PDF, DOCX, TXT, MD), and ask questions in natural language. Edvora extracts relevant information from their uploaded materials, maintains conversation context across turns, grounds answers with source citations, and provides interactive study tools such as summaries, practice MCQs, concept comparisons, and exam questions.

---

## Brand & Visual Experience

- **Application Name:** **Edvora**
- **Tagline:** *Your AI-powered study companion.*
- **Aesthetic:** Dark AI-product workspace with deep obsidian charcoal surfaces, dark glassmorphism cards, subtle gradients, electric indigo & violet accents, clean typography (`Outfit`, `Plus Jakarta Sans`, `JetBrains Mono`), and fluid micro-animations.

---

## Key Features & Architecture

### 1. Modern Workspace Layout
- **Left Sidebar:**
  - Edvora logo & glowing brand spark
  - **New Chat** button (`Ctrl+K` keyboard shortcut)
  - **Knowledge Base manager** (active documents count, format icons, chunk badges, remove button)
  - **Quick Study Modes** (Explain My Notes, Practice MCQs, Chapter Summary, High-Yield Topics)
  - **Recent Study Sessions** (clickable session switcher)
  - **Engine Status** & Settings modal trigger
- **Main Study Area:**
  - Clean topbar with session title, RAG status pill, and mobile drawer toggles
  - **Interactive Welcome Hero** (*"What are you learning today?"*) with 6 interactive suggestion cards
  - **Collapsible Document Upload Drawer** with multi-stage progress animation (`Uploading` → `Processing` → `Understanding` → `Ready`)
  - **Conversational Chat Feed** with user cards, Edvora tutor cards, code blocks with syntax copy, and formatted comparison tables
  - **RAG Knowledge Banner & Citations** (*"Using your knowledge base • N relevant sections found"*) with source cards and match confidence percentages
  - **Bottom Floating Composer** with quick prompt chips, direct file attachment (`📎`), auto-resizing input, and glowing send button
- **Collapsible Right Context Panel (Knowledge Inspector):**
  - **Sources Used Tab:** Inspect exact retrieved chunks and similarity scores for the latest AI answer
  - **Documents Tab:** View indexed documents, pages, chunk counts, and character statistics
  - **Study Tools Tab:** Active recall quizzes, flashcard builders, and concept hierarchy mappers

### 2. Document Ingestion & Vector Retrieval (RAG)
- **Multi-Format Extraction:** PDF (`pypdf`), Word `.docx` (`python-docx`), Plain Text `.txt`, and Markdown `.md`.
- **Recursive Chunking:** Contextual sliding-window chunking with overlap, preserving page and chunk IDs.
- **In-Memory Vector Store:** Sublinear TF-IDF vectors, cosine similarity ranking, and keyword match boosting.
- **Context-Aware Retrieval:** Resolves follow-up queries and pronouns (*"How does it work?"*, *"What are its advantages?"*) using conversational history.

---

## REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves Edvora AI Study Workspace |
| `GET` | `/api/health` | Backend status & LLM readiness |
| `POST` | `/api/chat/message` | Process query with vector retrieval & multi-turn memory |
| `POST` | `/api/chat/upload` | Ingest and index PDF, DOCX, TXT, MD study documents |
| `GET` | `/api/chat/documents` | List indexed documents for the current session |
| `DELETE` | `/api/chat/documents/<doc_id>` | Remove document from session vector memory |
| `GET` | `/api/chat/sessions` | List active chat sessions |
| `POST` | `/api/chat/sessions` | Create a new study session |
| `DELETE` | `/api/chat/sessions/<id>` | Delete a study session |
| `GET` | `/api/chat/history` | Retrieve conversation history for a session |
| `POST` | `/api/chat/reset` | Clear messages in current session |

---

## Running Locally

1. **Activate virtual environment:**
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Configure API key in `.env`:**
   ```env
   LLM_API_KEY=your_groq_or_gemini_api_key
   LLM_PROVIDER=groq
   ```
3. **Run Flask application:**
   ```powershell
   python app.py
   ```
4. **Open in browser:**
   ```
   http://127.0.0.1:5000
   ```

5. **Run test suite:**
   ```powershell
   python -m pytest
   ```
