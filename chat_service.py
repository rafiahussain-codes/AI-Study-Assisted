"""
AI Study Assistant - Chat Service
Handles:
- Multi-turn conversation sessions and context memory
- RAG integration with uploaded study notes (PDF, DOCX, TXT, MD)
- Pronoun and follow-up query resolution ("What are its advantages?")
- Distinction between notes-based answers (with citations) and general study knowledge
- Study tools: summarization, MCQs, step-by-step explanations, exam questions, concept comparisons, examples
"""

import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from rag import RAGPipeline
from llm_service import generate_chat_completion, is_llm_configured

logger = logging.getLogger(__name__)

STUDY_ASSISTANT_SYSTEM_PROMPT = """You are Edvora, an expert, patient, and highly intelligent AI Study Assistant and personal academic companion.
Tagline: Your AI-powered study companion.

Your purpose is to help students learn effectively, master complex subjects, prepare for exams, break down difficult ideas, and extract maximum value from their study materials.

Guidelines for your responses:
1. Tone & Style:
   - Clear, pedagogical, encouraging, structured, and insightful.
   - Use clean Markdown: bold essential terms, use bullet points, numbered steps, code blocks, or comparison tables where appropriate.
   - Never produce monotonous walls of unformatted text.

2. Source Grounding & Distinction:
   - When "UPLOADED STUDY NOTES CONTEXT" is provided:
     * Ground your answer primarily in those notes.
     * Accurately reflect definitions, formulas, and concepts present in the notes.
     * Reference the source documents and page/section numbers where applicable.
   - When study notes are uploaded but do NOT contain the answer to the student's question:
     * Explicitly inform the student at the beginning:
       "⚠️ *Note: This topic was not found in your uploaded study notes.* Here is a clear academic explanation from general knowledge:"
     * Then proceed with an accurate, well-structured educational explanation.
   - When no study notes are uploaded:
     * Answer as a versatile, comprehensive academic study tutor.

3. Study Capabilities:
   - "Explain simply / Beginner": Use intuitive real-world analogies, plain language, and step-by-step points.
   - "Summarize": Provide a concise high-level overview, key core concepts, and primary takeaways.
   - "Explain paragraph / Step-by-step": Break the concept into digestible logical steps.
   - "Generate MCQs": Create 4-5 high-yield multiple-choice questions with 4 options (A, B, C, D), followed by an Answer Key with brief explanations.
   - "Important concepts / Exam prep": Highlight key definitions, formulas, common pitfalls, and likely exam questions.
   - "Compare two concepts": Provide a clear comparison table or structured bullet contrasts (similarities vs. differences).
   - "Give examples": Provide concrete, practical scenarios illustrating the concept.
"""


class ChatSession:
    """
    State container for an individual user chat conversation and its RAG pipeline.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.title = "New Study Session"
        self.created_at = time.time()
        self.updated_at = time.time()
        self.messages: List[Dict[str, Any]] = []
        self.rag_pipeline = RAGPipeline()

    def add_message(self, role: str, content: str, citations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        msg = {
            "id": str(uuid.uuid4())[:8],
            "role": role,
            "content": content,
            "citations": citations or [],
            "timestamp": time.time()
        }
        self.messages.append(msg)
        self.updated_at = time.time()
        if role == "user" and self.title == "New Study Session":
            clean = " ".join(content.strip().split())
            if len(clean) > 36:
                clean = clean[:36].rstrip() + "..."
            self.title = clean or "Study Session"
        return msg

    def clear_history(self):
        self.messages = []
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
            "doc_count": len(self.rag_pipeline.list_documents())
        }


class ChatService:
    """
    Session manager and conversational orchestrator.
    """

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        if not session_id or session_id not in self.sessions:
            new_id = session_id or str(uuid.uuid4())
            self.sessions[new_id] = ChatSession(new_id)
            return self.sessions[new_id]
        return self.sessions[session_id]

    def reset_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            self.sessions[session_id].clear_history()
            return True
        return False

    def create_new_session(self, title: str = "New Study Session") -> ChatSession:
        new_id = f"edvora-{str(uuid.uuid4())[:8]}"
        session = ChatSession(new_id)
        session.title = title
        self.sessions[new_id] = session
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        sorted_sessions = sorted(
            self.sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True
        )
        return [s.to_dict() for s in sorted_sessions]

    def remove_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def _resolve_contextual_query(self, current_query: str, history: List[Dict[str, Any]]) -> str:
        """
        Enrich queries containing pronouns or short follow-ups (e.g. "What are its advantages?")
        using the previous user message so vector retrieval targets the true subject.
        """
        query_lower = current_query.lower()
        pronouns = {"it", "its", "they", "them", "their", "this", "that", "these", "those"}
        words = set(query_lower.split())

        has_pronoun = bool(words.intersection(pronouns))
        is_short_followup = len(current_query.split()) <= 5

        if (has_pronoun or is_short_followup) and history:
            last_user_query = ""
            for msg in reversed(history):
                if msg.get("role") == "user":
                    last_user_query = msg.get("content", "")
                    break
            if last_user_query:
                return f"{last_user_query} {current_query}"

        return current_query

    def handle_message(
        self,
        session_id: Optional[str],
        user_message: str
    ) -> Dict[str, Any]:
        """
        Process a student chat message:
        1. Resolve session
        2. Contextual query expansion for follow-ups
        3. Retrieve relevant chunks from uploaded notes via RAG
        4. Assemble conversational prompts
        5. Invoke LLM
        6. Persist message history and return answer with citations
        """
        session = self.get_or_create_session(session_id)
        clean_user_message = user_message.strip()

        if not clean_user_message:
            return {
                "success": False,
                "error": "Message cannot be empty."
            }

        # 1. RAG Retrieval
        retrieval_query = self._resolve_contextual_query(clean_user_message, session.messages)
        active_docs = session.rag_pipeline.list_documents()
        has_uploaded_docs = len(active_docs) > 0

        context_str, citations, has_relevant_context = session.rag_pipeline.retrieve_context(
            retrieval_query,
            top_k=4,
            score_threshold=0.08
        )

        # 2. Build augmented system instructions
        system_instruction = STUDY_ASSISTANT_SYSTEM_PROMPT

        if has_uploaded_docs:
            doc_names = ", ".join(f"'{d['filename']}'" for d in active_docs)
            system_instruction += f"\n\nCurrently uploaded student notes: {doc_names}."
            if has_relevant_context and context_str:
                system_instruction += (
                    f"\n\n=== UPLOADED STUDY NOTES CONTEXT ===\n"
                    f"{context_str}\n"
                    f"=== END OF CONTEXT ===\n"
                    f"Prioritize the above context and cite relevant sources and page numbers in your answer."
                )
            else:
                system_instruction += (
                    f"\n\nNote: The user's query did not find a direct match in the uploaded notes ({doc_names}). "
                    f"Clearly notify the student that this information was not found in their uploaded notes, "
                    f"then provide a clear general academic explanation."
                )

        # 3. Assemble message turns for multi-turn LLM context (last 8 turns)
        recent_history = session.messages[-8:]
        chat_turns = []
        for msg in recent_history:
            chat_turns.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Append current user message
        chat_turns.append({
            "role": "user",
            "content": clean_user_message
        })

        # 4. Invoke LLM
        success, result, status_code = generate_chat_completion(
            messages=chat_turns,
            system_instruction=system_instruction,
            max_tokens=1500
        )

        if not success:
            logger.error("Chat completion failed: %s (status=%s)", result.get("error"), status_code)
            return {
                "success": False,
                "error": result.get("error", "Unable to generate response from AI service."),
                "session_id": session.session_id
            }

        assistant_reply = result.get("reply", "")

        # 5. Record history
        session.add_message("user", clean_user_message)
        session.add_message("assistant", assistant_reply, citations=citations)

        return {
            "success": True,
            "session_id": session.session_id,
            "reply": assistant_reply,
            "citations": citations,
            "has_notes": has_uploaded_docs,
            "notes_cited": len(citations) > 0,
            "model": result.get("model", "")
        }


# Singleton service instance
chat_service = ChatService()
