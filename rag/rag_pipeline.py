"""
RAG - Unified Pipeline Coordinator
Orchestrates:
1. Document ingestion from files or bytes streams
2. Multi-page text extraction
3. Recursive chunking with metadata
4. Vector space indexing and retrieval
5. Context synthesis and source citation formatting for LLMs
"""

import io
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from .document_loader import load_document, DocumentLoadError
from .text_splitter import split_text_into_chunks
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Manages session-level or global RAG operations.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def process_and_index_document(
        self,
        file_source: Union[str, io.BytesIO],
        filename: str,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full ingestion pipeline:
        1. Extract text from PDF, DOCX, TXT, MD
        2. Chunk text with metadata
        3. Index into vector store
        
        Returns metadata about indexed document.
        """
        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]

        # Step 1: Extract text
        pages = load_document(file_source, filename=filename)
        if not pages:
            raise DocumentLoadError(f"Document '{filename}' appears to be empty or contains no readable text.")

        # Step 2: Chunk text
        chunks = split_text_into_chunks(
            pages,
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        if not chunks:
            raise DocumentLoadError(f"No usable text chunks could be extracted from '{filename}'.")

        # Step 3: Index chunks in vector store
        num_chunks = self.vector_store.add_document(doc_id, filename, chunks)

        total_chars = sum(len(c.text) for c in chunks)
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": num_chunks,
            "page_count": len(pages),
            "char_count": total_chars
        }

    def retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.05
    ) -> Tuple[str, List[Dict[str, Any]], bool]:
        """
        Retrieve relevant chunks for a user query.
        
        Returns:
            Tuple of:
            - formatted_context_str: text block suitable for LLM prompt injection
            - citations: list of structured source reference dicts for the UI
            - has_relevant_context: boolean indicating whether meaningful notes were found
        """
        results = self.vector_store.search(query, top_k=top_k, score_threshold=score_threshold)

        if not results:
            return "", [], False

        context_blocks = []
        citations = []
        seen_snippets = set()

        for idx, res in enumerate(results, 1):
            source_file = res["filename"]
            page_num = res["page"]
            text_snippet = res["text"].strip()

            context_blocks.append(
                f"[Source {idx}: {source_file}, Page/Section {page_num}]\n{text_snippet}\n"
            )

            # Avoid duplicate citations
            citation_key = f"{source_file}:{page_num}"
            if citation_key not in seen_snippets:
                seen_snippets.add(citation_key)
                citations.append({
                    "id": res["chunk_id"],
                    "filename": source_file,
                    "page": page_num,
                    "score": res["score"],
                    "snippet": text_snippet[:180] + ("..." if len(text_snippet) > 180 else "")
                })

        formatted_context_str = "\n".join(context_blocks)
        return formatted_context_str, citations, True

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.vector_store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        return self.vector_store.delete_document(doc_id)

    def clear(self):
        self.vector_store.clear()
