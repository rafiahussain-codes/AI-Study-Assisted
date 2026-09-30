"""
RAG - Vector Store & Similarity Retrieval Engine
Provides vector space indexing and similarity retrieval over document chunks.
Uses scikit-learn TF-IDF vector embeddings with sublinear term-frequency,
n-gram expansion, and cosine similarity combined with lexical keyword scoring.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_splitter import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    In-memory vector store supporting chunk indexing, cosine similarity retrieval,
    document deletion, and citation tracking per session.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.documents_metadata: Dict[str, Dict[str, Any]] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def add_document(self, doc_id: str, filename: str, chunks: List[DocumentChunk]) -> int:
        """
        Add document chunks to the store and re-index the vector space.
        Returns the number of chunks indexed.
        """
        if not chunks:
            return 0

        # Remove existing chunks for this doc_id if updating
        self.delete_document(doc_id)

        self.chunks.extend(chunks)
        self.documents_metadata[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_count": len(chunks),
            "char_count": sum(c.char_count for c in chunks)
        }

        self._reindex()
        logger.info("Indexed %d chunks for document '%s' (doc_id=%s)", len(chunks), filename, doc_id)
        return len(chunks)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove a document and its chunks from the store.
        """
        initial_len = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        if doc_id in self.documents_metadata:
            del self.documents_metadata[doc_id]

        if len(self.chunks) != initial_len:
            self._reindex()
            return True
        return False

    def clear(self):
        """
        Clear all documents and chunks in the store.
        """
        self.chunks = []
        self.documents_metadata = {}
        self.vectorizer = None
        self.tfidf_matrix = None

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Return metadata list of all active documents.
        """
        return list(self.documents_metadata.values())

    def _reindex(self):
        """
        Compute TF-IDF vector matrix over all stored chunk texts.
        """
        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000,
            strip_accents="unicode"
        )
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except Exception as exc:
            logger.error("Failed to vectorize corpus: %s", exc)
            self.tfidf_matrix = None

    def search(self, query: str, top_k: int = 4, score_threshold: float = 0.05) -> List[Dict[str, Any]]:
        """
        Query vector index and return top_k most relevant chunks.
        
        Returns:
            List of dicts: [
                {
                    "chunk_id": str,
                    "doc_id": str,
                    "filename": str,
                    "page": int,
                    "text": str,
                    "score": float
                }
            ]
        """
        query = query.strip()
        if not query or not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            cos_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        except Exception as exc:
            logger.error("Error computing cosine similarity for query '%s': %s", query, exc)
            cos_similarities = np.zeros(len(self.chunks))

        # Lexical keyword bonus for exact term presence
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        scored_results: List[Tuple[float, DocumentChunk]] = []

        for idx, chunk in enumerate(self.chunks):
            cos_score = float(cos_similarities[idx])
            chunk_lower = chunk.text.lower()

            # Exact keyword match boost
            kw_hits = sum(1 for term in query_terms if term in chunk_lower)
            kw_boost = 0.05 * min(kw_hits, 4)

            total_score = cos_score + kw_boost

            if total_score >= score_threshold:
                scored_results.append((total_score, chunk))

        # Sort by relevance score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored_results[:top_k]:
            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "score": round(score, 4)
            })

        return results
