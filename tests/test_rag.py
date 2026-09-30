"""
Unit Tests for RAG Module (DocumentLoader, TextSplitter, VectorStore, RAGPipeline)
"""

import io
import pytest
from rag.document_loader import load_document, DocumentLoadError, SUPPORTED_EXTENSIONS
from rag.text_splitter import split_text_into_chunks, recursive_split_text, DocumentChunk
from rag.vector_store import VectorStore
from rag.rag_pipeline import RAGPipeline


def test_text_loader():
    """Test loading plain text stream."""
    sample_text = "This is a study note about Operating Systems. Virtual memory and paging are core concepts."
    stream = io.BytesIO(sample_text.encode("utf-8"))
    docs = load_document(stream, filename="os_notes.txt")

    assert len(docs) == 1
    assert "Operating Systems" in docs[0]["text"]
    assert docs[0]["source"] == "os_notes.txt"
    assert docs[0]["page"] == 1


def test_unsupported_file_extension():
    """Ensure unsupported extensions raise DocumentLoadError."""
    stream = io.BytesIO(b"fake data")
    with pytest.raises(DocumentLoadError) as exc_info:
        load_document(stream, filename="data.xyz")
    assert "Unsupported file format" in str(exc_info.value)


def test_recursive_split_text():
    """Test text chunking with length constraint and overlap."""
    text = (
        "Paragraph one is about Computer Architecture. Pipelining divides instruction execution into stages.\n\n"
        "Paragraph two describes hazards. Structural hazards occur when hardware resources conflict.\n\n"
        "Paragraph three covers data hazards. Data hazards occur when instructions depend on previous results.\n\n"
        "Paragraph four discusses branch prediction and control hazards."
    )
    chunks = recursive_split_text(text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_vector_store_indexing_and_search():
    """Test vector space indexing and similarity search."""
    store = VectorStore()
    doc_id = "doc1"
    filename = "computer_architecture.txt"

    chunk1 = DocumentChunk(
        chunk_id="c1",
        doc_id=doc_id,
        filename=filename,
        page=1,
        chunk_index=0,
        text="Pipelining is an implementation technique where multiple instructions are overlapped in execution.",
        char_count=98
    )
    chunk2 = DocumentChunk(
        chunk_id="c2",
        doc_id=doc_id,
        filename=filename,
        page=2,
        chunk_index=1,
        text="Cache memory is a small, high-speed memory located close to the CPU core to reduce latency.",
        char_count=92
    )

    count = store.add_document(doc_id, filename, [chunk1, chunk2])
    assert count == 2
    assert len(store.list_documents()) == 1

    # Search for pipelining
    results = store.search("Explain pipelining in simple terms", top_k=2)
    assert len(results) >= 1
    assert "pipelining" in results[0]["text"].lower()
    assert results[0]["filename"] == filename

    # Search for cache memory
    cache_results = store.search("What is cache memory?", top_k=2)
    assert len(cache_results) >= 1
    assert "cache" in cache_results[0]["text"].lower()

    # Delete document
    deleted = store.delete_document(doc_id)
    assert deleted is True
    assert len(store.list_documents()) == 0
    assert len(store.search("pipelining")) == 0


def test_rag_pipeline_end_to_end():
    """Test full RAG pipeline upload, retrieval, and citation generation."""
    pipeline = RAGPipeline()

    notes_content = (
        "Chapter 5: Pipelining Principles\n\n"
        "Pipelining improves CPU instruction throughput by overlapping execution phases.\n"
        "The five classic MIPS stages are:\n"
        "1. Instruction Fetch (IF)\n"
        "2. Instruction Decode (ID)\n"
        "3. Execute (EX)\n"
        "4. Memory Access (MEM)\n"
        "5. Write Back (WB)\n\n"
        "Advantages of pipelining include increased instruction throughput and better hardware utilization.\n"
        "However, hazards such as structural hazards and data dependencies must be resolved."
    )

    stream = io.BytesIO(notes_content.encode("utf-8"))
    res = pipeline.process_and_index_document(stream, filename="Chapter5_Pipelining.txt")

    assert res["success"] is True
    assert res["chunk_count"] > 0
    assert res["filename"] == "Chapter5_Pipelining.txt"

    # Query matching notes
    context_str, citations, has_notes = pipeline.retrieve_context("What are the advantages of pipelining?")
    assert has_notes is True
    assert "Chapter5_Pipelining.txt" in context_str
    assert len(citations) >= 1
    assert citations[0]["filename"] == "Chapter5_Pipelining.txt"

    # Query completely unrelated topic with high threshold
    unrelated_ctx, unrelated_cits, has_unrelated = pipeline.retrieve_context(
        "Photosynthesis in green plants",
        score_threshold=0.3
    )
    assert has_unrelated is False
    assert len(unrelated_cits) == 0
