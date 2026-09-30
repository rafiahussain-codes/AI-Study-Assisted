"""
RAG - Text Splitter Module
Implements recursive hierarchical text chunking with overlap to preserve
semantic cohesion and section context for retrieval.
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def recursive_split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 120) -> List[str]:
    """
    Split text recursively by paragraphs, sentences, or phrases while keeping
    chunk lengths below chunk_size and maintaining chunk_overlap between consecutive chunks.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Split separators in priority order
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def _split(txt: str, sep_idx: int = 0) -> List[str]:
        if len(txt) <= chunk_size or sep_idx >= len(separators):
            # Base case: text fits or no more separators
            if len(txt) <= chunk_size:
                return [txt]
            # Hard break if single word/token exceeds chunk_size
            chunks = []
            start = 0
            while start < len(txt):
                end = min(start + chunk_size, len(txt))
                chunks.append(txt[start:end])
                start += (chunk_size - chunk_overlap)
                if start >= len(txt) or end >= len(txt):
                    break
            return chunks

        sep = separators[sep_idx]
        parts = txt.split(sep)
        result = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                # If individual part is too large, recurse with next separator
                if len(part) > chunk_size:
                    sub_chunks = _split(part, sep_idx + 1)
                    result.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current:
            result.append(current)

        return result

    raw_chunks = _split(text, 0)

    # Now apply sliding overlap to enhance context continuity
    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return [c.strip() for c in raw_chunks if c.strip()]

    final_chunks = []
    for i, c in enumerate(raw_chunks):
        c_clean = c.strip()
        if not c_clean:
            continue
        if i == 0:
            final_chunks.append(c_clean)
        else:
            # Prepend the tail of previous chunk as overlap if beneficial
            prev_tail = raw_chunks[i - 1].strip()[-chunk_overlap:]
            # Avoid duplicating text if already starts with same words
            if prev_tail and not c_clean.startswith(prev_tail):
                combined = f"...{prev_tail} {c_clean}"
            else:
                combined = c_clean
            final_chunks.append(combined)

    return final_chunks


def split_text_into_chunks(
    pages_or_docs: List[Dict[str, Any]],
    doc_id: str,
    chunk_size: int = 600,
    chunk_overlap: int = 120
) -> List[DocumentChunk]:
    """
    Split a list of extracted document pages into chunk objects with metadata.
    
    pages_or_docs: list of dicts with {"text": str, "page": int, "source": str}
    doc_id: unique document identifier
    """
    all_chunks: List[DocumentChunk] = []
    global_chunk_idx = 0

    for doc_item in pages_or_docs:
        page_text = doc_item.get("text", "")
        page_num = doc_item.get("page", 1)
        source = doc_item.get("source", "document")

        chunks_text = recursive_split_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_text in chunks_text:
            chunk_id = f"{doc_id}_p{page_num}_c{global_chunk_idx}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                filename=source,
                page=page_num,
                chunk_index=global_chunk_idx,
                text=chunk_text,
                char_count=len(chunk_text)
            )
            all_chunks.append(chunk)
            global_chunk_idx += 1

    return all_chunks
