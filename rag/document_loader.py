"""
RAG - Document Loader Module
Extracts textual content and metadata from multiple file formats:
- PDF (.pdf) using pypdf
- Word Documents (.docx) using python-docx
- Plain Text (.txt, .text)
- Markdown (.md, .markdown)
"""

import os
import io
import logging
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".text", ".md", ".markdown"}


class DocumentLoadError(Exception):
    """Raised when a document cannot be read or parsed."""
    pass


def extract_from_pdf(stream_or_path: Union[str, io.BytesIO], filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF file.
    Returns a list of dicts: [{"text": str, "page": int, "source": str}]
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocumentLoadError("pypdf is not installed. Please install pypdf to read PDF files.")

    try:
        if isinstance(stream_or_path, str):
            reader = PdfReader(stream_or_path)
        else:
            stream_or_path.seek(0)
            reader = PdfReader(stream_or_path)

        pages_data = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned = page_text.strip()
            if cleaned:
                pages_data.append({
                    "text": cleaned,
                    "page": idx + 1,
                    "source": filename
                })

        if not pages_data:
            logger.warning("No readable text found in PDF: %s", filename)

        return pages_data
    except Exception as exc:
        logger.error("Failed to parse PDF %s: %s", filename, exc)
        raise DocumentLoadError(f"Could not read PDF '{filename}': {str(exc)}")


def extract_from_docx(stream_or_path: Union[str, io.BytesIO], filename: str = "document.docx") -> List[Dict[str, Any]]:
    """
    Extract text paragraph-by-paragraph from a DOCX file.
    Returns a list of dicts: [{"text": str, "page": int, "source": str}]
    """
    try:
        import docx
    except ImportError:
        raise DocumentLoadError("python-docx is not installed. Please install python-docx to read DOCX files.")

    try:
        if isinstance(stream_or_path, str):
            doc = docx.Document(stream_or_path)
        else:
            stream_or_path.seek(0)
            doc = docx.Document(stream_or_path)

        paragraphs = []
        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                paragraphs.append(txt)

        # Also extract table cells if any
        for table in doc.tables:
            for row in table.rows:
                row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_texts:
                    paragraphs.append(" | ".join(row_texts))

        if not paragraphs:
            return []

        full_text = "\n\n".join(paragraphs)
        return [{
            "text": full_text,
            "page": 1,
            "source": filename
        }]
    except Exception as exc:
        logger.error("Failed to parse DOCX %s: %s", filename, exc)
        raise DocumentLoadError(f"Could not read Word document '{filename}': {str(exc)}")


def extract_from_text(stream_or_path: Union[str, io.BytesIO], filename: str = "document.txt") -> List[Dict[str, Any]]:
    """
    Extract text from a plain text or Markdown file.
    Handles UTF-8 and fallback encodings (latin-1, cp1252).
    """
    raw_bytes = b""
    if isinstance(stream_or_path, str):
        with open(stream_or_path, "rb") as f:
            raw_bytes = f.read()
    else:
        stream_or_path.seek(0)
        raw_bytes = stream_or_path.read()

    # Try common encodings
    text = ""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    text = text.strip()
    if not text:
        return []

    return [{
        "text": text,
        "page": 1,
        "source": filename
    }]


def load_document(source: Union[str, io.BytesIO], filename: str = "") -> List[Dict[str, Any]]:
    """
    Unified entrypoint to load a document given a path or stream.
    Validates file extension and dispatches to appropriate parser.
    """
    if isinstance(source, str):
        if not filename:
            filename = os.path.basename(source)
        ext = os.path.splitext(filename)[1].lower()
    else:
        if not filename:
            raise DocumentLoadError("filename must be provided when passing a stream")
        ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(f"Unsupported file format '{ext}'. Supported formats: {supported_str}")

    if ext == ".pdf":
        return extract_from_pdf(source, filename)
    elif ext == ".docx":
        return extract_from_docx(source, filename)
    elif ext in {".txt", ".text", ".md", ".markdown"}:
        return extract_from_text(source, filename)
    else:
        raise DocumentLoadError(f"Unhandled file extension '{ext}'")
