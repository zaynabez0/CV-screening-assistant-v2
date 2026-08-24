"""Shared text extraction — used by BOTH JD processing and CV preprocessing.

One function, two callers. The JD file-upload path and every CV go through this
exact code; there is deliberately no second version of it anywhere in the repo.

Scope note (v1): PDF and DOCX only. A PDF with no text layer (a scan) yields
~0 characters and is rejected by `has_extractable_text` — there is no OCR
fallback, by design.
"""

import fitz  # pymupdf
import docx  # python-docx


def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from a PDF or DOCX file. Raises ValueError for anything else."""
    ext = file_path.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    elif ext == "docx":
        d = docx.Document(file_path)
        text = "\n".join(p.text for p in d.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    return text.strip()


def has_extractable_text(text: str, min_chars: int = 50) -> bool:
    """A scanned PDF with no text layer returns ~0 chars — treat as invalid input, not a special case."""
    return len(text) >= min_chars
