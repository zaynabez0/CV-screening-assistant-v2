"""Tests for the shared text-extraction function (screening/common/text_extraction.py).

This function is the single entry point for BOTH JD file uploads and CV uploads,
so a bug here corrupts every downstream stage. Each fixture pins one behaviour;
see fixtures/README.md for what each file represents.
"""

import shutil
from pathlib import Path

import pytest

from screening.common.text_extraction import extract_text_from_file, has_extractable_text

FIXTURES = Path(__file__).parent / "fixtures"


# --- The happy paths: a real PDF and a real DOCX both yield usable text -------

def test_clean_pdf_extracts_text_from_every_page():
    text = extract_text_from_file(str(FIXTURES / "clean_cv.pdf"))

    assert "Amina Benali" in text          # page 1
    assert "Atlas Data" in text            # page 1
    assert "ENSIAS" in text                # page 2 — proves pages are joined, not just the first
    assert "Django" in text                # page 2
    assert has_extractable_text(text)


def test_clean_docx_extracts_text_from_every_paragraph():
    text = extract_text_from_file(str(FIXTURES / "clean_cv.docx"))

    assert "Youssef El Amrani" in text     # first paragraph
    assert "Meridian Retail" in text
    assert "Universite Mohammed V" in text  # last paragraph
    assert has_extractable_text(text)


def test_extracted_text_is_stripped():
    """The function ends with .strip() — no leading/trailing whitespace reaches the LLM."""
    text = extract_text_from_file(str(FIXTURES / "clean_cv.pdf"))

    assert text == text.strip()
    assert not text.startswith("\n")


def test_extension_check_is_case_insensitive(tmp_path):
    """The spec lowercases the path before checking, so CV.PDF must work like cv.pdf.

    Real uploads from Windows machines routinely arrive with uppercase extensions.
    """
    upper = tmp_path / "CLEAN_CV.PDF"
    shutil.copy(FIXTURES / "clean_cv.pdf", upper)

    assert "Amina Benali" in extract_text_from_file(str(upper))


# --- The rejection paths -----------------------------------------------------

def test_scanned_pdf_with_no_text_layer_is_caught_by_has_extractable_text():
    """A scan opens fine as a PDF but holds zero text. This is v1's ENTIRE handling
    of non-text-layer PDFs — rejected as invalid input, no OCR fallback."""
    text = extract_text_from_file(str(FIXTURES / "no_text_layer.pdf"))

    assert text == ""
    assert has_extractable_text(text) is False


def test_corrupted_pdf_raises():
    """A damaged file fails inside the PDF parser, NOT with a ValueError.

    Pinned deliberately: the upload-validation code in CV preprocessing has to
    catch this exception type too, or a corrupt upload becomes a 500 error.
    """
    with pytest.raises(RuntimeError):  # pymupdf raises FileDataError, a RuntimeError subclass
        extract_text_from_file(str(FIXTURES / "corrupted.pdf"))


def test_unsupported_extension_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported file type: .txt"):
        extract_text_from_file(str(FIXTURES / "unsupported.txt"))


# --- has_extractable_text on its own -----------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("", False),                    # scanned PDF
        ("x" * 49, False),              # just under the threshold
        ("x" * 50, True),               # exactly the threshold — inclusive (>=)
        ("x" * 51, True),
    ],
)
def test_has_extractable_text_threshold(text, expected):
    assert has_extractable_text(text) is expected


def test_has_extractable_text_threshold_is_overridable():
    assert has_extractable_text("short", min_chars=3) is True
    assert has_extractable_text("short", min_chars=99) is False
