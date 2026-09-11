"""Tests for the shared text-extraction function (screening/common/text_extraction.py).

This function is the single entry point for BOTH JD file uploads and CV uploads,
so a bug here corrupts every downstream stage. Each fixture pins one behaviour;
see fixtures/README.md for what each file represents.

Extraction is OCR-backed. The tests that need a real Tesseract are marked
`needs_ocr` and skip themselves when it is not installed, so the suite stays green
on a machine without it — which is also the promise the module makes to the app:
no Tesseract means degraded, never broken.
"""

import shutil
from pathlib import Path

import pymupdf as fitz
import pytest

from screening.common import ocr
from screening.common.text_extraction import (
    TextExtractionError,
    extract_text_and_images_from_pdf,
    extract_text_from_file,
    has_extractable_text,
)

FIXTURES = Path(__file__).parent / "fixtures"

needs_ocr = pytest.mark.skipif(
    not ocr.ocr_available(), reason="Tesseract is not installed on this machine"
)


@pytest.fixture(autouse=True)
def _fresh_ocr_probe():
    """The engine probe is cached, so a test that changes the configuration must not
    leave that cache behind for the next one."""
    yield
    ocr.reset_ocr_cache()


def without_tesseract(monkeypatch):
    """Simulate a machine where the binary is not installed.

    Makes the engine probe itself fail, which is the real path a machine without
    Tesseract takes — rather than flipping a flag that only exists for tests.
    """
    def missing():
        raise OSError("tesseract is not installed")

    monkeypatch.setattr(ocr, "_pytesseract", missing)
    ocr.reset_ocr_cache()

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


# --- OCR: the files that have no text layer at all ----------------------------

@needs_ocr
def test_a_scanned_pdf_is_read_by_ocr():
    """The case v1 rejected outright. Every page is a picture; nothing has a text layer."""
    document = fitz.open(str(FIXTURES / "scanned_cv.pdf"))
    assert [page.get_text().strip() for page in document] == ["", ""]  # no text layer at all
    document.close()

    text = extract_text_from_file(str(FIXTURES / "scanned_cv.pdf"))

    assert "KARIM ZAHRAOUI" in text
    assert "Nador Steelworks" in text          # page 1
    assert "Mechanical Engineering" in text    # page 2 — both pages get OCR'd, not just the first
    assert has_extractable_text(text)


@needs_ocr
def test_a_hybrid_pdf_ocrs_only_the_page_that_needs_it():
    """A typed first page and a scanned second one — the reason the OCR decision is
    made per page rather than per document."""
    text = extract_text_from_file(str(FIXTURES / "hybrid_cv.pdf"))

    assert "Northwind Logistics" in text       # page 1, from the text layer
    assert "Certified Scrum Master" in text    # page 2, only reachable through OCR


@needs_ocr
def test_an_image_file_is_read_by_ocr():
    """A photo or a scanner dump of a printed CV, with no PDF wrapper at all."""
    text = extract_text_from_file(str(FIXTURES / "scanned_cv.png"))

    assert "KARIM ZAHRAOUI" in text
    assert "Atlas Foundry" in text
    assert has_extractable_text(text)


@needs_ocr
def test_a_sideways_scan_is_turned_upright_before_reading():
    """A page fed into the scanner rotated does NOT come back short — Tesseract returns
    a confident page of nonsense — so orientation is checked before recognition."""
    text = extract_text_from_file(str(FIXTURES / "rotated_cv.png"))

    assert "KARIM ZAHRAOUI" in text
    assert "Maintenance Engineer" in text


@needs_ocr
def test_a_scan_pasted_into_word_is_read_by_ocr():
    """The other half of the scanned-CV problem: the picture arrives inside a .docx."""
    text = extract_text_from_file(str(FIXTURES / "scanned_cv.docx"))

    assert "KARIM ZAHRAOUI" in text
    assert "Nador Steelworks" in text


def test_a_text_layer_is_preferred_over_ocr():
    """OCR only replaces a page's own text when it clearly recovers more.

    clean_cv.pdf's second page holds 91 characters — under the threshold, so it is
    OCR'd — but its text layer is exact and OCR's reading of it is a guess. Pinned by
    asserting the page comes back byte-for-byte as the PDF stores it.
    """
    text = extract_text_from_file(str(FIXTURES / "clean_cv.pdf"))

    assert "EDUCATION\nMSc Computer Science, ENSIAS, 2021\nSKILLS" in text



def test_extraction_degrades_rather_than_fails_without_tesseract(monkeypatch):
    """The promise the module makes to the app: no Tesseract installed means scans come
    back empty and get rejected with a reason — never an exception, and never a CV with
    a text layer failing to import."""
    without_tesseract(monkeypatch)

    assert extract_text_from_file(str(FIXTURES / "scanned_cv.pdf")) == ""
    assert "Amina Benali" in extract_text_from_file(str(FIXTURES / "clean_cv.pdf"))


def test_an_image_file_without_ocr_says_so(monkeypatch):
    """An image has no text layer to fall back on, so this is not "empty file" — it is a
    server configuration problem, and the reason has to say which."""
    without_tesseract(monkeypatch)

    with pytest.raises(TextExtractionError, match="Tesseract is not available"):
        extract_text_from_file(str(FIXTURES / "scanned_cv.png"))


# --- DOCX: the shapes python-docx's paragraph list does not reach -------------

def test_docx_tables_text_boxes_and_headers_are_all_extracted():
    """A CV template whose whole body is a table, with contact details in the page
    header and skills in a text box. `document.paragraphs` sees one word of it.

    This needs no OCR — it is a plain Word file — and it was the single biggest hole
    in the old extractor: the file looks perfect in Word and arrived as 7 characters.
    """
    text = extract_text_from_file(str(FIXTURES / "table_cv.docx"))

    assert "Senior Accountant, Ibn Batouta Group" in text     # table cell
    assert "Anouar Fassi" in text                             # table nested inside a cell
    assert "Arabic (native), French (fluent)" in text         # text box
    assert "nadia.cherkaoui@example.com" in text              # page header
    assert has_extractable_text(text)


# --- The rejection paths -----------------------------------------------------

def test_a_pdf_with_no_glyphs_still_yields_nothing():
    """A page of drawn rectangles is not a scan of anything — OCR runs and finds no
    text, which is the correct answer. `has_extractable_text` then rejects it."""
    text = extract_text_from_file(str(FIXTURES / "no_text_layer.pdf"))

    assert text == ""
    assert has_extractable_text(text) is False


def test_corrupted_pdf_raises():
    """A damaged file fails inside the PDF parser, NOT with a ValueError.

    Pinned deliberately: the upload-validation code in CV preprocessing has to
    catch this exception type too, or a corrupt upload becomes a 500 error.
    TextExtractionError subclasses RuntimeError so that catch keeps working.
    """
    with pytest.raises(RuntimeError):
        extract_text_from_file(str(FIXTURES / "corrupted.pdf"))

    with pytest.raises(TextExtractionError, match="unreadable PDF"):
        extract_text_from_file(str(FIXTURES / "corrupted.pdf"))


def test_a_password_protected_pdf_says_so(tmp_path):
    """Rendering it for OCR would fail just as reading it does, so it is named as its
    own reason — "could not read file: password-protected PDF" is actionable, and a
    recruiter can go and ask for an unlocked copy."""
    locked = tmp_path / "locked.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Amina Benali, Software Engineer")
    document.save(str(locked), encryption=fitz.PDF_ENCRYPT_AES_256,
                  user_pw="secret", owner_pw="secret")
    document.close()

    with pytest.raises(TextExtractionError, match="password-protected"):
        extract_text_from_file(str(locked))


def test_unsupported_extension_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported file type: .txt"):
        extract_text_from_file(str(FIXTURES / "unsupported.txt"))


# --- Page images --------------------------------------------------------------

def test_extract_text_and_images_returns_one_jpeg_per_page():
    """The variant that also hands back the rendered pages, for a caller that wants to
    show the page a value came from. Nothing in the pipeline uses it, so it must not
    cost anything for callers that don't: the plain function renders nothing."""
    text, images = extract_text_and_images_from_pdf(str(FIXTURES / "clean_cv.pdf"))

    assert "Amina Benali" in text
    assert len(images) == 2                          # one per page
    assert all(image.startswith(b"\xff\xd8\xff") for image in images)   # JPEG magic bytes


# --- has_extractable_text on its own -----------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("", False),                    # a scan OCR could not read
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
