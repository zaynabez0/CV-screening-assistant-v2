"""Tests for the CV upload validation chain (screening/cv/validation.py).

The five checks run in CLAUDE.md's order, and the order is itself load-bearing: a
rejection reason must name the earliest thing wrong with a file, not a downstream
symptom. `test_extension_is_checked_before_content` pins that.
"""

import shutil
from pathlib import Path

import pytest

from screening.cv import validation
from screening.cv.validation import (
    CVValidationError,
    DuplicateFileError,
    check_batch_size,
    check_file_size,
    compute_file_hash,
    validate_cv_file,
)

FIXTURES = Path(__file__).parent / "fixtures"


# --- Files that should pass all five checks ----------------------------------

@pytest.mark.parametrize("filename", ["clean_cv.pdf", "clean_cv.docx"])
def test_clean_files_pass_validation(filename):
    result = validate_cv_file(str(FIXTURES / filename), filename)

    assert len(result["text"]) > 50
    assert len(result["file_hash"]) == 64          # sha256 hex
    assert result["size_bytes"] > 0


# --- Step 1: extension whitelist ---------------------------------------------

def test_unsupported_extension_is_rejected():
    with pytest.raises(CVValidationError, match="unsupported file type"):
        validate_cv_file(str(FIXTURES / "unsupported.txt"), "unsupported.txt")


def test_file_with_no_extension_is_rejected():
    with pytest.raises(CVValidationError, match="unsupported file type"):
        validate_cv_file(str(FIXTURES / "clean_cv.pdf"), "resume_no_extension")


# --- Step 2: magic bytes, not the filename -----------------------------------

def test_docx_disguised_as_pdf_is_rejected():
    """The whole point of the magic-byte check: the name says .pdf, the bytes say DOCX."""
    with pytest.raises(CVValidationError, match="does not match its .pdf extension"):
        validate_cv_file(str(FIXTURES / "disguised.pdf"), "disguised.pdf")


def test_extension_is_checked_before_content():
    """Order matters. A .txt file whose bytes are a valid PDF must be rejected for its
    EXTENSION, not for its content — otherwise the reason misleads the recruiter."""
    txt_named_copy = FIXTURES / "unsupported.txt"
    with pytest.raises(CVValidationError, match="unsupported file type"):
        validate_cv_file(str(FIXTURES / "clean_cv.pdf"), str(txt_named_copy.name))


# --- Step 3: size limits ------------------------------------------------------

def test_file_below_minimum_size_is_rejected(tmp_path):
    tiny = tmp_path / "tiny.pdf"
    tiny.write_bytes(b"%PDF-1.4\n")
    with pytest.raises(CVValidationError, match="file too small"):
        check_file_size(str(tiny))


def test_file_above_maximum_size_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "MAX_FILE_BYTES", 100)
    big = tmp_path / "big.pdf"
    big.write_bytes(b"x" * 500)
    with pytest.raises(CVValidationError, match="file too large"):
        check_file_size(str(big))


def test_batch_over_the_total_limit_is_rejected():
    check_batch_size(validation.MAX_BATCH_BYTES)            # exactly at the limit is fine
    with pytest.raises(CVValidationError, match="batch too large"):
        check_batch_size(validation.MAX_BATCH_BYTES + 1)


# --- Step 4: extractable text -------------------------------------------------

def test_scanned_pdf_is_rejected_with_the_specified_reason():
    """CLAUDE.md names this reason string exactly. It is v1's entire handling of
    scanned documents — no OCR fallback."""
    with pytest.raises(CVValidationError, match="no extractable text found"):
        validate_cv_file(str(FIXTURES / "no_text_layer.pdf"), "no_text_layer.pdf")


def test_corrupted_pdf_is_rejected_not_crashed():
    """A damaged PDF raises FileDataError (a RuntimeError) deep inside pymupdf.

    This test is the reason validate_cv_file catches RuntimeError as well as
    ValueError: without it, one corrupt upload becomes a 500 error page and takes
    the whole batch down with it.
    """
    with pytest.raises(CVValidationError, match="could not read file"):
        validate_cv_file(str(FIXTURES / "corrupted.pdf"), "corrupted.pdf")


# --- Step 5: dedup by sha256 --------------------------------------------------

def test_hash_is_stable_and_content_based(tmp_path):
    renamed = tmp_path / "totally_different_name.pdf"
    shutil.copy(FIXTURES / "clean_cv.pdf", renamed)

    assert compute_file_hash(str(renamed)) == compute_file_hash(str(FIXTURES / "clean_cv.pdf"))


def test_different_files_hash_differently():
    assert compute_file_hash(str(FIXTURES / "clean_cv.pdf")) != compute_file_hash(
        str(FIXTURES / "clean_cv.docx")
    )


def test_duplicate_is_rejected_with_its_own_exception_type():
    """DuplicateFileError so the caller can set status="duplicate" rather than
    "rejected" without string-matching the reason."""
    known = compute_file_hash(str(FIXTURES / "clean_cv.pdf"))

    with pytest.raises(DuplicateFileError, match="duplicate"):
        validate_cv_file(str(FIXTURES / "clean_cv.pdf"), "clean_cv.pdf", seen_hashes={known})


def test_a_resubmitted_file_under_a_new_name_is_still_a_duplicate(tmp_path):
    """Renaming a CV must not sneak it past dedup — the hash is over bytes, not names."""
    renamed = tmp_path / "candidate_final_v2.pdf"
    shutil.copy(FIXTURES / "clean_cv.pdf", renamed)
    known = compute_file_hash(str(FIXTURES / "clean_cv.pdf"))

    with pytest.raises(DuplicateFileError):
        validate_cv_file(str(renamed), "candidate_final_v2.pdf", seen_hashes={known})


def test_dedup_runs_last_so_a_broken_duplicate_reports_the_real_problem():
    """A corrupt file that is ALSO a duplicate should say it's unreadable — the
    recruiter needs the actionable reason, not the incidental one."""
    known = compute_file_hash(str(FIXTURES / "corrupted.pdf"))

    with pytest.raises(CVValidationError, match="could not read file"):
        validate_cv_file(str(FIXTURES / "corrupted.pdf"), "corrupted.pdf", seen_hashes={known})
