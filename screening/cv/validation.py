"""Upload validation for CV files — five checks, run in a fixed order.

The order is load-bearing: a rejection must name the earliest thing wrong with a file,
not a downstream symptom of it. See the checks below, and test_cv_validation.py.

Pure Python: no Django, no database. The dedup check (step 5) needs to know which
hashes already exist for the run, so the caller passes them in — that keeps the
database lookup on the Django side and the comparison here.

Every rejection raises CVValidationError carrying a human-readable `reason`, which
the caller stores verbatim on Candidate.rejection_reason. GDPR-wise this matters:
a rejected file must always be able to say *why* it was rejected.
"""

import hashlib
import os

import magic

from screening.common.ocr import ocr_available
from screening.common.text_extraction import (
    TextExtractionError,
    extract_text_from_file,
    has_extractable_text,
)

# Step 1 — extension whitelist. PDF, DOCX, and the image formats a scanner or a phone
# produces: extraction OCRs anything without a text layer, so a scanned CV is now an
# ordinary input rather than a rejection.
ALLOWED_EXTENSIONS = {"pdf", "docx", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}

# Step 2 — what the file must actually BE, by magic bytes, regardless of its name.
# Several MIME strings per image type on purpose: libmagic's answer for the same
# bytes differs by version and platform (BMP is the usual offender).
ALLOWED_MIME_TYPES = {
    "pdf": {"application/pdf"},
    "docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    "png": {"image/png"},
    "jpg": {"image/jpeg"},
    "jpeg": {"image/jpeg"},
    "tif": {"image/tiff"},
    "tiff": {"image/tiff"},
    "bmp": {"image/bmp", "image/x-ms-bmp", "image/x-bmp"},
    "webp": {"image/webp"},
}

# Step 3 — size limits. Starting values, tune against real HR batches:
#   min  catches only EMPTY/truncated uploads. Deliberately low: nothing under ~100
#        bytes can be a structurally valid PDF or DOCX, but a small valid PDF can be
#        well under 1 KB. Judging whether a file holds enough *content* is step 4's
#        job (has_extractable_text, 50 chars) — duplicating that judgement here just
#        produces a misleading rejection reason ("too small" for a readable file).
#   max  raised from 10 MB when OCR arrived: a 10 MB ceiling was fine while scans were
#        rejected on principle, but a colour scan of a three-page CV at 300 DPI lands
#        between 10 and 20 MB routinely, and rejecting it for its size would undo the
#        thing OCR is here to do
#   batch guards against one upload filling the disk
MIN_FILE_BYTES = 100
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_BATCH_BYTES = 200 * 1024 * 1024

_HASH_CHUNK_BYTES = 65536


class CVValidationError(ValueError):
    """An uploaded file failed validation. `reason` is shown to the recruiter and stored."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class DuplicateFileError(CVValidationError):
    """Step 5 failure. Separate from the others so the caller can set status="duplicate"
    rather than "rejected" without matching on the reason string."""


def check_extension(filename: str) -> str:
    """Step 1. Returns the lowercase extension so later steps don't re-parse it."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        accepted = ", ".join("." + e for e in sorted(ALLOWED_EXTENSIONS))
        raise CVValidationError(
            f"unsupported file type: .{ext or 'none'} (accepted: {accepted})"
        )
    return ext


def check_magic_type(file_path: str, ext: str) -> None:
    """Step 2. Trust the bytes, not the filename — a .docx renamed to .pdf is caught here."""
    detected = magic.from_file(file_path, mime=True)
    if detected not in ALLOWED_MIME_TYPES[ext]:
        raise CVValidationError(
            f"file content does not match its .{ext} extension (detected: {detected})"
        )


def check_file_size(file_path: str) -> int:
    """Step 3, per file. Returns the size so the caller can total a batch."""
    size = os.path.getsize(file_path)
    if size < MIN_FILE_BYTES:
        raise CVValidationError(f"file too small ({size} bytes, minimum {MIN_FILE_BYTES})")
    if size > MAX_FILE_BYTES:
        raise CVValidationError(f"file too large ({size} bytes, maximum {MAX_FILE_BYTES})")
    return size


def check_batch_size(total_bytes: int) -> None:
    """Step 3, per batch. Called once for the whole upload, before any file is processed."""
    if total_bytes > MAX_BATCH_BYTES:
        raise CVValidationError(
            f"upload batch too large ({total_bytes} bytes, maximum {MAX_BATCH_BYTES})"
        )


def compute_file_hash(file_path: str) -> str:
    """Step 5's input. sha256 over the raw bytes, read in chunks so a large file
    never lands in memory all at once."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cv_file(file_path: str, original_filename: str, seen_hashes=()) -> dict:
    """Run all five checks in order and return {"text", "file_hash", "size_bytes"}.

    `seen_hashes` is the set of file hashes already recorded for this run; the caller
    (a Django view) reads them from the database and passes them in.

    Raises CVValidationError on the first failure, so the reason names the earliest
    thing wrong with the file rather than a downstream symptom.
    """
    ext = check_extension(original_filename)
    check_magic_type(file_path, ext)
    size_bytes = check_file_size(file_path)

    # Step 4. A corrupted PDF fails inside the parser with FileDataError (a RuntimeError),
    # NOT a ValueError — catching only ValueError here would turn a damaged upload into
    # a 500 error page instead of a per-file rejection. TextExtractionError is caught by
    # the same clause (it subclasses RuntimeError) but named separately because it
    # carries a reason worth showing: "password-protected PDF" is actionable, the bare
    # exception type is not.
    #
    # This is where OCR happens, so it is the slow step: a scanned CV costs a second or
    # two per page here. That is deliberate — validation has to know whether a file
    # holds readable text before the recruiter is told it was accepted.
    try:
        text = extract_text_from_file(file_path)
    except TextExtractionError as exc:
        raise CVValidationError(f"could not read file: {exc}") from exc
    except (ValueError, RuntimeError) as exc:
        raise CVValidationError(f"could not read file: {type(exc).__name__}") from exc

    if not has_extractable_text(text):
        # The reason is the recruiter's only explanation, and "no extractable text" is
        # misleading on a server with no Tesseract: the file may be perfectly readable
        # and the *server* is what needs fixing. Say which it is.
        if not ocr_available():
            raise CVValidationError(
                "no extractable text found (this file has no text layer and OCR is not "
                "installed on the server)"
            )
        raise CVValidationError("no extractable text found")

    file_hash = compute_file_hash(file_path)
    if file_hash in seen_hashes:
        raise DuplicateFileError("duplicate of a file already uploaded to this run")

    return {"text": text, "file_hash": file_hash, "size_bytes": size_bytes}
