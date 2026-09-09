"""Shared text extraction — used by BOTH JD processing and CV preprocessing.

One function, two callers. The JD file-upload path and every CV go through this
exact code; there is deliberately no second version of it anywhere in the repo.

Extraction is OCR-backed, so a file no longer has to carry a text layer to be
readable. What each format does:

  * **PDF** — the embedded text layer first, read page by page. Any page holding
    fewer than `OCR_MIN_CHARS_PER_PAGE` characters is rendered at `OCR_DPI` and
    read with Tesseract instead. The decision is per *page*, not per document, so
    a hybrid CV — a typed first page, a scanned diploma on page 3 — comes back
    whole. Per page also means the text layer is preferred wherever it exists:
    it is exact, and OCR is a guess.
  * **DOCX** — paragraphs, tables (nested ones included), headers and footers, in
    document order; then a raw-XML sweep that catches text boxes, shapes and
    SmartArt, which python-docx's paragraph API does not see and which the
    designer-style CV templates lean on heavily; then, if the document is still
    essentially empty, OCR of the pictures inside it — which is what a scan
    pasted into Word actually is.
  * **Images** — png / jpg / jpeg / tif / tiff / bmp / webp, straight to OCR. A
    phone photo of a printed CV and a scanner dump are both ordinary HR inbox
    arrivals, and OCR makes them no harder than a PDF.

PDF pages are rasterized with PyMuPDF rather than pdf2image, which keeps Poppler
off the install list on every platform (on Windows it is a manual download-and-
add-to-PATH step). PyMuPDF is already a dependency for the text layer.

If Tesseract is not installed nothing here breaks: OCR is skipped, the text layer
is still returned, and `screening.common.ocr.ocr_available()` reports false so a
caller can explain why a scan came back empty.
"""

import io
import logging
import zipfile

import docx
import pymupdf as fitz  # `import fitz` is the deprecated spelling of the same module
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

from screening.common.ocr import (
    OCR_DPI,
    OCR_MAX_PAGES,
    OCR_MIN_CHARS_PER_PAGE,
    image_bytes_to_text,
    image_file_to_text,
    image_to_text,
    ocr_available,
    prepare_image,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}
SUPPORTED_EXTENSIONS = {"pdf", "docx"} | IMAGE_EXTENSIONS

# Quality of the per-page JPEGs from extract_text_and_images_from_pdf. 51 keeps a
# page legible to a human at roughly a tenth of the PNG size.
PAGE_JPEG_QUALITY = 51

# How much more text OCR must find before it is trusted over a page's own text layer.
# 1.0 would swap a perfectly good short page — a skills-only last page, say — for a
# guess at the same words; 1.25 means OCR has to have actually found something.
OCR_REPLACE_RATIO = 1.25

# How deep to follow tables inside table cells. CV templates nest two or three deep;
# the limit is only there so a malformed document cannot recurse forever.
MAX_TABLE_DEPTH = 5


class TextExtractionError(RuntimeError):
    """A supported file that could not be read at all — damaged, encrypted, empty.

    Subclasses RuntimeError deliberately. Both callers already catch
    (ValueError, RuntimeError) around this function, so a corrupt upload keeps
    turning into a per-file rejection rather than a 500 page, exactly as it did
    when the failure came raw out of pymupdf.
    """


# --- Public API ---------------------------------------------------------------

def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from a PDF, DOCX or image file, using OCR where needed.

    Raises ValueError for an unsupported extension, and TextExtractionError (a
    RuntimeError) when a supported file cannot be opened at all. A file that opens
    but holds no readable text returns "" — that is `has_extractable_text`'s call
    to make, not this function's.
    """
    ext = file_path.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        text, _ = _extract_pdf(file_path)
    elif ext == "docx":
        text = _extract_docx(file_path)
    elif ext in IMAGE_EXTENSIONS:
        text = _extract_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    return text.strip()


def extract_text_and_images_from_pdf(file_path: str) -> tuple[str, list[bytes]]:
    """The same PDF extraction, plus one grayscale JPEG per page.

    Nothing in the screening pipeline needs the images — every stage reads the
    structured JSON, never the file. It is here for callers that want to show the
    page a value was read from, or to send the page itself to a vision model.
    Rendering every page costs real time, so `extract_text_from_file` does not.
    """
    return _extract_pdf(file_path, want_images=True)


def has_extractable_text(text: str, min_chars: int = 50) -> bool:
    """False when a file yielded nothing usable — a scan OCR could not read, or a
    machine with no Tesseract installed at all. Either way the file cannot be
    screened, and the caller rejects it with a reason."""
    return len(text) >= min_chars


# --- PDF ----------------------------------------------------------------------

def _open_pdf(file_path: str) -> fitz.Document:
    try:
        document = fitz.open(file_path)
    except Exception as exc:  # noqa: BLE001 — FileDataError, EmptyFileError, OSError…
        raise TextExtractionError(f"unreadable PDF ({type(exc).__name__})") from exc

    # An owner-password PDF opens but refuses its pages; an empty user password is
    # the common "protected against editing, not reading" case and is worth trying.
    if document.needs_pass and not document.authenticate(""):
        document.close()
        raise TextExtractionError("password-protected PDF")
    return document


def _render_page(page: fitz.Page) -> Image.Image:
    """One PDF page as a grayscale PIL image at OCR_DPI, without going through PNG."""
    pixmap = page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csGRAY, alpha=False)
    return Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)


def _as_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=PAGE_JPEG_QUALITY)
    return buffer.getvalue()


def _extract_pdf(file_path: str, want_images: bool = False) -> tuple[str, list[bytes]]:
    """Page text, page by page, with OCR standing in for whichever pages need it."""
    document = _open_pdf(file_path)
    ocr_ready = ocr_available()
    pages: list[str] = []
    images: list[bytes] = []
    ocr_used = 0
    ocr_skipped = 0

    try:
        for number, page in enumerate(document, start=1):
            try:
                text = _clean(page.get_text())
            except Exception as exc:  # noqa: BLE001 — one damaged page, not the file
                logger.warning("Page %s of %s: %s. Falling back to OCR.",
                               number, file_path, type(exc).__name__)
                text = ""

            wants_ocr = len(text) < OCR_MIN_CHARS_PER_PAGE and ocr_ready
            if wants_ocr and ocr_used >= OCR_MAX_PAGES:
                wants_ocr = False
                ocr_skipped += 1
            if wants_ocr or want_images:
                try:
                    image = _render_page(page)
                except Exception as exc:  # noqa: BLE001 — a page that will not rasterize
                    logger.warning("Could not render page %s of %s (%s).",
                                   number, file_path, type(exc).__name__)
                else:
                    if want_images:
                        images.append(_as_jpeg(image))
                    if wants_ocr:
                        ocr_used += 1
                        recovered = _clean(image_to_text(prepare_image(image)))
                        # OCR only *replaces* a text layer when it clearly recovers more
                        # than was there — a page with a two-line heading over a scanned
                        # body must not keep the two lines and drop the body. A short but
                        # genuine page keeps its text layer, which is exact where OCR
                        # merely guesses.
                        if len(recovered) > len(text) * OCR_REPLACE_RATIO:
                            text = recovered
            pages.append(text)
    finally:
        document.close()

    if ocr_skipped:
        logger.warning("Stopped OCR at %s pages for %s, leaving %s page(s) unread — raise "
                       "OCR_MAX_PAGES if a CV is genuinely this long.",
                       OCR_MAX_PAGES, file_path, ocr_skipped)
    if not any(pages) and not ocr_ready:
        logger.warning("%s has no text layer and OCR is unavailable — nothing extracted.",
                       file_path)
    return "\n".join(page for page in pages if page), images


# --- DOCX ---------------------------------------------------------------------

def _extract_docx(file_path: str) -> str:
    """Body, tables, headers and footers — then the two fallbacks, in that order."""
    try:
        document = docx.Document(file_path)
    except Exception as exc:  # noqa: BLE001 — PackageNotFoundError, BadZipFile, OSError…
        raise TextExtractionError(f"unreadable DOCX ({type(exc).__name__})") from exc

    text = _clean("\n".join(_docx_blocks(document)))

    # A sweep of every text run in every part of the package. It loses table layout,
    # so it is only preferred when it finds *more* than the structured pass did —
    # which is exactly the text-box case that the structured pass cannot see.
    sweep = _clean(_docx_xml_sweep(file_path))
    if len(sweep) > len(text):
        text = sweep

    # Still empty: this is a scan someone pasted into Word. The pictures are the CV.
    if len(text) < OCR_MIN_CHARS_PER_PAGE and ocr_available():
        recovered = _clean("\n".join(_docx_image_text(file_path)))
        if recovered:
            text = _clean("\n".join(part for part in (text, recovered) if part))
    return text


def _docx_blocks(document) -> list[str]:
    """Every paragraph and table in document order, plus each section's header/footer.

    Order matters: python-docx's `document.paragraphs` silently skips tables, and a
    great many CV templates are *entirely* a table — one column of dates, one of
    descriptions. Reading only the paragraphs of one of those returns nothing.
    """
    lines = _element_lines(document.element.body, document)
    for section in document.sections:
        for name in ("header", "footer", "first_page_header", "first_page_footer",
                     "even_page_header", "even_page_footer"):
            try:
                part = getattr(section, name, None)
                if part is not None:
                    lines.extend(_element_lines(part._element, part))
            except Exception:  # noqa: BLE001 — a section without that part defined
                continue
    return [line for line in lines if line.strip()]


def _element_lines(element, parent, depth: int = 0) -> list[str]:
    lines: list[str] = []
    for child in element.iterchildren():
        if child.tag == qn("w:p"):
            lines.append(Paragraph(child, parent).text)
        elif child.tag == qn("w:tbl"):
            lines.extend(_table_lines(Table(child, parent), depth))
    return lines


def _table_lines(table, depth: int) -> list[str]:
    """One line per row, cells tab-separated, nested tables followed in place."""
    if depth >= MAX_TABLE_DEPTH:
        return [_xml_lines_text(table._tbl)]
    lines = []
    try:
        for row in table.rows:
            cells = [
                " ".join(_element_lines(cell._tc, cell, depth + 1)).strip()
                for cell in row.cells
            ]
            lines.append("\t".join(cell for cell in cells if cell))
    except Exception as exc:  # noqa: BLE001 — merged cells trip row.cells on some files
        logger.debug("Falling back to a raw read of one table (%s).", type(exc).__name__)
        return [_xml_lines_text(table._tbl)]
    return lines


def _docx_xml_sweep(file_path: str) -> str:
    """Every text run in every XML part of the package, in document order.

    Reads the .docx as the zip it is, so it picks up what python-docx's object model
    does not expose: text boxes, drawing shapes, SmartArt, footnotes and endnotes.
    """
    fragments = []
    try:
        with zipfile.ZipFile(file_path) as archive:
            names = sorted(
                name for name in archive.namelist()
                if name.startswith("word/") and name.endswith(".xml")
                and not name.startswith("word/theme/")
            )
            # document.xml first, then headers/footers/notes, so the CV body leads.
            names.sort(key=lambda name: (name != "word/document.xml", name))
            for name in names:
                try:
                    root = parse_xml(archive.read(name))
                except Exception:  # noqa: BLE001 — a part that is not parseable is skipped
                    continue
                fragment = _xml_lines_text(root)
                if fragment:
                    fragments.append(fragment)
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        logger.warning("Could not sweep %s for text (%s).", file_path, type(exc).__name__)
    return "\n".join(fragments)


def _xml_lines_text(root) -> str:
    """Text under an element, one line per w:p — tabs and breaks kept as themselves."""
    lines: list[str] = []
    current: list[str] = []
    for node in root.iter():
        tag = node.tag
        if tag == qn("w:p"):
            if current:
                lines.append("".join(current))
                current = []
        elif tag == qn("w:t"):
            current.append(node.text or "")
        elif tag == qn("w:tab"):
            current.append("\t")
        elif tag in (qn("w:br"), qn("w:cr")):
            current.append("\n")
    if current:
        lines.append("".join(current))
    return "\n".join(line for line in lines if line.strip())


def _docx_image_text(file_path: str) -> list[str]:
    """OCR of every picture stored in the document, in file order."""
    texts = []
    try:
        with zipfile.ZipFile(file_path) as archive:
            names = sorted(
                name for name in archive.namelist()
                if name.lower().startswith("word/media/")
                and name.lower().rsplit(".", 1)[-1] in IMAGE_EXTENSIONS
            )
            for name in names[:OCR_MAX_PAGES]:
                recovered = image_bytes_to_text(archive.read(name))
                if recovered:
                    texts.append(recovered)
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        logger.warning("Could not read images out of %s (%s).", file_path, type(exc).__name__)
    return texts


# --- Images -------------------------------------------------------------------

def _extract_image(file_path: str) -> str:
    """A scan or a photo: there is no text layer, so OCR is the whole path."""
    if not ocr_available():
        raise TextExtractionError(
            "image files need OCR, and Tesseract is not available on this machine"
        )
    try:
        return _clean(image_file_to_text(file_path))
    except TextExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001 — a truncated or unsupported image file
        raise TextExtractionError(f"unreadable image ({type(exc).__name__})") from exc


# --- Shared -------------------------------------------------------------------

def _clean(text: str) -> str:
    """Trim the noise OCR and PDF text layers both produce.

    Trailing spaces, stray NULs and runs of blank lines carry no information, cost
    tokens on every LLM call downstream, and make two extractions of the same CV
    compare unequal for no reason.
    """
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    blank = False
    for line in text.split("\n"):
        line = line.rstrip()
        if line.strip():
            lines.append(line)
            blank = False
        elif not blank and lines:
            lines.append("")
            blank = True
    return "\n".join(lines).strip()
