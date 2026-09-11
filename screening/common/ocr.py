"""Tesseract plumbing: the one place in the project that talks to the OCR engine.

`text_extraction` decides *what* to OCR — a PDF page that came back blank, a picture
pasted into a Word file, an uploaded scan. This module knows *how*.

Nothing here raises. OCR is a fallback, so a machine without Tesseract must still
process the CVs that do carry a text layer: every failure path returns "" and logs,
and `ocr_available()` reports whether to expect anything at all.

The settings below are constants rather than environment variables. They are tuning
values, not deployment ones — the same numbers are right on every machine — so they
live here where they can be read next to the code that uses them. The exception is
`TESSERACT_CMD`, which really does differ per machine and is in `.env.example`.
"""

import io
import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageSequence

logger = logging.getLogger(__name__)

# Same reasoning as screening/common/llm.py: load the project's .env at import so the
# web server, the stage subprocesses, pytest and a bare script are all configured
# identically, and a real shell variable still wins over the file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Full path to the binary, for a Windows install that is not on PATH.
TESSERACT_CMD = (os.getenv("TESSERACT_CMD") or "").strip()

# Language packs to read with, "+"-joined. A pack that is not installed is dropped
# rather than failing the call — see ocr_languages().
OCR_LANGUAGES = "eng+fra"

# Resolution for rendering PDF pages. 300 is what Tesseract is trained for; 150 is
# visibly worse.
OCR_DPI = 300

# A page with less text than this counts as a scan and gets OCR'd.
OCR_MIN_CHARS_PER_PAGE = 100

# Cap on pages OCR'd per file, so one huge document cannot tie up an upload.
OCR_MAX_PAGES = 30

# Below this, an image is upscaled before OCR: Tesseract reads ~300 DPI well and
# small images badly, and a CV photographed on a phone at a distance lands here.
MIN_OCR_PIXELS = 1200

# A ceiling on the upscaled image, so a pathological input cannot exhaust memory.
MAX_OCR_PIXELS = 40_000_000

# Below this, Tesseract's orientation guess is noise and is ignored. An upright page
# scores ~13 in practice; a guess from an image too small to judge scores under 1.
MIN_ORIENTATION_CONFIDENCE = 2.0

# Cached engine probe. Reset with reset_ocr_cache() — the tests do, after changing
# the module's configuration.
_state: dict = {"probed": False, "available": False, "version": "", "languages": None}


def _pytesseract():
    """The pytesseract module, with the binary path applied if one is configured.

    Imported lazily so that importing this module never fails on a machine where
    the wrapper is missing — `ocr_available()` reports that instead.
    """
    import pytesseract

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    return pytesseract


def ocr_available() -> bool:
    """True when Tesseract is installed and reachable.

    Probed once and cached: it shells out to the binary, and the CV upload path
    would otherwise pay for that on every page of every file.
    """
    if not _state["probed"]:
        _state["probed"] = True
        try:
            version = _pytesseract().get_tesseract_version()
        except Exception as exc:  # noqa: BLE001 — ImportError, TesseractNotFound, OSError…
            _state["available"] = False
            logger.warning(
                "OCR unavailable (%s: %s). Files with a text layer still work; scanned "
                "ones will come back empty. Install Tesseract, or set TESSERACT_CMD to "
                "its full path if it is installed but not on PATH.",
                type(exc).__name__, exc,
            )
        else:
            _state["available"] = True
            _state["version"] = str(version)
            logger.info("OCR ready: Tesseract %s, languages %r.", version, ocr_languages())
    return _state["available"]


def ocr_version() -> str:
    """The engine version as a string, or "" when OCR is unavailable."""
    return _state["version"] if ocr_available() else ""


def installed_languages() -> list[str]:
    """Every language pack Tesseract can see. Empty when it cannot be asked."""
    try:
        return sorted(_pytesseract().get_languages(config=""))
    except Exception:  # noqa: BLE001 — old pytesseract, or no binary
        return []


def ocr_languages() -> str:
    """The language string to actually pass to Tesseract, narrowed to what is installed.

    Tesseract fails the *whole* call if any requested pack is missing, so asking for
    "eng+fra" on a machine with only English would lose every scanned CV rather than
    just the French wording. Resolved once and cached.
    """
    if _state["languages"] is None:
        wanted = [lang for lang in OCR_LANGUAGES.split("+") if lang.strip()]
        installed = set(installed_languages())
        usable = [lang for lang in wanted if lang in installed] if installed else list(wanted)
        if installed and len(usable) < len(wanted):
            missing = [lang for lang in wanted if lang not in installed]
            logger.warning(
                "OCR language pack(s) %s are not installed — continuing with %s. "
                "Install them (e.g. tesseract-ocr-fra) for accented text.",
                "+".join(missing), "+".join(usable) or "eng",
            )
        _state["languages"] = "+".join(usable) or "eng"
    return _state["languages"]


def reset_ocr_cache() -> None:
    """Forget the probe. Only needed when the configuration changes at runtime — tests."""
    _state.update({"probed": False, "available": False, "version": "", "languages": None})


def prepare_image(image: Image.Image) -> Image.Image:
    """Normalize any image into what Tesseract reads best.

    Three things, each earned by a real failure mode:
      * EXIF transpose — a CV photographed on a phone is stored sideways with an
        orientation flag, and Tesseract reads pixels, not flags;
      * grayscale — colour carries no glyph information and triples the work;
      * upscale — a small screenshot or a low-DPI scan is unreadable at native size.
    """
    try:
        image = ImageOps.exif_transpose(image) or image
    except Exception:  # noqa: BLE001 — a broken EXIF block must not lose the image
        pass
    if image.mode != "L":
        image = image.convert("L")

    width, height = image.size
    smallest = min(width, height)
    if smallest and smallest < MIN_OCR_PIXELS:
        factor = min(4, math.ceil(MIN_OCR_PIXELS / smallest))
        while factor > 1 and width * height * factor * factor > MAX_OCR_PIXELS:
            factor -= 1
        if factor > 1:
            image = image.resize((width * factor, height * factor), Image.LANCZOS)
    return image


def _detect_rotation(image: Image.Image) -> int:
    """How many degrees clockwise this image must turn to be upright. 0 if it already is.

    Runs at full resolution on purpose: Tesseract's orientation pass needs the glyph
    detail, and on a downscaled copy it starts failing outright on sideways pages —
    precisely the case it exists to catch.
    """
    pytesseract = _pytesseract()
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = int(osd.get("rotate", 0)) % 360
        confidence = float(osd.get("orientation_conf", 0) or 0)
    except Exception:  # noqa: BLE001 — no osd traineddata, or too little text to judge
        return 0
    if angle and confidence < MIN_ORIENTATION_CONFIDENCE:
        logger.debug("Ignoring a %s° orientation guess at confidence %.1f.", angle, confidence)
        return 0
    return angle


def image_to_text(image: Image.Image) -> str:
    """Read one already-prepared image. Returns "" rather than raising, always.

    Orientation is checked before recognition, not after. A page fed in sideways does
    not come back short — Tesseract confidently returns a full page of nonsense — so
    there is nothing to notice afterwards, and a scanner that took a stack in rotated
    would otherwise poison a whole batch silently.
    """
    if not ocr_available():
        return ""
    angle = _detect_rotation(image)
    if angle:
        logger.info("Rotating a page %s° clockwise before OCR.", angle)
        # OSD reports the clockwise turn needed; PIL rotates counter-clockwise.
        image = image.rotate(-angle, expand=True, fillcolor=255)
    return _run_tesseract(image)


def _run_tesseract(image: Image.Image) -> str:
    pytesseract = _pytesseract()
    try:
        return (pytesseract.image_to_string(image, lang=ocr_languages()) or "").strip()
    except Exception as exc:  # noqa: BLE001 — one unreadable page must not sink the file
        logger.warning("OCR failed on one image (%s: %s).", type(exc).__name__, exc)
        return ""


def image_file_to_text(file_path: str) -> str:
    """OCR an image file on disk, every frame of it.

    Multi-frame TIFF is what a sheet-fed scanner produces for a two-page CV, so the
    frames are read and joined rather than only the first one.
    """
    if not ocr_available():
        return ""
    pages = []
    with Image.open(file_path) as opened:
        frames = getattr(opened, "n_frames", 1)
        if frames > 1:
            for index, frame in enumerate(ImageSequence.Iterator(opened)):
                if index >= OCR_MAX_PAGES:
                    logger.warning("Stopping at %s frames of %s.", OCR_MAX_PAGES, file_path)
                    break
                pages.append(image_to_text(prepare_image(frame)))
        else:
            pages.append(image_to_text(prepare_image(opened)))
    return "\n".join(page for page in pages if page)


def image_bytes_to_text(data: bytes) -> str:
    """OCR an image held in memory — a picture unzipped out of a .docx."""
    if not ocr_available():
        return ""
    try:
        with Image.open(io.BytesIO(data)) as opened:
            return image_to_text(prepare_image(opened))
    except Exception as exc:  # noqa: BLE001 — EMF/WMF and other formats PIL cannot open
        logger.debug("Skipping an embedded image PIL could not open (%s).", type(exc).__name__)
        return ""
