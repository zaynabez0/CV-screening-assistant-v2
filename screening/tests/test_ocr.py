"""Tests for the OCR plumbing (screening/common/ocr.py).

Deliberately Tesseract-free: every test here fakes the engine. The point is the
*policy* around it — what happens when it is missing, when a language pack is not
installed, when the orientation guess is worthless — and that policy has to hold on
machines where the real engine cannot be made to misbehave on demand.

The end-to-end "does it actually read a scan" tests live in test_text_extraction.py
and skip themselves without a real Tesseract.
"""

import pytest
from PIL import Image

from screening.common import ocr


class FakeTesseract:
    """Stands in for the pytesseract module. Records what it was asked for."""

    class Output:
        DICT = "dict"

    class pytesseract:  # noqa: N801 — mirrors pytesseract.pytesseract.tesseract_cmd
        tesseract_cmd = "tesseract"

    def __init__(self, version="5.3.4", languages=("eng", "fra", "osd"), osd=None, text="page text"):
        self.version, self.languages, self.osd, self.text = version, languages, osd, text
        self.calls = []

    def get_tesseract_version(self):
        if self.version is None:
            raise OSError("tesseract is not installed")
        return self.version

    def get_languages(self, config=""):
        return list(self.languages)

    def image_to_osd(self, image, output_type=None):
        if self.osd is None:
            raise RuntimeError("Too few characters to determine orientation")
        return self.osd

    def image_to_string(self, image, lang=None, config=None):
        self.calls.append({"size": image.size, "lang": lang, "config": config})
        return self.text


@pytest.fixture(autouse=True)
def _fresh_probe():
    ocr.reset_ocr_cache()
    yield
    ocr.reset_ocr_cache()


@pytest.fixture
def engine(monkeypatch):
    """Install a fake engine and hand it back so tests can assert on the calls.

    OCR_ENABLED is forced on: these tests are about what happens around the engine,
    and must not change their answer because the machine running them has OCR switched
    off in its own .env.
    """
    def install(**kwargs):
        fake = FakeTesseract(**kwargs)
        monkeypatch.setattr(ocr, "OCR_ENABLED", True)
        monkeypatch.setattr(ocr, "_pytesseract", lambda: fake)
        return fake
    return install


def page(size=(2000, 2800), colour="white"):
    return Image.new("RGB", size, colour)


# --- Is OCR there at all ------------------------------------------------------

def test_a_missing_tesseract_is_reported_not_raised(engine):
    engine(version=None)

    assert ocr.ocr_available() is False
    assert ocr.image_to_text(page()) == ""      # and reading still returns a value


def test_ocr_can_be_disabled_without_touching_the_engine(engine, monkeypatch):
    fake = engine()
    monkeypatch.setattr(ocr, "OCR_ENABLED", False)

    assert ocr.ocr_available() is False
    assert ocr.image_to_text(page()) == ""
    assert fake.calls == []


def test_the_engine_is_probed_once(engine, monkeypatch):
    """The probe shells out to the binary. Uploading fifty CVs must not do that per page."""
    fake = engine()
    probes = []
    monkeypatch.setattr(fake, "get_tesseract_version", lambda: probes.append(1) or "5.3.4")

    assert [ocr.ocr_available() for _ in range(5)] == [True] * 5
    assert len(probes) == 1


# --- Language packs -----------------------------------------------------------

def test_missing_language_packs_are_dropped_rather_than_failing_the_file(engine):
    """Tesseract aborts the whole image if any requested pack is missing, so asking for
    "eng+fra" on an English-only machine would lose every scanned CV, not just the
    French wording."""
    engine(languages=("eng",))

    assert ocr.ocr_languages() == "eng"


def test_configured_languages_are_used_when_they_are_installed(engine, monkeypatch):
    engine(languages=("eng", "fra", "spa"))
    monkeypatch.setattr(ocr, "OCR_LANGUAGES", "fra+spa")

    assert ocr.ocr_languages() == "fra+spa"


def test_english_is_the_last_resort(engine, monkeypatch):
    engine(languages=("deu",))
    monkeypatch.setattr(ocr, "OCR_LANGUAGES", "fra")

    assert ocr.ocr_languages() == "eng"


def test_the_language_string_reaches_the_engine(engine, monkeypatch):
    fake = engine(languages=("eng", "fra"))
    monkeypatch.setattr(ocr, "OCR_CONFIG", "--psm 4")

    ocr.image_to_text(page(), detect_orientation=False)

    assert fake.calls == [{"size": (2000, 2800), "lang": "eng+fra", "config": "--psm 4"}]


# --- Orientation --------------------------------------------------------------

def test_a_sideways_page_is_rotated_before_it_is_read(engine):
    fake = engine(osd={"rotate": 90, "orientation_conf": 12.0})

    ocr.image_to_text(page(size=(2800, 2000)))

    assert fake.calls[0]["size"] == (2000, 2800)   # turned upright, so w/h swapped


def test_a_low_confidence_orientation_guess_is_ignored(engine):
    """Below the confidence floor the guess is noise, and acting on it would rotate a
    page that was already upright."""
    fake = engine(osd={"rotate": 90, "orientation_conf": 0.4})

    ocr.image_to_text(page(size=(2800, 2000)))

    assert fake.calls[0]["size"] == (2800, 2000)   # left exactly as it came


def test_an_unavailable_orientation_check_does_not_stop_the_read(engine):
    """image_to_osd fails on a page with too little text, and on a machine with no
    osd traineddata. Neither is a reason to skip reading the page."""
    fake = engine(osd=None)

    assert ocr.image_to_text(page()) == "page text"
    assert len(fake.calls) == 1


def test_orientation_detection_can_be_switched_off(engine, monkeypatch):
    fake = engine(osd={"rotate": 90, "orientation_conf": 12.0})
    monkeypatch.setattr(ocr, "OCR_DETECT_ORIENTATION", False)

    ocr.image_to_text(page(size=(2800, 2000)))

    assert fake.calls[0]["size"] == (2800, 2000)


# --- Preparing the image ------------------------------------------------------

def test_colour_is_dropped():
    """Colour carries no glyph information and triples the work."""
    assert ocr.prepare_image(page(colour="red")).mode == "L"


def test_a_small_image_is_upscaled():
    """A screenshot or a low-DPI scan is unreadable at native size."""
    prepared = ocr.prepare_image(page(size=(400, 560)))

    assert min(prepared.size) >= ocr.MIN_OCR_PIXELS


def test_a_large_image_is_left_alone():
    prepared = ocr.prepare_image(page(size=(2000, 2800)))

    assert prepared.size == (2000, 2800)


def test_upscaling_stays_under_the_memory_ceiling():
    """A long thin sliver would otherwise be scaled into hundreds of megapixels."""
    prepared = ocr.prepare_image(page(size=(300, 20000)))

    assert prepared.size[0] * prepared.size[1] <= ocr.MAX_OCR_PIXELS


# --- Reading a file that PIL cannot open --------------------------------------

def test_an_unopenable_embedded_image_is_skipped(engine):
    """A .docx can hold EMF/WMF drawings PIL knows nothing about. One of those must not
    take down the OCR of the pictures around it."""
    engine()

    assert ocr.image_bytes_to_text(b"not an image at all") == ""
