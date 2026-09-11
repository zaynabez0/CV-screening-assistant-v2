"""`python manage.py check_ocr` — is OCR actually wired up on this machine?

Worth its own command because the failure it diagnoses is silent: with Tesseract
missing, every CV that carries a text layer still works perfectly, and only the
scans come back empty. That reads as "the tool rejected this CV" rather than
"the server is missing a package", which is a slow and confusing thing to find
out on a batch of fifty.
"""

from pathlib import Path

from django.core.management.base import BaseCommand

from screening.common import ocr


class Command(BaseCommand):
    help = "Report whether OCR is available, and with which languages and settings."

    def add_arguments(self, parser):
        parser.add_argument(
            "file", nargs="?",
            help="Optionally, a PDF/DOCX/image to run extraction against as a live check.",
        )
        parser.add_argument(
            "--full", action="store_true",
            help="Print the whole extracted text instead of the first 12 lines.",
        )
        parser.add_argument(
            "--out", metavar="PATH",
            help="Write the extracted text to a UTF-8 file. This is what the LLM would "
                 "be sent, so it is the thing to read when a CV structures badly.",
        )

    def handle(self, *args, **options):
        if ocr.ocr_available():
            self.stdout.write(self.style.SUCCESS(f"Tesseract {ocr.ocr_version()} is available."))
            self.stdout.write(f"  languages in use : {ocr.ocr_languages()}")
            self.stdout.write(f"  installed packs  : {', '.join(ocr.installed_languages()) or '(none)'}")
            self.stdout.write(f"  render DPI       : {ocr.OCR_DPI}")
            self.stdout.write(f"  OCR below        : {ocr.OCR_MIN_CHARS_PER_PAGE} chars/page")
            self.stdout.write(f"  page cap         : {ocr.OCR_MAX_PAGES}")
        else:
            self.stdout.write(self.style.ERROR("Tesseract is NOT available."))
            self.stdout.write(
                "CVs with a text layer still work; scans and photos will be rejected with "
                "'no extractable text found'.\n"
                "Install the Tesseract binary (see README section 1), or set TESSERACT_CMD "
                "to its full path if it is installed but not on PATH."
            )

        path = options.get("file")
        if not path:
            return

        from screening.common.text_extraction import (
            TextExtractionError,
            extract_text_from_file,
            has_extractable_text,
        )

        self.stdout.write(f"\nReading {path} …")
        try:
            text = extract_text_from_file(path)
        except (ValueError, TextExtractionError) as exc:
            self.stdout.write(self.style.ERROR(f"  failed: {exc}"))
            return

        verdict = self.style.SUCCESS("usable") if has_extractable_text(text) else self.style.ERROR(
            "too little text — this file would be rejected at upload"
        )
        self.stdout.write(f"  {len(text)} characters — {verdict}")

        if options.get("out"):
            # UTF-8 explicitly: a Windows console defaults to cp1252 and drops the very
            # accented characters this file is most likely being inspected for.
            Path(options["out"]).write_text(text, encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"  written to {options['out']}"))

        lines = text.splitlines()
        shown = lines if options.get("full") else lines[:12]
        if shown:
            self.stdout.write("\n  " + "\n  ".join(shown))
        if len(shown) < len(lines):
            self.stdout.write(self.style.WARNING(
                f"\n  … {len(lines) - len(shown)} more line(s). Use --full to print all of "
                f"it, or --out extracted.txt to read it in an editor."
            ))
