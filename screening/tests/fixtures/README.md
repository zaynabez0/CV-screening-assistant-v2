# Test fixtures — shared text extraction

Small binary files, committed on purpose so the test suite runs identically on
any machine with no setup step. They contain invented people and invented
career histories — **never put a real CV in this folder**, it would put personal
data in Git history permanently.

Each fixture pins one behaviour of `screening/common/text_extraction.py`:

| Fixture | Represents | What it proves |
|---|---|---|
| `clean_cv.pdf` | A normal PDF CV, 2 pages | Text is extracted **and every page is included** — the second page holds `ENSIAS`/`Django`, so a bug that returns only page 1 fails the test. Also the case-insensitive extension check (copied to `CLEAN_CV.PDF` at test time). |
| `clean_cv.docx` | A normal Word CV, 8 paragraphs incl. one empty | Every paragraph is joined, and a blank paragraph doesn't truncate or crash the join. |
| `no_text_layer.pdf` | A valid PDF whose page is drawn shapes — no text layer and no glyphs to recognise either | Extraction runs OCR, finds nothing, and returns `""`, so `has_extractable_text()` returns `False`. Since OCR arrived this no longer stands for "a scan" — it stands for a file with genuinely nothing in it. |
| `corrupted.pdf` | A damaged / truncated upload — a real PDF header, garbage body | Failure happens **inside the PDF parser** and is re-raised as `TextExtractionError`, a `RuntimeError` subclass — *not* a `ValueError`. Upload validation must catch this too. |
| `unsupported.txt` | A file type v1 does not accept | The function's third branch raises `ValueError: Unsupported file type: .txt`. |
| `disguised.pdf` | A DOCX renamed to `.pdf` — byte-identical to `clean_cv.docx` | Validation step 1 (extension) **passes**, step 2 (`python-magic`) catches it. This is the only fixture that proves the magic-byte check does anything. |

`unsupported.txt` is the one fixture beyond the four originally requested — it
covers the `else` branch of `extract_text_from_file`, which is otherwise untested.

## The OCR fixtures

These arrived with OCR. Each one is a file the old text-layer-only extractor
returned nothing (or almost nothing) for.

| Fixture | Represents | What it proves |
|---|---|---|
| `scanned_cv.pdf` | A 2-page CV that exists **only as pictures** — every page rasterized, zero text layer | OCR reads both pages, not just the first. The case the tool used to reject outright. |
| `hybrid_cv.pdf` | Page 1 typed (has a text layer), page 2 a scan | The OCR decision is made **per page**: page 1 keeps its exact text, page 2 is recovered by OCR. A per-*document* decision would silently drop one or the other. |
| `scanned_cv.png` | A photo / scanner dump of a printed CV, no PDF wrapper | Image files are a first-class input now — extension whitelist, magic-byte check and OCR all handle them. |
| `rotated_cv.png` | The same page, 90° sideways | Orientation is detected *before* recognition. A sideways page does not OCR to nothing — it OCRs to a confident page of nonsense, which is why there is nothing to notice afterwards. |
| `scanned_cv.docx` | A scan pasted into Word: one picture, two characters of real text | The pictures inside a `.docx` are OCR'd when the document's own text is thin. |
| `table_cv.docx` | A CV whose entire body is a **table**, contact details in the page **header**, skills in a **text box**, and one nested table | Needs no OCR at all — it is the plain-Word hole in the old extractor. `document.paragraphs` finds 7 characters of it; extraction finds 454. |

**Never put a real CV here**, scanned or otherwise — the same rule as above, and a
scan of a real person is if anything worse.

## Regenerating them

The four original generated files were produced with `pymupdf` and `python-docx`
directly. `corrupted.pdf` is simply a PDF header followed by repeated junk
bytes; `no_text_layer.pdf` is a one-page document containing only `draw_rect`
calls and no `insert_text` call.

The OCR fixtures come from the same two libraries, with one extra step: the
"scans" are made by writing a normal text PDF with `page.insert_text`, rendering
each page to a pixmap at 200 DPI, and building a second PDF whose pages hold only
that image (`page.insert_image`) — so the result is a real, glyph-free scan rather
than a PDF with hidden text. `scanned_cv.png` is that same pixmap saved as a
grayscale PNG, `rotated_cv.png` is it turned 90°, and `scanned_cv.docx` is a
one-paragraph document with the PNG placed by `document.add_picture`. Nothing here
depends on external sample files.


## `synthetic_cvs/` — plain-text CVs for the structuring tests

These are `.txt`, not PDFs: they stand in for the *output* of
`extract_text_from_file`, and feed `structure_candidate` directly. They are never
passed through the extraction function (`.txt` is rejected by design).

| File | Edge case | Why it matters |
|---|---|---|
| `career_gap.txt` | A 3-year gap between roles (Jan 2020 → Mar 2023) | Both roles and both date ranges must survive to the LLM, or the gap becomes invisible and Stage 1 can't reason about continuous experience. |
| `non_standard_degree.txt` | `Diplôme d'Ingénieur d'État`, CPGE, and a non-degree Coursera certificate | Credentials that don't map cleanly onto `degree`/`field`. Also the only fixture carrying **accented French characters** — the gap identified when the em dash in `clean_cv.pdf` was silently mangled to a middot. |
| `long_cv.txt` | 8 roles, 12 talks, ~3.8k characters | Proves nothing truncates the input silently. A dropped final role would be an invisible failure. |

With the LLM mocked, these test **our** handling — that the text arrives intact, that
the prompt prefix stays constant for caching, that the response is stored unmodified.
Whether the model extracts *well* from them is a separate question for the week-2
golden set; these files are ready-made inputs for that manual check.
