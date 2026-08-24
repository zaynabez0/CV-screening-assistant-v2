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
| `no_text_layer.pdf` | A **scanned** CV — a valid PDF whose page is drawn shapes, zero text | Extraction succeeds but returns `""`, so `has_extractable_text()` returns `False`. This is v1's entire handling of scanned documents: rejected as invalid input, **no OCR fallback** — a deliberate scope cut. |
| `corrupted.pdf` | A damaged / truncated upload — a real PDF header, garbage body | Failure happens **inside the PDF parser**, raising `fitz.FileDataError` (a `RuntimeError` subclass), *not* a `ValueError`. Upload validation must catch this too. |
| `unsupported.txt` | A file type v1 does not accept | The function's third branch raises `ValueError: Unsupported file type: .txt`. |
| `disguised.pdf` | A DOCX renamed to `.pdf` — byte-identical to `clean_cv.docx` | Validation step 1 (extension) **passes**, step 2 (`python-magic`) catches it. This is the only fixture that proves the magic-byte check does anything. |

`unsupported.txt` is the one fixture beyond the four originally requested — it
covers the `else` branch of `extract_text_from_file`, which is otherwise untested.

## Regenerating them

The four generated files were produced with `pymupdf` and `python-docx`
directly. `corrupted.pdf` is simply a PDF header followed by repeated junk
bytes; `no_text_layer.pdf` is a one-page document containing only `draw_rect`
calls and no `insert_text` call. Nothing here depends on external sample files.


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
