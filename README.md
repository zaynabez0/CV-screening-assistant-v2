# CV Screening Assistant

An internal HR tool. You give it a job description and a pile of CVs; it walks them
through a three-stage pipeline and gives you a ranked shortlist where every decision
comes with the evidence behind it.

**A recruiter drives every stage, and nothing is ever auto-rejected.** No stage runs
until a human presses the button, and a candidate filtered out at any point stays on
screen with the reason why.

The pipeline runs end to end in the browser, from pasting a job description to
downloading the final Excel summary.

---

## How the screening works

Six steps. You confirm each one before the next becomes available, and you can go back
and look at any of them afterwards.

```
  Job description ──► requirements list        (you review and edit)
         │
  CV files ─────────► structured records       (you review)
         │
         ├─► Stage 1   hard filter    must-haves, pass/fail
         │
         ├─► Stage 2   ranking        job fit + your keywords, top N% continue
         │
         └─► Stage 3   scoring        weighted rubric, 1-5 per criterion
                             │
                    Final review ──► Excel + CSV
```

**1. The job description becomes a list of requirements.** You paste the text or upload
a file. The LLM returns one flat list, each item tagged `must_have`, `nice_to_have` or
`inferred`. You edit the list and the tags, then confirm, after which it is locked,
because the stages that judge against it would otherwise stop matching what they judged.

That tag is the highest-leverage thing in the tool. `must_have` becomes a pass/fail gate
in Stage 1; the other two become Stage 3's scoring criteria. Over-tagging a must-have
costs you good candidates.

**2. The CVs become structured records.** Upload PDFs, DOCX files or images. Each one is
checked first, file type by its actual bytes, size, duplicates, and whether any text can
be read out of it at all. Then the LLM turns that text into a record with education,
experience, skills, languages and years of experience. You review the extractions before
moving on.

**3. Stage 1** For each candidate, the LLM checks every `must_have`
requirement and answers met or not met, quoting the evidence either way. Python does the
ANDing: miss one requirement and you are filtered out. Nothing is decided by the model on
its own, and a filtered candidate keeps their full evaluation list on screen.

**4. Stage 2** One LLM call per surviving candidate, doing two jobs at once:

- a **job-fit score** (0-100) for how well the experience meets the requirements overall.
  It judges the work, not the wording, a CV saying "server-side programming for web
  services" can satisfy a "Python backend development" requirement with no words in common.
- a **per-keyword check** against a list you control, present or absent with a quote. It
  reads the whole record, not just the skills list, so a tool used in a job description
  counts.

Python rescales both to 0-1 and blends them 50/50. The top N% by your cutoff go on to
Stage 3. The keyword list is your main lever here: short concrete terms ("Kubernetes",
"PostgreSQL") work better than whole sentences. Changing keywords needs a re-run; moving
the cutoff just re-slices the ranking you already have.

**5. Stage 3** The `nice_to_have` and `inferred` requirements become a rubric,
each with a 1-5 scale and a weight you can adjust. The LLM scores every criterion with a
quote and a confidence level; Python computes the weighted total. A job description with
no soft requirements has nothing to score, and the page offers to skip the stage.

**6. Final review.** Every candidate's history in one place, whatever stage they stopped
at. Log an override with a written reason, and download the summary.

Two things hold throughout. **The model gives judgements; Python does the arithmetic**,
every score is recomputable from what the model actually said, which is what makes a
result auditable months later. And **candidates are anonymized before every LLM call**:
name, email and phone are stripped from the record that gets sent.

---

## 1. What you need

- **Python 3.11+** (developed on 3.11)

- **`libmagic`** — a system library, not a Python package. `python-magic` is a wrapper
  around it and will not import without it.

  ```bash
  sudo apt-get install libmagic1      # Debian / Ubuntu
  brew install libmagic               # macOS
  # Windows: pip install python-magic-bin  instead of python-magic
  ```

- **Tesseract OCR** — also a system program, not a Python package. This is what reads
  scanned CVs, photos of printed ones, and pictures pasted into Word.

  ```bash
  sudo apt-get install tesseract-ocr tesseract-ocr-fra   # the 2nd is the French pack
  brew install tesseract                                 # macOS, all languages
  ```

  On Windows, install the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)
  and tick "French" under *Additional language data*. Either tick "Add to PATH" during
  setup, or set `TESSERACT_CMD` in your `.env` to the full path of `tesseract.exe`.

  You do **not** need Poppler or `pdf2image` — PDF pages are rasterized by `pymupdf`,
  which pip installs for you.

  Skipping Tesseract is survivable but lossy: CVs with a text layer keep working, and
  everything else is rejected with *"no extractable text found"*.

- **An API key for an LLM**, and the endpoint it lives behind.

## 2. Install and run

```bash

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Configure the LLM (next section), then create the database and an account:

```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open <http://127.0.0.1:8000/>. That page lists every run and is where you start new ones.

The database is a single SQLite file (`db.sqlite3`) — no server process, no Docker. It is
gitignored, because it will hold real candidate data.

> **After every `git pull`, run `python manage.py migrate`.** New code often carries a
> schema change, and an out-of-date database fails at the first page that touches the new
> column rather than at startup.

### Checking OCR works

```bash
python manage.py check_ocr
```

This prints the Tesseract version, the language packs it found and the settings in force
— or tells you exactly what is missing. Point it at a file for a live read:

```bash
python manage.py check_ocr samples/some_cv.pdf
python manage.py check_ocr samples/some_cv.pdf --full             # print all the text
python manage.py check_ocr samples/some_cv.pdf --out extracted.txt
```

What it prints is exactly the text the LLM gets, so it is the first thing to look at when
a CV structures badly.

Keep real CVs in `samples/`, which is gitignored. Never in a folder that gets committed:
a real CV in Git history is personal data you cannot take back out.

OCR has four tuning values, and they are constants at the top of
`screening/common/ocr.py` rather than settings — `OCR_LANGUAGES` (`eng+fra`), `OCR_DPI`
(300), `OCR_MIN_CHARS_PER_PAGE` (100, below which a page is treated as a scan) and
`OCR_MAX_PAGES` (30).

## 3. Configure the LLM

```bash
cp .env.example .env
```

`.env.example` is the complete list of what this project reads — one name per setting, no
alternatives, four lines:

| Variable | Required | What it is |
|---|---|---|
| `KEY_REGULATOR_API_KEY` | **yes** | The API key your provider issued. |
| `MODEL` | **yes** | The model id, spelled exactly as your provider exposes it. |
| `BASE_URL` | **yes** | The endpoint to call. |
| `TESSERACT_CMD` | no | Full path to the Tesseract binary, when it is not on `PATH`. |

`.env` is loaded automatically, on every platform — the web server, the background stage
processes and a plain `python your_script.py` all read the same file, and nothing needs
exporting in your shell. A real environment variable still wins over the file, so you can
override any of these for one command:

```bash
MODEL=some-other-model python manage.py runserver
```

**None of the three LLM settings has a default.** Which model to run and where to reach it
are deployment decisions, and nothing here assumes a particular provider — the `openai`
package is used only as the client library, so any endpoint speaking the OpenAI Responses
API will do. A blank line counts as unset, so a copied-but-unfilled template fails at
startup naming the variable, rather than at the first API call halfway through a run.

To see what the app will actually use:

```bash
python -c "from screening.common.llm import get_client, get_model; c = get_client(); print('key     :', c.api_key[:8] + '...'); print('endpoint:', c.base_url); print('model   :', get_model())"
```

Run it from the project root. `.env` is found relative to the code, but `screening` is
imported relative to where you started Python.

`.env` is gitignored. Keep it that way.

## 4. Walking a run

Every screen has a **Confirm** button that advances the run. Nothing advances on its own.
Replace `1` with your run's ID.

| | Page | What you do |
|---|---|---|
| 1 | `/run/1/jd/review/` | **Extract requirements**, fix the types, **Confirm** → CV upload |
| 2 | `/run/1/cvs/upload/` | Upload CVs. Validation only, no LLM call — though scans are OCR'd here, which is why a batch of them is not instant |
| 3 | `/run/1/cvs/review/` | **Structure**, check the extractions, **Confirm** → Stage 1 |
| 4 | `/run/1/stage1/review/` | **Run Stage 1**, read the passes *and* the fails, **Confirm** → Stage 2 |
| 5 | `/run/1/stage2/review/` | Set keywords and cutoff %, **Run Stage 2**, **Confirm** → Stage 3 |
| 6 | `/run/1/stage3/review/` | Adjust weights, **Run Stage 3**, **Confirm** → final review |
| 7 | `/run/1/review/` | Full history per candidate, log overrides, download the exports |

Worth knowing as you go:

- **Structuring and all three stages run in a background process**, with a progress bar
  that advances one candidate at a time. If a stage fails, the error appears on the page
  rather than vanishing. Press the button again to retry — a re-run replaces the previous
  results rather than duplicating them.
- **Stage 2 takes a few minutes on a large field.** One API call per surviving candidate.
- **Stage 2 is not deterministic.** Re-running it can reorder candidates that score close
  together. Exact ties break on upload order, and the order you confirm is what ships.
- **A confirmed job description cannot be edited.** The stages already judged against it
  would no longer match. Start a new run instead.
- **Logging an override requires being signed in** — overrides record who made them — and
  every override needs a written reason.

## 5. The exports

Both cover **every** candidate, including those rejected at upload or filtered at Stage 1.
Nobody silently disappears from the record.

| | **Excel** (`export.xlsx`) | **CSV** (`export.csv`) |
|---|---|---|
| For | reading, and sending to a hiring manager | audit, and loading into other tools |
| Candidates sheet | one row each: outcome, stage-by-stage verdicts, why they stopped, Stage 3 score, low-confidence criteria, overrides | one row each, with every per-requirement reason in full |
| Run summary sheet | the cutoff, the keywords, and the requirements with their types — so the scores can still be interpreted months later | — |
| Formatting | frozen header, filters, pass/fail colouring, scores as sortable numbers | plain text |

Both carry candidate names and emails. Treat the downloads as personal data.

## 6. Project layout

```
config/          Django settings and root URLs
screening/       ALL pipeline logic — pure Python, ZERO Django imports
  common/          llm.py (the single LLM call helper), text_extraction.py
                   (PDF/DOCX/images), ocr.py (the Tesseract plumbing)
  jd/              JD_SCHEMA, get_jd_text, extract_requirements
  cv/              validation, CV_SCHEMA structuring, anonymization
  stage1_hard_filter/   must-have evaluation
  stage2_ranking/       keyword seeding, the matching call, the 50/50 blend
  stage3_scoring/       rubric building, scoring, weighted aggregation
jobs/            JobDescription model, JD input and review views
candidates/      Candidate model, upload and review views
runs/            ScreeningRun, StageResult, Override; the stage views;
                 services.py (what the background jobs actually do);
                 exports.py (Excel + CSV); management/commands/ (check_ocr and
                 the four stage runners the views launch)
```

**The rule that matters: `screening/` never imports Django.** Django views call into it,
never the reverse. That is what keeps the pipeline logic usable from a plain script, and
what lets you reason about a stage without a database in front of you.

A stage is launched by its view starting a management command as a subprocess; that
command calls the matching function in `runs/services.py`, which is where the Django
models meet the `screening/` functions.

Candidate data reaches the LLM provider at four points: CV structuring, and each of the three stages. Job description text reaches it once more, at extraction.
