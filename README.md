# CV Screening Assistant — v2

An internal HR tool: provide a job description, upload CVs, and run them through a
three-stage selection pipeline to get a ranked shortlist with per-criterion evidence.

**A recruiter reviews and can override every stage. Nothing is ever auto-rejected.**
Every stage is triggered explicitly by a human, and every candidate filtered out stays
visible with the reason for the decision — this is a GDPR Article 22 requirement, not a
UI preference.

---

## Current status

The pipeline runs end to end in the browser, from pasting a job description through to
exporting the final ranked list.

| Part of the pipeline | Status |
|---|---|
| JD processing — text/file input, requirement extraction, review & edit | ✅ built and tested |
| Shared text extraction — PDF, DOCX, scans and photos, OCR-backed | ✅ built and tested |
| CV preprocessing — upload, validation, structuring, review | ✅ built and tested |
| Stage 1 — hard filter | ✅ built and tested |
| Stage 2 — ranking | ✅ built and tested |
| Stage 3 — scoring | ✅ built and tested |
| Final Review — history, overrides, Excel + CSV export | ✅ built and tested |
| Closing a run (the `done` state) | ❌ not built — `review` is the last stage reachable |

The Excel summary is the deliverable at the end of a run: one row per candidate,
which stages they passed or failed, why they stopped where they did, and the final
score for anyone scored at Stage 3.

---

## 1. Prerequisites

- **Python 3.11+** (developed on 3.11)
- **`libmagic`** — a system library, not a Python package. `python-magic` is only a
  wrapper around it and will fail to import without it.

  ```bash
  # Debian / Ubuntu
  sudo apt-get install libmagic1

  # macOS
  brew install libmagic

  # Windows: pip install python-magic-bin  (instead of python-magic)
  ```

- **Tesseract OCR** — a system program, not a Python package. `pytesseract` is only a
  wrapper around it and returns nothing useful without it. This is what reads scanned
  CVs, photos of printed CVs, and pictures pasted into Word.

  ```bash
  # Debian / Ubuntu — the second package is the French language pack
  sudo apt-get install tesseract-ocr tesseract-ocr-fra

  # macOS (Homebrew installs all languages with it)
  brew install tesseract

  # Windows: install the UB Mannheim build, ticking "French" under
  # "Additional language data" in the installer:
  #   https://github.com/UB-Mannheim/tesseract/wiki
  # Then either tick "Add to PATH" during setup, or put this in your .env:
  #   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
  ```

  You do **not** need Poppler or `pdf2image`: PDF pages are rasterized by `pymupdf`,
  which pip installs for you. That is one fewer manual install on every machine, and
  Poppler on Windows is a download-and-edit-your-PATH job.

  Skipping Tesseract is survivable but lossy: CVs that carry a text layer keep working
  exactly as before, and everything else is rejected with *"no extractable text found
  (this file has no text layer and OCR is not installed on the server)"*.

- **An OpenAI API key** — needed for CV structuring and all three screening stages.

## 2. Install

```bash
git clone <your-repo-url>
cd CV-screening-assistant-v1

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Then check the OCR half of that is actually wired up:

```bash
python manage.py check_ocr
```

It prints the Tesseract version, which language packs it found and which settings are in
force — or tells you exactly what is missing. Point it at a file to do a live read:

```bash
python manage.py check_ocr screening/tests/fixtures/scanned_cv.pdf
```

That fixture is a CV that exists only as pictures. If its text comes back, OCR works.

To see what a *real* CV extracts to — which is exactly the text the LLM is sent, so it is
the first thing to look at when a CV structures badly:

```bash
python manage.py check_ocr samples/some_cv.pdf --full      # print all of it
python manage.py check_ocr samples/some_cv.pdf --out extracted.txt   # or read it in an editor
```

Put real CVs in `samples/`, which is gitignored. **Not** in `screening/tests/fixtures/` —
that folder is committed, and a real CV in Git history is personal data that cannot be
taken back out.

### Tuning OCR

Every one of these is optional; the defaults are what runs if you set none of them.
They go in `.env`, like the settings in [§3](#3-configure-the-llm). They are
deliberately **not** in `.env.example`: that file lists only what you have to fill
in, and everything below already works without being set.

| Variable | Default | What it does |
|---|---|---|
| `OCR_ENABLED` | on | `0` turns OCR off — extraction falls back to text layers only. |
| `OCR_LANGUAGES` | `eng+fra` | Language packs, `+`-joined. Each must be installed; any that is not is dropped with a warning rather than failing the file. |
| `OCR_DPI` | `300` | Render resolution for PDF pages. 300 is what Tesseract is trained for; lower is faster and noticeably worse. |
| `OCR_MIN_CHARS_PER_PAGE` | `100` | A page with less real text than this is treated as a scan and sent to OCR. |
| `OCR_MAX_PAGES` | `30` | Cap on pages OCR'd per file. |
| `OCR_CONFIG` | *(empty)* | Extra Tesseract flags, e.g. `--psm 4` for a CV laid out in columns. |
| `OCR_DETECT_ORIENTATION` | on | Check each OCR'd page for a sideways scan first. Costs one extra pass per page. |
| `TESSERACT_CMD` | *(empty)* | Full path to the binary, when it is installed but not on `PATH`. |

**After every `git pull`, run `python manage.py migrate`.** New code often carries a
schema change, and an out-of-date database fails at the first page that touches the new
column rather than at startup. Django warns about it when the server boots — read the
lines above the traceback.

## 3. Configure the LLM

Copy the template and fill it in — **`.env` is loaded automatically**, so this is all
you need. Nothing has to be exported in your shell, on any platform: both
`config/settings.py` and `screening/common/llm.py` load it, so the web server, the stage
subprocesses, the tests and a plain `python your_script.py` all see the same values.

```bash
cp .env.example .env
```

`.env.example` is the complete list of what this project reads. One name per setting,
no aliases, four lines:

| Variable | Required | What it is |
|---|---|---|
| `KEY_REGULATOR_API_KEY` | **yes** | The API key your provider issued. |
| `MODEL` | **yes** | The id of the model to call, spelled exactly as your provider or gateway exposes it. |
| `BASE_URL` | **yes** | The endpoint to call, e.g. your gateway's. No default — the project is not tied to one provider. |
| `TESSERACT_CMD` | no | Full path to the Tesseract binary, when it is installed but not on `PATH`. See [§1](#1-prerequisites). |

**Neither `MODEL` nor `BASE_URL` has a default, on purpose.** Which model to run and
where to reach it are deployment decisions, and nothing here assumes an OpenAI model or
an OpenAI address. The `openai` package is used only as the client library, so any
provider speaking the OpenAI Responses API will do.

Blank counts as unset for all three, which is why the template ships with every line
present and empty: a line that exists but is empty fails at startup, naming the variable,
rather than at the first API call — once per candidate, halfway through a run. Values are
stripped, so a trailing space picked up while copy-pasting does no harm.

A real environment variable always wins over `.env`, so you can override any of these for
a single command:

```bash
MODEL=some-other-model python manage.py runserver
```

### Check what the app will actually use

Run this **from the project root** — no Django bootstrap needed:

```bash
python -c "from screening.common.llm import get_client, get_model; c = get_client(); print('key     :', c.api_key[:8] + '...'); print('endpoint:', c.base_url); print('model   :', get_model())"
```

If all three print, `.env` is being read. If `endpoint` isn't what you expect, your
`BASE_URL` isn't reaching the app. If you get
`ModuleNotFoundError: No module named 'screening'`, you are not in the project root —
`.env` is found relative to the code, but `screening` is imported relative to where you
started Python.

Every LLM-calling part of the pipeline uses this key: JD extraction, CV structuring,
and Stages 1, 2 and 3. `.env` is gitignored; keep it that way.

## 4. Create the database and an admin user

```bash
python manage.py migrate
python manage.py createsuperuser
```

The database is a single SQLite file (`db.sqlite3`). No server process, no Docker.
It is gitignored, because it will contain real candidate data.

Start the server:

```bash
python manage.py runserver
```

Then open <http://127.0.0.1:8000/admin/> and sign in.

## 5. How Stage 2 ranks

Stage 2 makes **one LLM call per surviving candidate**, and that call does two jobs at
once:

- a **job-fit score** (0-100) — how well the candidate's experience meets the
  requirements taken as a whole. It judges the work, not the wording: a CV that says
  "server-side programming for web services" can meet a "Python backend development"
  requirement with no words in common.
- a **per-keyword check** — for each keyword in your list, present or absent, with the
  evidence quoted. It reads the whole candidate record, not just the extracted skills
  list, so a skill demonstrated in a job description counts.

Both are rescaled to 0-1 and blended 50/50 into the score the ranking uses. Only the
*judgements* come from the model; every number is computed in Python from them, so a
stored ranking can always be checked against what the model actually said.

The **keyword list is your lever**. Short, concrete terms ("Kubernetes", "PostgreSQL")
work better than whole requirement sentences. Editing the list marks an existing ranking
stale and needs a re-run; moving the *cutoff* does not.

### Check it works

Two real API calls, worth running once to see what the model does before you trust it on
real CVs:

```bash
python -c "
from screening.stage2_ranking.ranking import score_candidate_match
jd = [{'requirement': 'Python backend development',
       'description': 'Building server-side services in Python'}]
keywords = ['Python', 'Kubernetes']
candidates = {
    'backend dev': {'skills': ['Python', 'Django'], 'total_years_experience': 5,
                    'experience': [{'title': 'Backend Engineer', 'company': 'Kite',
                                    'description': 'Built server-side APIs in Python.'}]},
    'pastry chef': {'skills': ['Pastry'], 'total_years_experience': 5,
                    'experience': [{'title': 'Pastry Chef', 'company': 'Cafe',
                                    'description': 'Prepared desserts.'}]},
}
for label, cv in candidates.items():
    s = score_candidate_match(jd, cv, keywords)
    print(f\"{s['final_score']:.3f}  fit {s['semantic_score']:.2f}  keyword {s['keyword_score']:.2f}  {label}\")
"
```

Scores run from 0 to 1. Expect the backend developer clearly above the pastry chef. If
they score alike, something is wrong — don't trust the Stage 2 ranking until it's fixed.

### What it costs, and what it does not promise

- **One API call per surviving candidate.** Stage 2 is no longer the cheap stage: a
  100-CV batch that clears Stage 1 makes 100 Stage 2 calls, and a large field takes a
  few minutes. Long keyword lists add output tokens on every candidate, so keep the list
  to terms you actually care about.
- **It is not deterministic.** Re-running Stage 2 on the same candidates can reorder
  ones that score close together — the same tradeoff Stage 1 already makes. Exact ties
  still break on upload order, the ranking is written once and re-sliced by the cutoff
  without re-running, and the order you confirm is the order that ships. The stored
  `fit_reasoning` and per-keyword evidence are what make a given ranking explainable
  months later.
- **Candidate data leaves your machine at this stage too.** Stage 2 used to run
  entirely locally. It no longer does: the anonymized record goes to the API exactly as
  it does at Stages 1 and 3.

## 6. Start a screening run

Open <http://127.0.0.1:8000/> — this lists every run and is where you start new ones.

Press **Start a new run**, then either paste the job description text or upload it as a
PDF, DOCX or an image. A file with no text layer is read with OCR, so a scan is fine —
it just takes a second or two per page.

Creating the run does *not* call the LLM. You land on the job description review page.

## 7. Review the extracted requirements

Press **Extract requirements**. The LLM returns one flat list, each item tagged
`must_have`, `nice_to_have` or `inferred`. Then check it — this is the highest-leverage
screen in the whole tool, because that single tag decides how each requirement is used:

- **`must_have`** → gated pass/fail in Stage 1. A candidate missing any one of these is
  filtered out, so an over-tagged must-have costs you good candidates.
- **`nice_to_have`** and **`inferred`** → become Stage 3's scoring criteria, each with a
  1–5 scale and a weight.

On this page you can change any requirement's **type**, edit its text or description,
delete it, or add one the model missed. Switching a row to `must_have` clears its scale
and weight automatically, since Stage 3 never reads them for that type.

Press **Confirm requirements and continue** when the list is right. Confirming:

- locks the list — **it cannot be edited afterwards**, because the stages that run
  against it would no longer match what they were judged on;
- moves the run to `cv_pending` and sends you to CV upload.

You cannot confirm an empty list. If there are no `must_have` items you'll get a warning
rather than a block — that's legal, it just means Stage 1 has no hard gate and everyone
passes it.

## 8. Walk the rest of the pipeline

Replace `1` with your run's ID. Each screen has a **Confirm** button that advances the
run; nothing advances on its own.

| Step | URL | What you do |
|---|---|---|
| 0 | `/run/1/jd/review/` | (above) Extract, edit and confirm requirements → `cv_pending` |
| 1 | `/run/1/cvs/upload/` | Upload CVs — PDF, DOCX, or images of printed pages. This only **validates** — no LLM call yet, though scans are OCR'd here, which is why a batch of them is not instant. |
| 2 | `/run/1/cvs/review/` | Press **Structure** — it runs in the background with a progress bar — check each extraction, then **Confirm** → `stage1_pending` |
| 3 | `/run/1/stage1/review/` | **Run Stage 1**, read every pass *and* fail with reasons, then **Confirm** → `stage2_pending` |
| 4 | `/run/1/stage2/review/` | Edit keywords and the cutoff %, **Run Stage 2**, then **Confirm** → `stage3_pending` |
| 5 | `/run/1/stage3/review/` | Adjust criterion weights, **Run Stage 3**, then **Confirm** → `review` |
| 6 | `/run/1/review/` | Full history per candidate, log overrides, download the Excel summary or the CSV |

Useful things to know as you go:

- **Stage 2 costs one API call per surviving candidate** and takes a few minutes for a
  large field — see [How Stage 2 ranks](#5-how-stage-2-ranks). It is no longer the cheap
  stage, and re-running it can reorder candidates that score close together.
- **Stages run in a background process** and the page polls for progress. If a stage
  fails, the error appears on the page rather than vanishing.
- **Changing the Stage 2 cutoff does not require a re-run** — it re-slices the existing
  ranking instantly. Changing *keywords* does require a re-run, and the page tells you.
- **A job description with no `nice_to_have` or `inferred` requirements** has nothing for
  Stage 3 to score. The Stage 3 page will offer to skip it, and the Stage 2 ranking
  becomes the final order.
- **Logging an override requires being signed in** (overrides record who made them), and
  every override needs a written reason.

### The two exports

Both cover **every** candidate, including those rejected at upload or filtered at
Stage 1 — nobody silently disappears from the record.

| | **Excel summary** (`export.xlsx`) | **CSV** (`export.csv`) |
|---|---|---|
| For | reading, and sending to a hiring manager | audit, and loading into other tools |
| Sheet 1 | one row per candidate: outcome, stage-by-stage verdicts, why they stopped, Stage 3 score, low-confidence criteria, overrides | one row per candidate with **every** per-requirement reason in full |
| Sheet 2 | the run's settings — cutoff, keywords, and the job requirements with their types — so the scores can still be interpreted months later | — |
| Formatting | frozen header, filters, pass/fail colouring, scores as real numbers you can sort | plain text |

Both files carry candidate names and emails. Treat the downloads as personal data.

## 9. Progress

Every step that makes one call per CV — structuring, Stage 1, Stage 2, Stage 3 — runs in a
background process and shows a progress bar that advances one candidate at a time. JD
extraction is a single call and has no bar; there would be nothing to count.

## 10. Run the tests

```bash
python -m pytest              # 404 passed
python -m pytest -v           # see individual test names
```

Nothing reaches the network — the LLM is mocked throughout, Stage 2 included, and OCR
runs against a local binary.

The only tests that skip are the ones that need a real Tesseract, and they skip
themselves with a reason when it is not installed (`python -m pytest -rs` lists them).
That is the same promise the code makes to the app: no Tesseract means degraded, never
broken. The OCR *policy* — what happens when the engine is missing, when a language pack
is not installed, when the orientation guess is worthless — is tested against a fake
engine in `screening/tests/test_ocr.py` and always runs.

## 11. Project layout

```
config/          Django settings, URLs
screening/       ALL pipeline logic — pure Python, ZERO Django imports
  common/          text extraction (PDF/DOCX/images, OCR-backed), the Tesseract
                   plumbing, the one LLM helper
  jd/              JD_SCHEMA, get_jd_text, extract_requirements
  cv/              validation, CV_SCHEMA structuring, anonymization
  stage1_hard_filter/
  stage2_ranking/
  stage3_scoring/
  tests/           pytest suite + fixtures
candidates/      Candidate model, upload & review views
jobs/            JobDescription model, JD input + review/edit views
runs/            ScreeningRun, StageResult, Override, run list, stage views,
                 services.py (the four background jobs), exports.py (Excel
                 summary) + CSV export
```

The rule that matters: **`screening/` never imports Django.** Django views call into it,
never the reverse. That is why the pipeline logic can be tested without a database or a
web server, and it is enforced by a check in the test suite.

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `OperationalError: no such column: ...` right after pulling | New code, old database. Run `python manage.py migrate`. Django warns about this on startup too, a few lines above the traceback. |
| `ImportError: failed to find libmagic.  Check your installation` | The system library is missing — see [Prerequisites](#1-prerequisites). |
| Every scanned CV is rejected with "no extractable text found" | Run `python manage.py check_ocr`. It will say whether Tesseract is missing, off (`OCR_ENABLED=0`), or installed but not on `PATH` (set `TESSERACT_CMD`). |
| `TesseractNotFoundError` | Tesseract is installed but not on `PATH` — the usual Windows case. Set `TESSERACT_CMD` in `.env` to the full path of `tesseract.exe`. |
| A scan is read, but the accented words are mangled | The French pack is missing: `sudo apt-get install tesseract-ocr-fra`, or re-run the Windows installer and tick French. `check_ocr` lists the packs it found. |
| Uploading a batch of scans is slow | Expected: OCR runs at upload, a second or two per page, because validation has to know whether the file holds readable text before it says the CV was accepted. It is not repeated at structuring — the text is stored on the candidate. |
| A scan comes back as gibberish | It is probably sideways and the orientation check could not tell. Try a higher `OCR_DPI`, or `OCR_CONFIG=--psm 4` for a columned layout. |
| `KEY_REGULATOR_API_KEY is not set, so ...` | Check `.env` exists in the project root and that the key line is filled in. A line that is present but empty counts as unset. |
| `MODEL is not set, so there is no model to call` | Same file, same rule — there is no default model, so this line must name one. |
| `BASE_URL is not set, so there is no endpoint ...` | Same again — there is no default endpoint either. Fill in the address your provider gave you. |
| `404` or "model not found" from a gateway | Usually the **model id**, not the key. Gateways often rename models — set `MODEL` to whatever yours exposes. |
| Connection errors on any step | These are all LLM calls now. Check `BASE_URL` points at your gateway; run the check snippet in [§3](#3-configure-the-llm). |
| Upload page says the run is at the wrong stage | Confirm the job description first — that is what moves the run to `cv_pending`. |
| "The job description's requirements have not been confirmed yet" | Press **Confirm requirements and continue** on the JD review page. |
| Stage 2 is slow | It makes one API call per surviving candidate. Watch the progress bar — it advances once per call. |
| The progress bar sits at 0% | The background process did not start. Check that `manage.py run_cv_structuring <run_id>` (or `run_stage1`, etc.) works from the project root. |
| A stage says "failed" with a network error | Expected if the API is unreachable. Fix the connection and press the button again — re-running replaces the previous results rather than duplicating them. |
| `database is locked` | SQLite limitation: a background stage and the web UI wrote at the same time. Retry. Known and accepted for v1. |
| A confirmed job description can't be edited | By design — the stages already judged against it would no longer match. Start a new run. |

## 13. Before this goes anywhere near real candidate data

This is a development build. Three things must be fixed first:

1. **There is no authentication.** Only the override action requires a login. Every other
   page — CV upload, extracted personal data, the CSV export — is open to anyone who can
   reach the server. Add `LoginRequiredMiddleware` or `@login_required` to every view.
2. **`DEBUG = True` and the `SECRET_KEY` is committed** in `config/settings.py`. Set
   `DEBUG = False`, move the secret key to an environment variable, and fill in
   `ALLOWED_HOSTS`.
3. **The CSV export contains candidate names and emails.** Anonymization protects the LLM
   stages, not the recruiter's own review — a shortlist without names would be unusable.
   That file is personal data the moment it is downloaded.

Also worth knowing: anonymization strips `full_name`, `email` and `phone` from the top
level of the candidate JSON, but a name quoted inside an evidence line still reaches the
LLM. Consider that before relying on the pipeline being fully blind.

And note the scope of what leaves the machine: candidate data now reaches the LLM
provider at **four** points in the pipeline, not three. Stage 2 used to rank locally with
a self-hosted embedding model and sent nothing anywhere; it now makes an API call per
surviving candidate like every other stage. That widens the vendor's scope in a DPIA — it
is not merely an implementation detail.
