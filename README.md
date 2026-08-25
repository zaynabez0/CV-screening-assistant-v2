# CV Screening Assistant — v1

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
| Shared text extraction (PDF/DOCX) | ✅ built and tested |
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

- **An OpenAI API key** — needed for CV structuring, Stage 1 and Stage 3.
- **~2.5 GB of free disk** — Stage 2 downloads the `BAAI/bge-m3` embedding model on
  first use and caches it in `~/.cache/huggingface`.

## 2. Install

```bash
git clone <your-repo-url>
cd CV-screening-assistant-v1

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

This pulls in PyTorch via `sentence-transformers`, so expect a few minutes and a large
download.

## 3. Set your API key

> ⚠️ **`.env` is not loaded automatically.** Nothing in this project reads it — there is
> no `python-dotenv` dependency. Copying `.env.example` to `.env` on its own does
> nothing. You must get the variable into your shell environment.

```bash
cp .env.example .env
# edit .env and paste your key after OPENAI_API_KEY=

# then, in every shell where you run the app:
set -a && source .env && set +a
```

Or skip the file entirely and export it directly:

```bash
export OPENAI_API_KEY="sk-..."
```

Verify it took:

```bash
echo "${OPENAI_API_KEY:0:7}..."    # should print sk-... , not just "..."
```

The key is read straight from the environment by the OpenAI client. It is never
hardcoded, and `.env` is gitignored — keep it that way.

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

## 5. Start a screening run

Open <http://127.0.0.1:8000/> — this lists every run and is where you start new ones.

Press **Start a new run**, then either paste the job description text or upload it as a
PDF or DOCX. A scanned PDF with no text layer is rejected; there is no OCR in v1.

Creating the run does *not* call the LLM. You land on the job description review page.

## 6. Review the extracted requirements

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

## 7. Walk the rest of the pipeline

Replace `1` with your run's ID. Each screen has a **Confirm** button that advances the
run; nothing advances on its own.

| Step | URL | What you do |
|---|---|---|
| 0 | `/run/1/jd/review/` | (above) Extract, edit and confirm requirements → `cv_pending` |
| 1 | `/run/1/cvs/upload/` | Upload PDF/DOCX CVs. This only **validates** — no LLM call yet. |
| 2 | `/run/1/cvs/review/` | Press **Structure** to send them to the LLM, check each extraction, then **Confirm** → `stage1_pending` |
| 3 | `/run/1/stage1/review/` | **Run Stage 1**, read every pass *and* fail with reasons, then **Confirm** → `stage2_pending` |
| 4 | `/run/1/stage2/review/` | Edit keywords and the cutoff %, **Run Stage 2**, then **Confirm** → `stage3_pending` |
| 5 | `/run/1/stage3/review/` | Adjust criterion weights, **Run Stage 3**, then **Confirm** → `review` |
| 6 | `/run/1/review/` | Full history per candidate, log overrides, download the Excel summary or the CSV |

Useful things to know as you go:

- **Stage 2 is slow the first time.** It downloads the ~2 GB embedding model. Later runs
  load it from cache.
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

## 8. Run the tests

```bash
python -m pytest              # 317 passed, 5 skipped
python -m pytest -v           # see individual test names
```

Five tests are skipped when `BAAI/bge-m3` can't be downloaded (no network, or a
restricted environment). They exercise the real embedding model and run automatically
once it is reachable:

```bash
python -m pytest screening/tests/test_stage2_embeddings_live.py -v
```

No test makes a network call to OpenAI — the LLM is mocked throughout.

## 9. Project layout

```
config/          Django settings, URLs
screening/       ALL pipeline logic — pure Python, ZERO Django imports
  common/          shared text extraction (PDF/DOCX) + the one LLM helper
  jd/              JD_SCHEMA, get_jd_text, extract_requirements
  cv/              validation, CV_SCHEMA structuring, anonymization
  stage1_hard_filter/
  stage2_ranking/
  stage3_scoring/
  tests/           pytest suite + fixtures
candidates/      Candidate model, upload & review views
jobs/            JobDescription model, JD input + review/edit views
runs/            ScreeningRun, StageResult, Override, run list, stage views,
                 exports.py (Excel summary) + CSV export
```

The rule that matters: **`screening/` never imports Django.** Django views call into it,
never the reverse. That is why the pipeline logic can be tested without a database or a
web server, and it is enforced by a check in the test suite.

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: failed to find libmagic.  Check your installation` | The system library is missing — see [Prerequisites](#1-prerequisites). |
| `OpenAIError: Missing credentials...set the OPENAI_API_KEY environment variable` | The key isn't in your environment. Copying `.env` is not enough — `set -a && source .env && set +a`. |
| Upload page says the run is at the wrong stage | Confirm the job description first — that is what moves the run to `cv_pending`. |
| "The job description's requirements have not been confirmed yet" | Press **Confirm requirements and continue** on the JD review page. |
| Stage 2 hangs on first run | It's downloading the 2 GB model. Watch `~/.cache/huggingface`. |
| A stage says "failed" with a network error | Expected if the API is unreachable. Fix the connection and press the button again — re-running replaces the previous results rather than duplicating them. |
| `database is locked` | SQLite limitation: a background stage and the web UI wrote at the same time. Retry. Known and accepted for v1. |
| A confirmed job description can't be edited | By design — the stages already judged against it would no longer match. Start a new run. |

## 11. Before this goes anywhere near real candidate data

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
