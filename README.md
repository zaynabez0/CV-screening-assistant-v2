# CV Screening Assistant — v1

An internal HR tool: provide a job description, upload CVs, and run them through a
three-stage selection pipeline to get a ranked shortlist with per-criterion evidence.

**A recruiter reviews and can override every stage. Nothing is ever auto-rejected.**
Every stage is triggered explicitly by a human, and every candidate filtered out stays
visible with the reason for the decision — this is a GDPR Article 22 requirement, not a
UI preference.

---

## Current status — read this before you start

The pipeline is built and tested from CV upload through to Final Review. **JD Processing
is not built yet.** That has one practical consequence you will hit immediately:

> A new screening run starts at `jd_pending`, and **nothing in the app can move it to
> `cv_pending`**. You must create the job description and advance the run by hand in the
> Django admin. [Step 5](#5-create-your-first-screening-run) explains how.

| Part of the pipeline | Status |
|---|---|
| Shared text extraction (PDF/DOCX) | ✅ built and tested |
| CV preprocessing — upload, validation, structuring, review | ✅ built and tested |
| Stage 1 — hard filter | ✅ built and tested |
| Stage 2 — ranking | ✅ built and tested |
| Stage 3 — scoring | ✅ built and tested |
| Final Review — history, overrides, CSV export | ✅ built and tested |
| **JD Processing — text/file input, requirement extraction, edit UI** | ❌ **not built** |
| Creating a run from the web UI | ❌ not built (use the admin) |
| Closing a run (`done` state) | ❌ not built |

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

## 5. Create your first screening run

Because JD Processing isn't built, this part happens in the admin.

**a. Create a Job Description** — `Jobs → Job descriptions → Add`
(<http://127.0.0.1:8000/admin/jobs/jobdescription/add/>):

- **Input mode**: `Free text`
- **Raw text**: paste the job description (for your own reference; nothing parses it yet)
- **Requirements**: paste a JSON list. This is the list JD extraction *would* have
  produced. Each item needs all five fields:

  ```json
  [
    {
      "requirement": "5+ years of professional Python experience",
      "description": "Hard requirement stated in the job description",
      "type": "must_have",
      "scale": "",
      "suggested_weight": 0
    },
    {
      "requirement": "Experience with Django or a similar web framework",
      "description": "Stated as a plus",
      "type": "nice_to_have",
      "scale": "5=deep framework experience, 3=some exposure, 1=none",
      "suggested_weight": 3
    },
    {
      "requirement": "Team leadership",
      "description": "Implied by the seniority of the role",
      "type": "inferred",
      "scale": "5=led teams, 3=mentored, 1=none",
      "suggested_weight": 1
    }
  ]
  ```

  `type` must be one of `must_have`, `nice_to_have`, `inferred`. Must-haves are gated
  pass/fail in Stage 1; the other two become Stage 3's scoring criteria. For must-haves
  set `scale` to `""` and `suggested_weight` to `0` — both are unused for that type.

- **Confirmed by user**: ✅ **tick this.** Stages 1–3 refuse to run against an
  unconfirmed job description.

**b. Create a Screening Run** — `Runs → Screening runs → Add`
(<http://127.0.0.1:8000/admin/runs/screeningrun/add/>):

- **Job description**: the one you just made
- **Current stage**: set it to **`Awaiting CV upload and preprocessing`** (`cv_pending`)
- Leave everything else at its default

Note the run's ID from the URL — you need it for every page below.

## 6. Walk the pipeline

Replace `1` with your run's ID. Each screen has a **Confirm** button that advances the
run; nothing advances on its own.

| Step | URL | What you do |
|---|---|---|
| 1 | `/run/1/cvs/upload/` | Upload PDF/DOCX CVs. This only **validates** — no LLM call yet. |
| 2 | `/run/1/cvs/review/` | Press **Structure** to send them to the LLM, check each extraction, then **Confirm** → `stage1_pending` |
| 3 | `/run/1/stage1/review/` | **Run Stage 1**, read every pass *and* fail with reasons, then **Confirm** → `stage2_pending` |
| 4 | `/run/1/stage2/review/` | Edit keywords and the cutoff %, **Run Stage 2**, then **Confirm** → `stage3_pending` |
| 5 | `/run/1/stage3/review/` | Adjust criterion weights, **Run Stage 3**, then **Confirm** → `review` |
| 6 | `/run/1/review/` | Full history per candidate, log overrides, export CSV |

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

## 7. Run the tests

```bash
python -m pytest              # 243 passed, 5 skipped
python -m pytest -v           # see individual test names
```

Five tests are skipped when `BAAI/bge-m3` can't be downloaded (no network, or a
restricted environment). They exercise the real embedding model and run automatically
once it is reachable:

```bash
python -m pytest screening/tests/test_stage2_embeddings_live.py -v
```

No test makes a network call to OpenAI — the LLM is mocked throughout.

## 8. Project layout

```
config/          Django settings, URLs
screening/       ALL pipeline logic — pure Python, ZERO Django imports
  common/          shared text extraction (PDF/DOCX) + the one LLM helper
  jd/              (empty — JD extraction not built)
  cv/              validation, CV_SCHEMA structuring, anonymization
  stage1_hard_filter/
  stage2_ranking/
  stage3_scoring/
  tests/           pytest suite + fixtures
candidates/      Candidate model, upload & review views
jobs/            JobDescription model (models only)
runs/            ScreeningRun, StageResult, Override, stage views, CSV export
```

The rule that matters: **`screening/` never imports Django.** Django views call into it,
never the reverse. That is why the pipeline logic can be tested without a database or a
web server, and it is enforced by a check in the test suite.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: failed to find libmagic.  Check your installation` | The system library is missing — see [Prerequisites](#1-prerequisites). |
| `OpenAIError: Missing credentials...set the OPENAI_API_KEY environment variable` | The key isn't in your environment. Copying `.env` is not enough — `set -a && source .env && set +a`. |
| Upload page says the run is at the wrong stage | `current_stage` must be `cv_pending`. Change it in the admin. |
| "The job description's requirements have not been confirmed yet" | Tick **Confirmed by user** on the JobDescription. |
| Stage 2 hangs on first run | It's downloading the 2 GB model. Watch `~/.cache/huggingface`. |
| A stage says "failed" with a network error | Expected if the API is unreachable. Fix the connection and press the button again — re-running replaces the previous results rather than duplicating them. |
| `database is locked` | SQLite limitation: a background stage and the web UI wrote at the same time. Retry. Known and accepted for v1. |

## 10. Before this goes anywhere near real candidate data

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
