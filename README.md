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

- **An OpenAI API key** — needed for CV structuring and all three screening stages.

## 2. Install

```bash
git clone <your-repo-url>
cd CV-screening-assistant-v1

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Set your API key

Copy the template and fill it in — **`.env` is loaded automatically**, so this is all
you need. Nothing has to be exported in your shell, on any platform: both
`config/settings.py` and `screening/common/llm.py` load it, so the web server, the stage
subprocesses, the tests and a plain `python your_script.py` all see the same values.

```bash
cp .env.example .env
```

```ini
# Standard OpenAI key
OPENAI_API_KEY=sk-...
```

### If your key goes through a gateway

If you were issued a **sub-key** routed through a gateway or proxy rather than talking to
`api.openai.com` directly, set these instead:

```ini
# The key, under either name — OPENAI_API_KEY wins if you set both
KEY_REGULATOR_API_KEY=your-sub-key

# Your gateway's endpoint, under either name — OPENAI_BASE_URL wins if you set both
BASE_URL=https://your-gateway.example.com/v1

# Only if your gateway exposes the model under a different id
OPENAI_MODEL=internal-luna-v2
```

| Variable | When you need it |
|---|---|
| `OPENAI_API_KEY` | The normal case. |
| `KEY_REGULATOR_API_KEY` | Alternative name for the key, for a gateway-issued sub-key. Used only when `OPENAI_API_KEY` is unset. |
| `OPENAI_BASE_URL` | Only when *not* calling `api.openai.com` directly. Leave it out otherwise. |
| `BASE_URL` | Alternative name for the same endpoint. Used only when `OPENAI_BASE_URL` is unset. |
| `OPENAI_MODEL` | Only when your gateway renames the model. Defaults to `gpt-5.6-luna`. |

> ⚠️ **Don't leave a base URL blank.** An empty value is treated as "not set" by this
> project — so a leftover `OPENAI_BASE_URL=` falls through to `BASE_URL` instead of
> shadowing it — but if you removed that handling the OpenAI SDK would resolve it to an
> empty address and every request would fail with a confusing error. Either give it a
> real URL or comment the line out.

A real environment variable always wins over `.env`, so you can override any of these for
a single command:

```bash
OPENAI_MODEL=some-other-model python manage.py runserver
```

### Check what the app will actually use

Run this **from the project root** — no Django bootstrap needed:

```bash
python -c "from screening.common.llm import get_client, get_model; c = get_client(); print('key     :', c.api_key[:8] + '...'); print('endpoint:', c.base_url); print('model   :', get_model())"
```

If the key prints, `.env` is being read. If `endpoint` isn't what you expect, your
`OPENAI_BASE_URL` / `BASE_URL` isn't reaching the app. If you get
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
PDF or DOCX. A scanned PDF with no text layer is rejected; there is no OCR in v1.

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
| 1 | `/run/1/cvs/upload/` | Upload PDF/DOCX CVs. This only **validates** — no LLM call yet. |
| 2 | `/run/1/cvs/review/` | Press **Structure** to send them to the LLM, check each extraction, then **Confirm** → `stage1_pending` |
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

## 9. What a run costs

Every LLM call is timed and its token usage recorded, and each review page and the Final
Review carry a **Run cost** panel:

| Step | Calls | Tokens | Time | Per CV | Cost |
|---|---|---|---|---|---|
| JD extraction | 1 | 2,310 | 4.1 s | — | €0.0074 |
| CV structuring | 12 | 61,200 | 41.0 s | 3.4 s | €0.1710 |
| Stage 1 — hard filter | 12 | 40,560 | 42.8 s | 3.6 s | €0.0728 |
| Stage 2 — ranking | 9 | 52,200 | 34.2 s | 3.8 s | €0.1213 |
| Stage 3 — scoring | 3 | 19,200 | 14.3 s | 4.8 s | €0.0440 |
| **Total** | **37** | **175,470** | **2 min 16 s** | — | **€0.4163** |

**Per CV** is the average inference time for one candidate at that step — JD extraction
has none, being one call for the whole run. The Excel and CSV exports carry the same
figures for each candidate individually.

### Set your rates first

No prices are built in. This app cannot know what your key is billed, particularly
through a gateway sub-key, and a hardcoded guess would produce confident wrong numbers.
Until you enter rates the panel shows exact token counts and timings with the cost column
blank, and says which model is unpriced.

Add them at **<http://127.0.0.1:8000/admin/runs/modelprice/>** — the model id (`gpt-5.6-luna`
unless you set `OPENAI_MODEL`) and three rates in **EUR per million tokens**: input,
cached input, and output. All three come from your provider's pricing page.

The cached-input rate is worth getting right. Stages 1, 2 and 3 send the same job
description and rubric with every candidate, which the API caches and bills at a fraction
of the normal input rate — in the table above that is why Stage 1 costs less than CV
structuring despite a similar token count.

Cost is **computed from stored token counts, not frozen when the call is made**, so
entering a rate prices every run that has already finished, and correcting a rate corrects
history. The token counts themselves never change.

## 10. Run the tests

```bash
python -m pytest              # 413 passed
python -m pytest -v           # see individual test names
```

Nothing is skipped and nothing reaches the network — the LLM is mocked throughout,
Stage 2 included.

## 11. Project layout

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
runs/            ScreeningRun, StageResult, Override, LlmCall, ModelPrice,
                 run list, stage views, usage.py (cost + timing roll-up),
                 exports.py (Excel summary) + CSV export
```

The rule that matters: **`screening/` never imports Django.** Django views call into it,
never the reverse. That is why the pipeline logic can be tested without a database or a
web server, and it is enforced by a check in the test suite.

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: failed to find libmagic.  Check your installation` | The system library is missing — see [Prerequisites](#1-prerequisites). |
| `No LLM API key found in the environment. Set one of: ...` | Neither key variable is set. Check `.env` exists in the project root and holds `OPENAI_API_KEY` or `KEY_REGULATOR_API_KEY`. |
| `404` or "model not found" from a gateway | Usually the **model id**, not the key. Gateways often rename models — set `OPENAI_MODEL` to whatever yours exposes. |
| Connection errors on any stage, JD or CV structuring | These are all LLM calls now. Check `OPENAI_BASE_URL` or `BASE_URL` points at your gateway; run the check snippet in [§3](#3-set-your-api-key). |
| Upload page says the run is at the wrong stage | Confirm the job description first — that is what moves the run to `cv_pending`. |
| "The job description's requirements have not been confirmed yet" | Press **Confirm requirements and continue** on the JD review page. |
| Stage 2 is slow | It makes one API call per surviving candidate. The progress counter advances once per call. |
| The Run cost panel shows `—` for every cost | No rate is set for the model. Add one at `/admin/runs/modelprice/` — see [§9](#9-what-a-run-costs). Token counts and timings are unaffected. |
| The Run cost panel is missing entirely | No LLM call has been recorded for that run yet. It appears as soon as one is. |
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
