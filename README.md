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
- **~2.5 GB of free disk** — Stage 2 uses a self-hosted embedding model, `BAAI/bge-m3`,
  downloaded on first use. See [§5](#5-prepare-the-embedding-model) — fetch it before your
  first run rather than during it.

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

Copy the template and fill it in — **`.env` is loaded automatically**, so this is all
you need:

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

```bash
python -c "
import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings'); django.setup()
from screening.common.llm import get_client, get_model
c = get_client()
print('key    :', c.api_key[:8] + '...')
print('endpoint:', c.base_url)
print('model  :', get_model())
"
```

If the key prints, `.env` is being read. If `endpoint` isn't what you expect, your
`OPENAI_BASE_URL` / `BASE_URL` isn't reaching the app.

Only Stages 1 and 3, CV structuring and JD extraction use this key — Stage 2 runs the
embedding model locally and needs no key at all. `.env` is gitignored; keep it that way.

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

## 5. Prepare the embedding model

Stage 2 ranks candidates by **meaning**, not just keywords — a CV that says
"server-side programming for web services" scores close to a job description asking for
"Python backend development", even with no words in common. That is done by a model
called **`BAAI/bge-m3`**, which runs on your own machine: no API account, no per-use
cost, no candidate data leaving your computer.

It was chosen for multilingual coverage — French and English CVs land in one ranking
rather than being ranked by language.

### Download it before your first run

The model is about **2 GB** and is fetched the first time Stage 2 runs. If you skip this
step, your first Stage 2 will look like it has hung: the download happens inside a
background process whose output is hidden, so the page just sits at `0 / N` for several
minutes with no explanation.

Fetch it once, up front, where you can watch the progress bar:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

Let it finish. Every run after this loads from the local cache in a few seconds.

### Check it works

This prints the actual similarity scores Stage 2 uses — worth running once to see what
the model is doing before you trust it on real CVs:

```bash
python -c "
from screening.stage2_ranking.embeddings import semantic_scores
jd = 'Python backend development for web services'
candidates = [
    'Python backend development for web services',                # identical
    'Server-side programming in Django and Flask',                # same job, different words
    'Ingenieur logiciel backend developpant des services web',    # same job, in French
    'Pastry chef preparing desserts and viennoiseries',           # unrelated
]
for text, score in zip(candidates, semantic_scores(jd, candidates)):
    print(f'{score:.3f}  {text}')
"
```

Scores run from -1 to 1. Expect the identical line near 1.0, with the paraphrase and the
French line clearly above the pastry chef. If the unrelated line scores as high as the
others, something is wrong — don't trust the Stage 2 ranking until it's fixed.

You can also run the five tests that exercise the real model:

```bash
python -m pytest screening/tests/test_stage2_embeddings_live.py -v
```

They skip automatically when the model isn't available, which is why the main test run
reports 5 skipped until you've downloaded it.

### Where it lives, and working offline

The model is cached in `~/.cache/huggingface/hub`. To keep it somewhere else — a
different drive, a shared folder — set `HF_HOME` before running anything:

```bash
export HF_HOME=/path/to/your/cache
```

Once downloaded, Stage 2 needs no internet at all. To be certain nothing reaches out to
the network, run with:

```bash
export HF_HUB_OFFLINE=1
```

With that set, a missing model fails immediately with a clear error instead of silently
trying to download.

### If the download fails

| Symptom | What it means |
|---|---|
| Hangs, then a connection error | No internet, or a proxy/firewall blocking `huggingface.co`. There is no alternative source — fix the connection. |
| `No space left on device` | Needs ~2.5 GB free. Check the drive holding `HF_HOME`. |
| Stage 2 fails but Stages 1 and 3 work | Those use the OpenAI API; only Stage 2 uses this local model. The problem is the model, not your API key. |
| The process is killed part-way | Usually not enough memory. Close other applications and retry — the download resumes where it stopped. |

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

- **Stage 2 is slow the first time** unless you pre-downloaded the model — see
  [Prepare the embedding model](#5-prepare-the-embedding-model).
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

## 9. Run the tests

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

## 10. Project layout

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

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: failed to find libmagic.  Check your installation` | The system library is missing — see [Prerequisites](#1-prerequisites). |
| `No LLM API key found in the environment. Set one of: ...` | Neither key variable is set. Check `.env` exists in the project root and holds `OPENAI_API_KEY` or `KEY_REGULATOR_API_KEY`. |
| `404` or "model not found" from a gateway | Usually the **model id**, not the key. Gateways often rename models — set `OPENAI_MODEL` to whatever yours exposes. |
| Connection errors only on Stages 1/3, JD or CV structuring | Those are the LLM calls. Check `OPENAI_BASE_URL` or `BASE_URL` points at your gateway; run the check snippet in [§3](#3-set-your-api-key). |
| Upload page says the run is at the wrong stage | Confirm the job description first — that is what moves the run to `cv_pending`. |
| "The job description's requirements have not been confirmed yet" | Press **Confirm requirements and continue** on the JD review page. |
| Stage 2 hangs on first run | It's downloading the 2 GB model. Watch `~/.cache/huggingface`. |
| A stage says "failed" with a network error | Expected if the API is unreachable. Fix the connection and press the button again — re-running replaces the previous results rather than duplicating them. |
| `database is locked` | SQLite limitation: a background stage and the web UI wrote at the same time. Retry. Known and accepted for v1. |
| A confirmed job description can't be edited | By design — the stages already judged against it would no longer match. Start a new run. |

## 12. Before this goes anywhere near real candidate data

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
