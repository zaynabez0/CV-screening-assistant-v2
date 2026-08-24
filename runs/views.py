"""Stage 1 views: trigger the job, report its status, review the verdicts, confirm.

The stage runs in a subprocess-launched management command so the request returns
immediately; the browser polls `stage1_status` until the job reports done or failed.
"""

import subprocess
import sys

from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from candidates.models import Candidate

from .models import ScreeningRun, StageResult
from screening.stage2_ranking.keywords import seed_keywords
from screening.stage2_ranking.ranking import shortlist_size

from .services import StageNotReady, stage1_preflight, stage2_preflight


def start_stage1(request, run_id):
    """POST-only. Launches the Stage 1 job and returns straight away."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage1_review", run_id=run.pk)

    if run.job_status == "running":
        messages.info(request, "A job is already running for this run.")
        return redirect("runs:stage1_review", run_id=run.pk)

    # Check preconditions here as well as in the command, so a user who cannot run the
    # stage is told why now rather than after a subprocess silently fails.
    try:
        candidates = stage1_preflight(run)
    except StageNotReady as exc:
        messages.error(request, str(exc))
        return redirect("runs:stage1_review", run_id=run.pk)

    # Marked running before the subprocess starts: otherwise the redirect can land on a
    # page that still reads "idle" and offers the button again.
    run.mark_job_started("stage1", total=len(candidates))

    subprocess.Popen(
        [sys.executable, "manage.py", "run_stage1", str(run.pk)],
        cwd=str(settings.BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    messages.success(request, f"Stage 1 started for {len(candidates)} candidate(s).")
    return redirect("runs:stage1_review", run_id=run.pk)


def job_status(request, run_id):
    """The endpoint every stage's page polls. Reads the stage-agnostic job_* fields."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    return JsonResponse({
        "status": run.job_status,
        "stage": run.job_stage,
        "message": run.job_message,
        "processed": run.job_processed,
        "total": run.job_total,
        "percent": run.job_percent,
    })


def stage1_review(request, run_id):
    """The Stage 1 review checkpoint.

    Shows every evaluated candidate with per-requirement reasons — failures included.
    Never a survivors-only list: a filtered-out candidate whose reasons are invisible
    cannot be reviewed, and reviewing them is the point of this screen.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)

    results = (
        StageResult.objects.filter(run=run, stage="stage1")
        .select_related("candidate")
        .order_by("-passed", "candidate__pk")
    )
    ready_count = Candidate.objects.filter(run=run, status="ready").count()

    try:
        stage1_preflight(run)
    except StageNotReady as exc:
        blocked = str(exc)
    else:
        blocked = None

    return render(request, "runs/stage1_review.html", {
        "run": run,
        "results": results,
        "passed": [r for r in results if r.passed],
        "failed": [r for r in results if not r.passed],
        "ready_count": ready_count,
        "blocked": blocked,
        "has_results": results.exists(),
    })


def confirm_stage1(request, run_id):
    """POST-only. The human decision point between Stage 1 and Stage 2."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage1_review", run_id=run.pk)

    if run.current_stage != "stage1_pending":
        messages.error(request, f"Run is at '{run.get_current_stage_display()}', not Stage 1.")
        return redirect("runs:stage1_review", run_id=run.pk)

    if run.job_status == "running":
        messages.error(request, "Stage 1 is still running — wait for it to finish.")
        return redirect("runs:stage1_review", run_id=run.pk)

    results = StageResult.objects.filter(run=run, stage="stage1")
    if not results.exists():
        messages.error(request, "Run Stage 1 before confirming.")
        return redirect("runs:stage1_review", run_id=run.pk)

    survivors = results.filter(passed=True).count()
    run.current_stage = "stage2_pending"
    run.job_status = "idle"
    # Seed Stage 2's keyword list from the JD as a starting point the user then edits.
    if not run.stage2_keywords:
        run.stage2_keywords = seed_keywords(run.job_description.requirements)
    run.save(update_fields=["current_stage", "job_status", "stage2_keywords"])
    messages.success(request, f"Stage 1 confirmed. {survivors} candidate(s) continue to Stage 2.")
    return redirect("runs:stage1_review", run_id=run.pk)


# --- Stage 2 ------------------------------------------------------------------

def _parse_keywords(raw: str) -> list:
    """One keyword per line. Blank lines dropped, duplicates removed, order kept."""
    seen, keywords = set(), []
    for line in raw.splitlines():
        keyword = line.strip()
        if keyword and keyword.lower() not in seen:
            seen.add(keyword.lower())
            keywords.append(keyword)
    return keywords


def stage2_review(request, run_id):
    """The Stage 2 review checkpoint.

    Shows the FULL ranked list with both halves of every score — not just the slice
    above the cutoff — plus the two levers: the keyword list and the cutoff percentage.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)

    results = list(
        StageResult.objects.filter(run=run, stage="stage2")
        .select_related("candidate")
        .order_by("output_json__rank")
    )
    cut = shortlist_size(len(results), run.stage2_cutoff_percent)

    rows = [
        {"result": r, "shortlisted": position < cut, "rank": position + 1}
        for position, r in enumerate(results)
    ]

    # Scores were computed against whatever keywords were current at run time. If the
    # list has been edited since, the ranking on screen is stale.
    keywords_used = results[0].output_json.get("keywords_used") if results else None
    stale = bool(results) and keywords_used != list(run.stage2_keywords)

    try:
        stage2_preflight(run)
    except StageNotReady as exc:
        blocked = str(exc)
    else:
        blocked = None

    return render(request, "runs/stage2_review.html", {
        "run": run,
        "rows": rows,
        "keywords_text": "\n".join(run.stage2_keywords),
        "keyword_count": len(run.stage2_keywords),
        "shortlist_count": cut,
        "stale": stale,
        "blocked": blocked,
        "has_results": bool(results),
    })


def save_stage2_settings(request, run_id):
    """POST-only. Persists the recruiter's edits to the keyword list and the cutoff.

    Saving does not re-run anything. The cutoff re-slices the existing ranking for
    free; changed keywords mark the ranking stale and prompt a re-run.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage2_review", run_id=run.pk)

    if run.current_stage != "stage2_pending":
        messages.error(request, f"Run is at '{run.get_current_stage_display()}', not Stage 2.")
        return redirect("runs:stage2_review", run_id=run.pk)

    run.stage2_keywords = _parse_keywords(request.POST.get("keywords", ""))

    raw_cutoff = request.POST.get("cutoff_percent", "")
    try:
        cutoff = float(raw_cutoff)
    except ValueError:
        messages.error(request, f"'{raw_cutoff}' is not a valid cutoff percentage.")
        return redirect("runs:stage2_review", run_id=run.pk)
    if not 0 <= cutoff <= 100:
        messages.error(request, "The cutoff must be between 0 and 100 percent.")
        return redirect("runs:stage2_review", run_id=run.pk)
    run.stage2_cutoff_percent = cutoff

    run.save(update_fields=["stage2_keywords", "stage2_cutoff_percent"])
    messages.success(request, f"Saved {len(run.stage2_keywords)} keyword(s) and a {cutoff:g}% cutoff.")
    return redirect("runs:stage2_review", run_id=run.pk)


def reseed_stage2_keywords(request, run_id):
    """POST-only. Refills the keyword list from the JD's must-have + nice-to-have names.

    Explicit rather than automatic on every visit, so a recruiter who deliberately
    removed a keyword does not find it back on the next page load.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage2_review", run_id=run.pk)

    run.stage2_keywords = seed_keywords(run.job_description.requirements)
    run.save(update_fields=["stage2_keywords"])
    messages.success(request, f"Keyword list re-seeded from the JD ({len(run.stage2_keywords)} keyword(s)).")
    return redirect("runs:stage2_review", run_id=run.pk)


def start_stage2(request, run_id):
    """POST-only. Launches the Stage 2 job and returns straight away."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage2_review", run_id=run.pk)

    if run.job_status == "running":
        messages.info(request, "A job is already running for this run.")
        return redirect("runs:stage2_review", run_id=run.pk)

    try:
        survivors = stage2_preflight(run)
    except StageNotReady as exc:
        messages.error(request, str(exc))
        return redirect("runs:stage2_review", run_id=run.pk)

    run.mark_job_started("stage2", total=len(survivors))

    subprocess.Popen(
        [sys.executable, "manage.py", "run_stage2", str(run.pk)],
        cwd=str(settings.BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    messages.success(request, f"Stage 2 started for {len(survivors)} candidate(s).")
    return redirect("runs:stage2_review", run_id=run.pk)


def confirm_stage2(request, run_id):
    """POST-only. The human decision point between Stage 2 and Stage 3."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage2_review", run_id=run.pk)

    if run.current_stage != "stage2_pending":
        messages.error(request, f"Run is at '{run.get_current_stage_display()}', not Stage 2.")
        return redirect("runs:stage2_review", run_id=run.pk)

    if run.job_status == "running":
        messages.error(request, "Stage 2 is still running — wait for it to finish.")
        return redirect("runs:stage2_review", run_id=run.pk)

    count = StageResult.objects.filter(run=run, stage="stage2").count()
    if not count:
        messages.error(request, "Run Stage 2 before confirming.")
        return redirect("runs:stage2_review", run_id=run.pk)

    shortlisted = shortlist_size(count, run.stage2_cutoff_percent)
    run.current_stage = "stage3_pending"
    run.job_status = "idle"
    run.save(update_fields=["current_stage", "job_status"])
    messages.success(request, f"Stage 2 confirmed. {shortlisted} candidate(s) continue to Stage 3.")
    return redirect("runs:stage2_review", run_id=run.pk)
