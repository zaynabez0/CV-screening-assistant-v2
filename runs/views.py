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
from .services import StageNotReady, stage1_preflight


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


def stage1_status(request, run_id):
    """The endpoint the page polls. Small JSON, no rendering."""
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
    run.save(update_fields=["current_stage", "job_status"])
    messages.success(request, f"Stage 1 confirmed. {survivors} candidate(s) continue to Stage 2.")
    return redirect("runs:stage1_review", run_id=run.pk)
