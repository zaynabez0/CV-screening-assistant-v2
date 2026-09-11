"""CV preprocessing views: upload -> validate -> structure -> review checkpoint.

Every step here is user-triggered. Uploading does not structure; structuring does not
advance the run. The recruiter presses each button, and the run only leaves CV
preprocessing when they confirm the review checkpoint.
"""

import subprocess
import sys

from django.conf import settings
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render

from runs.models import ScreeningRun
from runs.services import StageNotReady, structuring_preflight
from runs.usage import step_cost
from screening.cv.validation import (
    CVValidationError,
    DuplicateFileError,
    check_batch_size,
    validate_cv_file,
)

from .models import Candidate

# The only stage at which CV preprocessing may be touched.
REQUIRED_STAGE = "cv_pending"


def _require_stage(run):
    """Returns an error string if the run isn't at CV preprocessing, else None.

    `current_stage` is the single source of truth for what the user can do right now.
    """
    if run.current_stage != REQUIRED_STAGE:
        return (
            f"This run is at '{run.get_current_stage_display()}', not CV preprocessing. "
            f"CV upload is only available while current_stage is '{REQUIRED_STAGE}'."
        )
    return None


def upload_cvs(request, run_id):
    """GET: the upload form. POST: validate and record each file in the batch."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    blocked = _require_stage(run)

    if request.method == "POST":
        if blocked:
            messages.error(request, blocked)
            return redirect("candidates:upload", run_id=run.pk)

        uploaded = request.FILES.getlist("cvs")
        if not uploaded:
            messages.error(request, "No files selected.")
            return redirect("candidates:upload", run_id=run.pk)

        # Step 3, batch half: reject the whole batch before writing anything to disk.
        try:
            check_batch_size(sum(f.size for f in uploaded))
        except CVValidationError as exc:
            messages.error(request, f"Batch rejected: {exc.reason}")
            return redirect("candidates:upload", run_id=run.pk)

        accepted, rejected, duplicates = 0, 0, 0
        for upload in uploaded:
            candidate = Candidate.objects.create(
                run=run,
                file=upload,
                original_filename=upload.name,
                file_hash="",
            )
            # Hashes already recorded for THIS run — step 5 compares against these.
            seen = set(
                Candidate.objects.filter(run=run)
                .exclude(pk=candidate.pk)
                .exclude(file_hash="")
                .values_list("file_hash", flat=True)
            )
            try:
                result = validate_cv_file(candidate.file.path, upload.name, seen_hashes=seen)
            except DuplicateFileError as exc:
                candidate.status = "duplicate"
                candidate.rejection_reason = exc.reason
                duplicates += 1
            except CVValidationError as exc:
                candidate.status = "rejected"
                candidate.rejection_reason = exc.reason
                rejected += 1
            else:
                candidate.file_hash = result["file_hash"]
                candidate.extracted_text = result["text"]  # so structuring need not OCR again
                candidate.status = "pending"  # validated; awaiting structuring
                accepted += 1
            candidate.save()

        messages.success(
            request,
            f"{accepted} file(s) validated, {rejected} rejected, {duplicates} duplicate(s). "
            "Nothing has been sent to the LLM yet.",
        )
        return redirect("candidates:review", run_id=run.pk)

    return render(
        request,
        "candidates/upload.html",
        {"run": run, "blocked": blocked, "candidates": Candidate.objects.filter(run=run)},
    )


def structure_cvs(request, run_id):
    """POST-only. Launches the structuring job and returns straight away.

    One LLM call per CV, so a batch of fifty is minutes of work. It used to run inline and
    the browser simply hung; now it is a background command and the review page shows a
    progress bar, the same arrangement the three stages use.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("candidates:review", run_id=run.pk)

    if run.job_status == "running":
        messages.info(request, "A job is already running for this run.")
        return redirect("candidates:review", run_id=run.pk)

    # Checked here as well as in the command, so a user who cannot run it is told why now
    # rather than after a subprocess silently fails.
    try:
        pending = structuring_preflight(run)
    except StageNotReady as exc:
        messages.error(request, str(exc))
        return redirect("candidates:review", run_id=run.pk)

    # Marked running before the subprocess starts: otherwise the redirect can land on a
    # page that still reads "idle" and offers the button again.
    run.mark_job_started("cv_structuring", total=len(pending))

    subprocess.Popen(
        [sys.executable, "manage.py", "run_cv_structuring", str(run.pk)],
        cwd=str(settings.BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    messages.success(request, f"Structuring started for {len(pending)} CV(s).")
    return redirect("candidates:review", run_id=run.pk)


def review_candidates(request, run_id):
    """The CV preprocessing review checkpoint.

    Shows every candidate's extracted_json and extraction_confidence, low-confidence
    ones flagged. This is where a garbled extraction gets caught before it silently
    produces a wrong Stage 1 decision.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    candidates = Candidate.objects.filter(run=run).select_related("run")

    ready = [c for c in candidates if c.status == "ready"]
    return render(
        request,
        "candidates/review.html",
        {
            "run": run,
            "candidates": candidates,
            "blocked": _require_stage(run),
            "step_done": run.has_passed(REQUIRED_STAGE),
            "awaiting_structuring": [c for c in candidates if c.status == "pending"],
            "low_confidence_count": sum(1 for c in ready if c.is_low_confidence),
            "ready_count": len(ready),
            "step_cost": step_cost(run, "cv_structuring"),
        },
    )


def confirm_preprocessing(request, run_id):
    """POST-only. The human decision point that ends CV preprocessing.

    Nothing advances the run but this. Requires at least one structured candidate, and
    warns rather than blocks when low-confidence extractions are being carried forward.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("candidates:review", run_id=run.pk)

    blocked = _require_stage(run)
    if blocked:
        messages.error(request, blocked)
        return redirect("candidates:review", run_id=run.pk)

    if run.job_status == "running":
        messages.error(request, "Structuring is still running — wait for it to finish.")
        return redirect("candidates:review", run_id=run.pk)

    ready = Candidate.objects.filter(run=run, status="ready")
    if not ready.exists():
        messages.error(request, "Structure at least one CV before confirming.")
        return redirect("candidates:review", run_id=run.pk)

    run.current_stage = "stage1_pending"
    run.job_status = "idle"          # clears the finished-job banner, as confirm_stage1 does
    run.save(update_fields=["current_stage", "job_status"])
    messages.success(request, f"CV preprocessing confirmed with {ready.count()} candidate(s). Stage 1 is now available.")
    return redirect("candidates:review", run_id=run.pk)
