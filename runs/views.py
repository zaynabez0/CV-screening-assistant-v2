"""Stage 1 views: trigger the job, report its status, review the verdicts, confirm.

The stage runs in a subprocess-launched management command so the request returns
immediately; the browser polls `stage1_status` until the job reports done or failed.
"""

import csv
import subprocess
import sys

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from candidates.models import Candidate

from .exports import build_summary_workbook
from .models import Override, ScreeningRun, StageResult
from screening.stage2_ranking.keywords import seed_keywords
from screening.stage2_ranking.ranking import shortlist_size

from screening.stage3_scoring.rubric import apply_user_weights, build_rubric, rubric_is_empty

from .services import (
    StageNotReady,
    candidate_history,
    stage1_preflight,
    stage2_preflight,
    stage2_shortlist,
    stage3_preflight,
)


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
        "step_done": run.has_passed("stage1_pending"),
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
        "step_done": run.has_passed("stage2_pending"),
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


# --- Stage 3 ------------------------------------------------------------------

def stage3_review(request, run_id):
    """The Stage 3 checkpoint: set criterion weights, run scoring, read the results.

    When the JD has no nice-to-have or inferred requirements the rubric is empty; the
    page then offers to skip Stage 3 and let Stage 2's ranking stand as the final order.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)

    rubric = build_rubric(run.job_description.requirements)
    weighted = apply_user_weights(rubric, run.rubric_weights or {}) if rubric else []

    # Weights are normalized to sum to 1. Surfaced rather than applied silently: a
    # recruiter who typed the numbers is entitled to see that they were adjusted.
    raw_total = sum(
        (run.rubric_weights or {}).get(c["requirement"], c["suggested_weight"]) for c in rubric
    )
    normalized = bool(rubric) and abs(raw_total - 1.0) > 0.01

    results = list(
        StageResult.objects.filter(run=run, stage="stage3")
        .select_related("candidate")
        .order_by("-output_json__final_score")
    )

    try:
        stage3_preflight(run)
    except StageNotReady as exc:
        blocked = str(exc)
    else:
        blocked = None

    return render(request, "runs/stage3_review.html", {
        "run": run,
        "step_done": run.has_passed("stage3_pending"),
        "rubric": weighted,
        "rubric_empty": not rubric,
        "normalized": normalized,
        "raw_total": raw_total,
        "results": results,
        "shortlist": stage2_shortlist(run),
        "low_confidence_total": sum(len(r.output_json.get("low_confidence_flags", [])) for r in results),
        "blocked": blocked,
        "has_results": bool(results),
    })


def save_stage3_weights(request, run_id):
    """POST-only. Stores {requirement: weight} on the run before scoring runs."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage3_review", run_id=run.pk)

    if run.current_stage != "stage3_pending":
        messages.error(request, f"Run is at '{run.get_current_stage_display()}', not Stage 3.")
        return redirect("runs:stage3_review", run_id=run.pk)

    weights = {}
    for criterion in build_rubric(run.job_description.requirements):
        name = criterion["requirement"]
        raw = request.POST.get(f"weight::{name}", "").strip()
        if raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            messages.error(request, f"'{raw}' is not a valid weight for “{name}”.")
            return redirect("runs:stage3_review", run_id=run.pk)
        if value < 0:
            messages.error(request, f"Weights cannot be negative (“{name}”).")
            return redirect("runs:stage3_review", run_id=run.pk)
        weights[name] = value

    run.rubric_weights = weights
    run.save(update_fields=["rubric_weights"])

    if weights and sum(weights.values()) <= 0:
        messages.warning(request, "Every weight was zero — the criteria will be weighted equally.")
    else:
        messages.success(request, f"Saved weights for {len(weights)} criterion/criteria.")
    return redirect("runs:stage3_review", run_id=run.pk)


def reset_stage3_weights(request, run_id):
    """POST-only. Drops the recruiter's weights back to the JD's suggestions."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage3_review", run_id=run.pk)

    run.rubric_weights = {}
    run.save(update_fields=["rubric_weights"])
    messages.success(request, "Weights reset to the JD's suggested values.")
    return redirect("runs:stage3_review", run_id=run.pk)


def start_stage3(request, run_id):
    """POST-only. Launches the Stage 3 job and returns straight away."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage3_review", run_id=run.pk)

    if run.job_status == "running":
        messages.info(request, "A job is already running for this run.")
        return redirect("runs:stage3_review", run_id=run.pk)

    try:
        shortlist = stage3_preflight(run)
    except StageNotReady as exc:
        messages.error(request, str(exc))
        return redirect("runs:stage3_review", run_id=run.pk)

    run.mark_job_started("stage3", total=len(shortlist))

    subprocess.Popen(
        [sys.executable, "manage.py", "run_stage3", str(run.pk)],
        cwd=str(settings.BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    messages.success(request, f"Stage 3 started for {len(shortlist)} candidate(s).")
    return redirect("runs:stage3_review", run_id=run.pk)


def skip_stage3(request, run_id):
    """POST-only. The empty-rubric path: advance to Final Review without scoring.

    A JD stated entirely as hard requirements has nothing to score, so Stage 2's
    ranking stands as the final order rather than an empty rubric breaking the
    aggregation maths.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage3_review", run_id=run.pk)

    if run.current_stage != "stage3_pending":
        messages.error(request, f"Run is at '{run.get_current_stage_display()}', not Stage 3.")
        return redirect("runs:stage3_review", run_id=run.pk)

    if not rubric_is_empty(run.job_description.requirements):
        messages.error(
            request,
            "This JD has scorable criteria — run Stage 3 rather than skipping it.",
        )
        return redirect("runs:stage3_review", run_id=run.pk)

    run.current_stage = "review"
    run.job_status = "idle"
    run.save(update_fields=["current_stage", "job_status"])
    messages.success(
        request,
        "Stage 3 skipped: the JD has no scorable criteria, so the Stage 2 ranking is the final order.",
    )
    return redirect("runs:stage3_review", run_id=run.pk)


def confirm_stage3(request, run_id):
    """POST-only. The human decision point between Stage 3 and Final Review."""
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:stage3_review", run_id=run.pk)

    if run.current_stage != "stage3_pending":
        messages.error(request, f"Run is at '{run.get_current_stage_display()}', not Stage 3.")
        return redirect("runs:stage3_review", run_id=run.pk)

    if run.job_status == "running":
        messages.error(request, "Stage 3 is still running — wait for it to finish.")
        return redirect("runs:stage3_review", run_id=run.pk)

    count = StageResult.objects.filter(run=run, stage="stage3").count()
    if not count:
        messages.error(request, "Run Stage 3 before confirming.")
        return redirect("runs:stage3_review", run_id=run.pk)

    run.current_stage = "review"
    run.job_status = "idle"
    run.save(update_fields=["current_stage", "job_status"])
    messages.success(request, f"Stage 3 confirmed with {count} scored candidate(s). Final Review is now available.")
    return redirect("runs:stage3_review", run_id=run.pk)


# --- Final Review ---------------------------------------------------------------

def final_review(request, run_id):
    """The complete outcome across every stage, for every candidate.

    Ranked by Stage 3 final_score, but nobody is omitted: a candidate filtered at
    Stage 1 or rejected at upload appears with their reason intact. That is what makes
    the pipeline reviewable rather than merely automatic.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    rows = candidate_history(run)

    return render(request, "runs/final_review.html", {
        "run": run,
        "rows": rows,
        "override_actions": Override.ACTION_CHOICES,
        "scored_count": sum(1 for r in rows if r["stage3"] is not None),
        "override_count": sum(len(r["overrides"]) for r in rows),
        "low_confidence_rows": [r for r in rows if r["low_confidence_flags"]],
        "stage3_skipped": run.current_stage == "review" and not any(r["stage3"] for r in rows),
    })


@login_required
def create_override(request, run_id):
    """POST-only. Logs a recruiter's reversal of a pipeline decision.

    Requires a reason and records who performed it — an override without either is not
    an audit trail. Never edits a StageResult: the automated record stays as written
    and the effective outcome is recomputed from both on read.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    if request.method != "POST":
        return redirect("runs:final_review", run_id=run.pk)

    candidate = get_object_or_404(Candidate, pk=request.POST.get("candidate_id"), run=run)

    action = request.POST.get("action", "")
    if action not in dict(Override.ACTION_CHOICES):
        messages.error(request, f"'{action}' is not a valid override action.")
        return redirect("runs:final_review", run_id=run.pk)

    reason = request.POST.get("reason", "").strip()
    if not reason:
        messages.error(request, "An override needs a reason — it is the audit record.")
        return redirect("runs:final_review", run_id=run.pk)

    adjusted_score = None
    if action == "adjust_score":
        raw = request.POST.get("adjusted_score", "").strip()
        try:
            adjusted_score = float(raw)
        except ValueError:
            messages.error(request, f"'{raw}' is not a valid score.")
            return redirect("runs:final_review", run_id=run.pk)
        if not 0 <= adjusted_score <= 100:
            messages.error(request, "An adjusted score must be between 0 and 100.")
            return redirect("runs:final_review", run_id=run.pk)

    Override.objects.create(
        run=run, candidate=candidate, action=action, reason=reason,
        adjusted_score=adjusted_score, performed_by=request.user,
    )
    messages.success(
        request,
        f"Logged “{dict(Override.ACTION_CHOICES)[action]}” for {candidate.original_filename}.",
    )
    return redirect("runs:final_review", run_id=run.pk)


def export_csv(request, run_id):
    """The final ranked list plus its rationale, one row per candidate.

    Every candidate is exported, including those filtered out — the export is the
    record of what the tool recommended and why, not just the shortlist.

    Note this file carries candidate names and emails: anonymization protects the LLM
    stages, not the recruiter's own review, and a shortlist without names is unusable.
    Treat the download as personal data.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    rows = candidate_history(run)

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="run-{run.pk}-final-review.csv"'

    writer = csv.writer(response)
    writer.writerow([
        "position", "filename", "candidate_name", "email", "outcome",
        "final_score", "automated_score", "score_adjusted", "reinstated",
        "stage1_passed", "stage1_unmet", "stage1_reasons",
        "stage2_rank", "stage2_semantic", "stage2_keyword", "stage2_combined",
        "stage3_criteria", "low_confidence_flags",
        "preprocessing_status", "rejection_reason", "overrides",
    ])

    for row in rows:
        candidate = row["candidate"]
        extracted = candidate.extracted_json or {}
        stage1, stage2, stage3 = row["stage1"], row["stage2"], row["stage3"]

        unmet = [e["requirement"] for e in stage1.unmet_requirements] if stage1 else []
        stage1_reasons = (
            " | ".join(f"{e['requirement']}: {'met' if e['met'] else 'NOT met'} — {e['reason']}"
                       for e in stage1.evaluations)
            if stage1 else ""
        )
        stage3_criteria = (
            " | ".join(f"{cs['criterion']}: {cs['score']}/5 ({cs['confidence']}) — {cs['evidence_quote']}"
                       for cs in stage3.output_json.get("criterion_breakdown", []))
            if stage3 else ""
        )
        overrides = " | ".join(
            f"{o.get_action_display()} by {o.performed_by.username} on "
            f"{o.created_at:%Y-%m-%d}: {o.reason}" for o in row["overrides"]
        )

        writer.writerow([
            row["position"], candidate.original_filename,
            extracted.get("full_name", ""), extracted.get("email", ""), row["outcome"],
            "" if row["effective_score"] is None else row["effective_score"],
            "" if row["automated_score"] is None else row["automated_score"],
            "yes" if row["score_adjusted"] else "",
            "yes" if row["reinstated"] else "",
            "" if stage1 is None else ("pass" if stage1.passed else "fail"),
            "; ".join(unmet), stage1_reasons,
            row["stage2_rank"] or "",
            "" if stage2 is None else round(stage2.output_json.get("semantic_score", 0), 4),
            "" if stage2 is None else round(stage2.output_json.get("keyword_score", 0), 4),
            "" if stage2 is None else round(stage2.output_json.get("final_score", 0), 4),
            stage3_criteria, "; ".join(row["low_confidence_flags"]),
            candidate.get_status_display(), candidate.rejection_reason, overrides,
        ])

    return response


# --- Run list -------------------------------------------------------------------

# Where each stage's checkpoint lives, so the list can link straight to it.
STAGE_URLS = {
    "jd_pending": "jobs:jd_review",
    "cv_pending": "candidates:review",
    "stage1_pending": "runs:stage1_review",
    "stage2_pending": "runs:stage2_review",
    "stage3_pending": "runs:stage3_review",
    "review": "runs:final_review",
    "done": "runs:final_review",
}


def run_list(request):
    """Every screening run and where each one is. The application's front door."""
    runs = (
        ScreeningRun.objects.select_related("job_description")
        .order_by("-created_at", "-pk")
    )
    rows = [{
        "run": run,
        "next_url": reverse(STAGE_URLS[run.current_stage], args=[run.pk]),
        "requirement_count": len(run.job_description.requirements or []),
        "candidate_count": Candidate.objects.filter(run=run).count(),
    } for run in runs]
    return render(request, "runs/run_list.html", {"rows": rows})


def export_xlsx(request, run_id):
    """The run as an Excel workbook: per-candidate outcomes plus the run's settings.

    The readable companion to the CSV. Same underlying data (candidate_history), but
    formatted for a person rather than a parser — and, like the CSV, it carries
    candidate names, so treat the download as personal data.
    """
    run = get_object_or_404(ScreeningRun, pk=run_id)
    workbook = build_summary_workbook(run)

    response = HttpResponse(
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = f'attachment; filename="run-{run.pk}-summary.xlsx"'
    workbook.save(response)
    return response
