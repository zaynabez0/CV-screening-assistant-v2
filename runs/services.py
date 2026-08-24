"""Orchestration for pipeline stages: Django models in, screening/ functions called.

This is the layer the management command runs. Keeping it separate from the command
means the tests exercise the real work without spawning a subprocess.
"""

from candidates.models import Candidate
from screening.cv.anonymization import anonymize_candidate
from screening.stage1_hard_filter.evaluation import evaluate_hard_filters, select_must_have

from .models import StageResult


class StageNotReady(Exception):
    """The run is not in a state where this stage may run."""


def stage1_preflight(run) -> list:
    """Everything that must be true before Stage 1 may start. Returns the candidates.

    Raises StageNotReady with a message the recruiter can act on — checked both by the
    trigger view (so the user sees it immediately) and by the command (so a stale
    subprocess can't run against a run that has since moved on).
    """
    if run.current_stage != "stage1_pending":
        raise StageNotReady(
            f"Run is at '{run.get_current_stage_display()}', not ready for Stage 1."
        )
    if not run.job_description.confirmed_by_user:
        raise StageNotReady("The job description's requirements have not been confirmed yet.")

    candidates = list(Candidate.objects.filter(run=run, status="ready").order_by("pk"))
    if not candidates:
        raise StageNotReady("No structured candidates in this run.")
    return candidates


def run_stage1(run) -> dict:
    """Evaluate every ready candidate against the JD's must-have requirements.

    Writes one StageResult per candidate — including failures. A rejected candidate's
    reasons must stay available: GDPR Art. 22 means nothing is auto-rejected out of
    sight, and the review checkpoint shows the full list, not just survivors.
    """
    candidates = stage1_preflight(run)
    must_haves = select_must_have(run.job_description.requirements)

    run.mark_job_started("stage1", total=len(candidates))

    # Re-running replaces the previous verdicts rather than stacking a second set.
    StageResult.objects.filter(run=run, stage="stage1").delete()

    passed_count = 0
    for index, candidate in enumerate(candidates, start=1):
        anon = anonymize_candidate(candidate.extracted_json)
        result = evaluate_hard_filters(anon, must_haves)

        StageResult.objects.create(
            run=run,
            candidate=candidate,
            stage="stage1",
            passed=result["passed"],
            output_json={"evaluations": result["evaluations"]},
        )
        passed_count += bool(result["passed"])
        run.mark_job_progress(index)

    summary = {
        "evaluated": len(candidates),
        "passed": passed_count,
        "failed": len(candidates) - passed_count,
        "must_have_count": len(must_haves),
    }
    note = "" if must_haves else " (the JD has no must-have requirements, so nothing was gated)"
    run.mark_job_finished(
        f"Stage 1 complete: {passed_count} of {len(candidates)} candidates met every "
        f"must-have requirement{note}."
    )
    return summary
