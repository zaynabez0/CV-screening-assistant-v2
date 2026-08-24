"""Orchestration for pipeline stages: Django models in, screening/ functions called.

This is the layer the management command runs. Keeping it separate from the command
means the tests exercise the real work without spawning a subprocess.
"""

from candidates.models import Candidate
from screening.cv.anonymization import anonymize_candidate
from screening.stage1_hard_filter.evaluation import evaluate_hard_filters, select_must_have
from screening.stage2_ranking.ranking import score_breakdown, shortlist_size
from screening.stage3_scoring.aggregation import aggregate_score
from screening.stage3_scoring.rubric import apply_user_weights, build_rubric, rubric_is_empty
from screening.stage3_scoring.scoring import score_candidate

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


def stage2_preflight(run) -> list:
    """Everything that must be true before Stage 2 may start. Returns the survivors."""
    if run.current_stage != "stage2_pending":
        raise StageNotReady(
            f"Run is at '{run.get_current_stage_display()}', not ready for Stage 2."
        )
    if not run.job_description.confirmed_by_user:
        raise StageNotReady("The job description's requirements have not been confirmed yet.")

    survivor_ids = StageResult.objects.filter(
        run=run, stage="stage1", passed=True
    ).values_list("candidate_id", flat=True)
    survivors = list(Candidate.objects.filter(pk__in=survivor_ids).order_by("pk"))
    if not survivors:
        raise StageNotReady("No candidates passed Stage 1, so there is nothing to rank.")
    return survivors


def run_stage2(run) -> dict:
    """Rank Stage 1's survivors by a 50/50 blend of semantic similarity and keywords.

    Writes one StageResult per survivor with both halves of the score and its rank.
    The cutoff is NOT applied here: it lives on the run and is applied when the
    shortlist is read, so the recruiter can move it without paying for a re-run.
    """
    survivors = stage2_preflight(run)
    keywords = list(run.stage2_keywords)

    run.mark_job_started("stage2", total=len(survivors))
    run.job_message = "Loading the embedding model and scoring candidates…"
    run.save(update_fields=["job_message"])

    StageResult.objects.filter(run=run, stage="stage2").delete()

    anonymized = [anonymize_candidate(c.extracted_json) for c in survivors]
    breakdown = score_breakdown(run.job_description.requirements, anonymized, keywords)

    # Highest score first; ties keep upload order so the ranking is reproducible.
    order = sorted(range(len(survivors)), key=lambda i: (-breakdown[i]["final_score"], i))

    for rank, index in enumerate(order, start=1):
        StageResult.objects.create(
            run=run,
            candidate=survivors[index],
            stage="stage2",
            passed=None,  # ranking has no pass/fail — the cutoff decides, and it is adjustable
            output_json={**breakdown[index], "rank": rank, "keywords_used": keywords},
        )
        run.mark_job_progress(rank)

    shortlisted = shortlist_size(len(survivors), run.stage2_cutoff_percent)
    run.mark_job_finished(
        f"Stage 2 complete: {len(survivors)} candidate(s) ranked. At the current "
        f"{run.stage2_cutoff_percent:g}% cutoff, {shortlisted} would go to Stage 3."
    )
    return {"ranked": len(survivors), "shortlisted": shortlisted, "keywords_used": keywords}


def stage2_shortlist(run) -> list:
    """The candidates above the Stage 2 cutoff, best-ranked first.

    Derived from the stored ranking and the run's current cutoff rather than frozen at
    Stage 2 time, so moving the cutoff changes who Stage 3 scores without a re-rank.
    """
    ranked = list(
        StageResult.objects.filter(run=run, stage="stage2")
        .select_related("candidate")
        .order_by("output_json__rank")
    )
    cut = shortlist_size(len(ranked), run.stage2_cutoff_percent)
    return [r.candidate for r in ranked[:cut]]


def stage3_preflight(run) -> list:
    """Everything that must be true before Stage 3 may run. Returns the shortlist."""
    if run.current_stage != "stage3_pending":
        raise StageNotReady(
            f"Run is at '{run.get_current_stage_display()}', not ready for Stage 3."
        )
    if not run.job_description.confirmed_by_user:
        raise StageNotReady("The job description's requirements have not been confirmed yet.")
    if rubric_is_empty(run.job_description.requirements):
        raise StageNotReady(
            "This JD has no nice-to-have or inferred requirements, so there is nothing "
            "to score. Skip Stage 3 and let the Stage 2 ranking stand as the final order."
        )

    shortlist = stage2_shortlist(run)
    if not shortlist:
        raise StageNotReady("No candidates are above the Stage 2 cutoff.")
    return shortlist


def weighted_rubric_for(run) -> list:
    """The rubric with the recruiter's confirmed weights applied.

    `rubric_weights` holds {requirement: weight}; anything absent falls back to the
    JD's suggested weight, and apply_user_weights normalizes the result.
    """
    rubric = build_rubric(run.job_description.requirements)
    return apply_user_weights(rubric, run.rubric_weights or {})


def run_stage3(run) -> dict:
    """Score the Stage 2 shortlist per criterion, then aggregate deterministically."""
    shortlist = stage3_preflight(run)
    weighted = weighted_rubric_for(run)

    run.mark_job_started("stage3", total=len(shortlist))
    StageResult.objects.filter(run=run, stage="stage3").delete()

    anomalies = 0
    for index, candidate in enumerate(shortlist, start=1):
        anon = anonymize_candidate(candidate.extracted_json)
        criterion_scores = score_candidate(anon, weighted)
        aggregated = aggregate_score(criterion_scores, weighted)

        StageResult.objects.create(
            run=run,
            candidate=candidate,
            stage="stage3",
            passed=None,  # scoring is not a gate — the recruiter reads the ranking
            output_json={
                **aggregated,
                "weights_used": {c["requirement"]: c["weight"] for c in weighted},
            },
        )
        anomalies += bool(aggregated["unknown_criteria"] or aggregated["unscored_criteria"])
        run.mark_job_progress(index)

    note = f" {anomalies} had criteria the model mis-named or skipped." if anomalies else ""
    run.mark_job_finished(f"Stage 3 complete: {len(shortlist)} candidate(s) scored.{note}")
    return {"scored": len(shortlist), "anomalies": anomalies}
