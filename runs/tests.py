"""Django-side Stage 1 tests: the job orchestration, the review page, the confirm gate.

The LLM is mocked throughout. The subprocess is mocked in the trigger-view tests and
bypassed in the work tests, which call the service directly — that way the real
evaluation logic is exercised without spawning a process the test runner can't see.
"""

import csv
import io
import json
from unittest.mock import patch

import pytest
from django.core.management import CommandError, call_command
from openpyxl import load_workbook
from django.urls import reverse

from candidates.models import Candidate
from jobs.models import JobDescription
from runs.models import Override, ScreeningRun, StageResult
from runs.services import (
    StageNotReady,
    candidate_history,
    run_stage1,
    run_stage2,
    run_stage3,
)

pytestmark = pytest.mark.django_db

MUST_HAVE = {"requirement": "5+ years Python", "description": "Professional Python",
             "type": "must_have", "scale": "", "suggested_weight": 0}
NICE_TO_HAVE = {"requirement": "Kubernetes", "description": "Container orchestration",
                "type": "nice_to_have", "scale": "5=expert, 3=some, 1=none", "suggested_weight": 0.5}


def extraction(name="Amina Benali", years=6):
    return {
        "full_name": name, "email": f"{name.split()[0].lower()}@example.com", "phone": "+212600000000",
        "languages": [{"language": "French", "proficiency": "fluent"}],
        "education": [{"degree": "MSc", "field": "CS", "institution": "ENSIAS", "end_year": 2021}],
        "experience": [{"title": "Backend Developer", "company": "Atlas Data",
                        "start_date": "2018-01", "end_date": None, "description": "Python services.",
                        "evidence_quote": "Built Python ETL pipelines"}],
        "skills": ["Python", "SQL"], "total_years_experience": years, "certifications": [],
        "extraction_confidence": "high", "extraction_notes": "",
    }


@pytest.fixture
def run():
    jd = JobDescription.objects.create(
        input_mode="text", raw_text="Backend engineer.",
        requirements=[MUST_HAVE, NICE_TO_HAVE], confirmed_by_user=True,
    )
    return ScreeningRun.objects.create(job_description=jd, current_stage="stage1_pending")


def add_candidate(run, filename, name="Amina Benali", years=6, status="ready"):
    return Candidate.objects.create(
        run=run, file=f"cvs/{filename}", original_filename=filename,
        file_hash=filename.ljust(64, "0"), status=status,
        extracted_json=extraction(name, years) if status == "ready" else None,
    )


def verdict(met: bool):
    return {"evaluations": [{
        "requirement": "5+ years Python", "met": met,
        "evidence": "Built Python ETL pipelines" if met else "",
        "reason": "6 years stated." if met else "No Python experience stated.",
    }]}


# --- The two anchor cases, end to end through the service --------------------

def test_a_passing_candidate_is_recorded_as_passed(run):
    candidate = add_candidate(run, "pass.pdf")

    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(True)):
        summary = run_stage1(run)

    result = StageResult.objects.get(run=run, candidate=candidate, stage="stage1")
    assert result.passed is True
    assert result.output_json["evaluations"][0]["evidence"] == "Built Python ETL pipelines"
    assert summary == {"evaluated": 1, "passed": 1, "failed": 0, "must_have_count": 1}


def test_a_failing_candidate_is_recorded_with_its_reason_not_deleted(run):
    """GDPR Art. 22: a filtered-out candidate keeps a stored, human-readable reason."""
    candidate = add_candidate(run, "fail.pdf")

    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(False)):
        run_stage1(run)

    result = StageResult.objects.get(run=run, candidate=candidate, stage="stage1")
    assert result.passed is False
    assert result.output_json["evaluations"][0]["reason"] == "No Python experience stated."
    assert Candidate.objects.filter(pk=candidate.pk).exists()      # nothing is deleted
    assert candidate.status == "ready"                              # nor silently re-statused


def test_only_must_have_requirements_are_sent(run):
    """The nice_to_have on this JD belongs to Stage 3 and must not gate anyone here."""
    add_candidate(run, "cv.pdf")

    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               return_value=verdict(True)) as mock:
        run_stage1(run)

    instructions = mock.call_args.args[0]
    assert "5+ years Python" in instructions
    assert "Kubernetes" not in instructions


def test_the_candidate_sent_to_the_llm_is_anonymized(run):
    add_candidate(run, "cv.pdf", name="Amina Benali")

    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               return_value=verdict(True)) as mock:
        run_stage1(run)

    sent = mock.call_args.args[1]
    assert "Amina Benali" not in sent
    assert "example.com" not in sent
    assert "+212600000000" not in sent
    assert "Python" in sent                                          # the useful data survives


def test_every_ready_candidate_gets_a_row_including_failures(run):
    add_candidate(run, "a.pdf")
    add_candidate(run, "b.pdf", name="Youssef El Amrani")
    verdicts = [verdict(True), verdict(False)]

    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               side_effect=verdicts):
        summary = run_stage1(run)

    assert StageResult.objects.filter(run=run, stage="stage1").count() == 2
    assert summary["passed"] == 1 and summary["failed"] == 1


def test_unstructured_candidates_are_skipped(run):
    add_candidate(run, "ready.pdf")
    add_candidate(run, "rejected.pdf", status="rejected")

    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               return_value=verdict(True)) as mock:
        run_stage1(run)

    assert mock.call_count == 1
    assert StageResult.objects.filter(run=run).count() == 1


def test_rerunning_replaces_verdicts_rather_than_stacking_them(run):
    add_candidate(run, "cv.pdf")

    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(True)):
        run_stage1(run)
    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(False)):
        run_stage1(run)

    results = StageResult.objects.filter(run=run, stage="stage1")
    assert results.count() == 1
    assert results.first().passed is False


# --- Preconditions --------------------------------------------------------------

def test_stage1_refuses_to_run_at_the_wrong_stage(run):
    run.current_stage = "cv_pending"
    run.save()
    add_candidate(run, "cv.pdf")

    with pytest.raises(StageNotReady, match="not ready for Stage 1"):
        run_stage1(run)


def test_stage1_refuses_an_unconfirmed_job_description(run):
    """Nothing downstream reads an unconfirmed requirements list."""
    run.job_description.confirmed_by_user = False
    run.job_description.save()
    add_candidate(run, "cv.pdf")

    with pytest.raises(StageNotReady, match="not been confirmed"):
        run_stage1(run)


def test_stage1_refuses_a_run_with_no_structured_candidates(run):
    with pytest.raises(StageNotReady, match="No structured candidates"):
        run_stage1(run)


# --- Job status the frontend polls -----------------------------------------------

def test_job_status_reaches_done_with_progress(run):
    add_candidate(run, "a.pdf")
    add_candidate(run, "b.pdf")

    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(True)):
        run_stage1(run)

    run.refresh_from_db()
    assert run.job_status == "done"
    assert (run.job_processed, run.job_total) == (2, 2)
    assert run.job_percent == 100
    assert "2 of 2" in run.job_message


def test_the_command_records_a_failure_instead_of_leaving_the_job_running(run):
    """Without this the UI polls a 'running' job forever after an API outage."""
    add_candidate(run, "cv.pdf")

    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               side_effect=RuntimeError("API down")):
        with pytest.raises(RuntimeError):
            call_command("run_stage1", run.pk)

    run.refresh_from_db()
    assert run.job_status == "failed"
    assert "API down" in run.job_message


def test_the_command_records_a_precondition_failure_too(run):
    run.current_stage = "cv_pending"
    run.save()

    with pytest.raises(CommandError):
        call_command("run_stage1", run.pk)

    run.refresh_from_db()
    assert run.job_status == "failed"
    assert "not ready for Stage 1" in run.job_message


def test_the_command_runs_the_stage(run):
    add_candidate(run, "cv.pdf")

    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(True)):
        call_command("run_stage1", run.pk)

    assert StageResult.objects.filter(run=run, stage="stage1", passed=True).count() == 1


def test_status_endpoint_returns_pollable_json(client, run):
    run.mark_job_started("stage1", total=4)
    run.mark_job_progress(1)

    data = json.loads(client.get(reverse("runs:job_status", args=[run.pk])).content)

    assert data["status"] == "running"
    assert data["processed"] == 1 and data["total"] == 4
    assert data["percent"] == 25


# --- Trigger view (subprocess mocked) ---------------------------------------------

def test_start_launches_the_management_command_in_a_subprocess(client, run):
    add_candidate(run, "cv.pdf")

    with patch("runs.views.subprocess.Popen") as popen:
        client.post(reverse("runs:stage1_start", args=[run.pk]))

    argv = popen.call_args.args[0]
    assert argv[1:] == ["manage.py", "run_stage1", str(run.pk)]
    run.refresh_from_db()
    assert run.job_status == "running"          # set before the redirect, not by the child


def test_start_refuses_when_preconditions_fail_and_launches_nothing(client, run):
    run.job_description.confirmed_by_user = False
    run.job_description.save()
    add_candidate(run, "cv.pdf")

    with patch("runs.views.subprocess.Popen") as popen:
        client.post(reverse("runs:stage1_start", args=[run.pk]))

    popen.assert_not_called()
    run.refresh_from_db()
    assert run.job_status == "idle"


def test_start_does_not_launch_a_second_job_while_one_runs(client, run):
    add_candidate(run, "cv.pdf")
    run.mark_job_started("stage1", total=1)

    with patch("runs.views.subprocess.Popen") as popen:
        client.post(reverse("runs:stage1_start", args=[run.pk]))

    popen.assert_not_called()


# --- Review checkpoint ------------------------------------------------------------

def test_review_shows_failures_with_their_reasons_not_just_survivors(client, run):
    add_candidate(run, "pass.pdf")
    add_candidate(run, "fail.pdf", name="Youssef El Amrani")

    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               side_effect=[verdict(True), verdict(False)]):
        run_stage1(run)

    html = client.get(reverse("runs:stage1_review", args=[run.pk])).content.decode()

    assert "pass.pdf" in html and "fail.pdf" in html
    assert "No Python experience stated." in html           # the failure's reason
    assert "Built Python ETL pipelines" in html             # the pass's evidence
    assert "5+ years Python" in html                        # per-requirement, not a summary


def test_review_separates_passed_and_failed(client, run):
    add_candidate(run, "a.pdf")
    add_candidate(run, "b.pdf")

    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               side_effect=[verdict(True), verdict(False)]):
        run_stage1(run)

    context = client.get(reverse("runs:stage1_review", args=[run.pk])).context

    assert len(context["passed"]) == 1
    assert len(context["failed"]) == 1


def test_review_page_works_before_stage1_has_run(client, run):
    add_candidate(run, "cv.pdf")

    response = client.get(reverse("runs:stage1_review", args=[run.pk]))

    assert response.status_code == 200
    assert response.context["has_results"] is False


# --- Confirm gate -------------------------------------------------------------------

def test_confirming_advances_to_stage2(client, run):
    add_candidate(run, "cv.pdf")
    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(True)):
        run_stage1(run)

    client.post(reverse("runs:stage1_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "stage2_pending"


def test_running_stage1_alone_does_not_advance_the_run(client, run):
    """No stage auto-runs the next — only the human confirmation moves current_stage."""
    add_candidate(run, "cv.pdf")

    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(True)):
        run_stage1(run)

    run.refresh_from_db()
    assert run.current_stage == "stage1_pending"


def test_cannot_confirm_before_stage1_has_run(client, run):
    add_candidate(run, "cv.pdf")

    client.post(reverse("runs:stage1_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "stage1_pending"


def test_cannot_confirm_while_the_job_is_still_running(client, run):
    add_candidate(run, "cv.pdf")
    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(True)):
        run_stage1(run)
    run.mark_job_started("stage1", total=1)

    client.post(reverse("runs:stage1_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "stage1_pending"


def test_confirming_with_every_candidate_failed_is_allowed(client, run):
    """An empty shortlist is a real outcome the recruiter may accept — the tool ranks
    and recommends, it does not block the human's decision."""
    add_candidate(run, "cv.pdf")
    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(False)):
        run_stage1(run)

    client.post(reverse("runs:stage1_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "stage2_pending"


# =============================================================================
# Stage 2 — ranking
# =============================================================================

def stage2_run(**kwargs):
    jd = JobDescription.objects.create(
        input_mode="text", raw_text="Backend engineer.",
        requirements=[MUST_HAVE, NICE_TO_HAVE], confirmed_by_user=True,
    )
    defaults = {"job_description": jd, "current_stage": "stage2_pending"}
    defaults.update(kwargs)
    return ScreeningRun.objects.create(**defaults)


def pass_stage1(run, candidate, passed=True):
    return StageResult.objects.create(
        run=run, candidate=candidate, stage="stage1", passed=passed,
        output_json={"evaluations": []},
    )


@pytest.fixture
def ranked_run():
    """A run at stage2_pending with three Stage 1 survivors."""
    run = stage2_run()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        pass_stage1(run, add_candidate(run, name))
    return run


@pytest.fixture
def fixed_semantics(monkeypatch):
    """Pin semantic_scores so ranking is deterministic without loading BGE-M3."""
    def install(scores):
        monkeypatch.setattr("screening.stage2_ranking.ranking.semantic_scores",
                            lambda jd_text, texts: list(scores))
    return install


# --- Running the stage ---------------------------------------------------------

def test_stage2_ranks_every_survivor_highest_first(ranked_run, fixed_semantics):
    fixed_semantics([0.2, 0.9, 0.5])

    summary = run_stage2(ranked_run)

    results = StageResult.objects.filter(run=ranked_run, stage="stage2").order_by("output_json__rank")
    assert [r.candidate.original_filename for r in results] == ["b.pdf", "c.pdf", "a.pdf"]
    assert [r.output_json["rank"] for r in results] == [1, 2, 3]
    assert summary["ranked"] == 3


def test_stage2_stores_both_halves_of_each_score(ranked_run, fixed_semantics):
    """The recruiter cannot sensibly edit keywords without seeing which half drove
    a candidate's position."""
    fixed_semantics([0.8, 0.4, 0.6])
    ranked_run.stage2_keywords = ["Python"]
    ranked_run.save()

    run_stage2(ranked_run)

    top = StageResult.objects.get(run=ranked_run, stage="stage2", output_json__rank=1)
    assert top.output_json["semantic_score"] == 0.8
    assert top.output_json["keyword_score"] == 1.0          # every candidate lists Python
    assert top.output_json["final_score"] == pytest.approx(0.9)


def test_stage2_results_have_no_pass_fail(ranked_run, fixed_semantics):
    """Ranking is not a gate — `passed` is null, per the StageResult contract."""
    fixed_semantics([0.5, 0.5, 0.5])

    run_stage2(ranked_run)

    assert all(r.passed is None for r in StageResult.objects.filter(run=ranked_run, stage="stage2"))


def test_stage2_records_the_keywords_that_produced_the_scores(ranked_run, fixed_semantics):
    fixed_semantics([0.5, 0.5, 0.5])
    ranked_run.stage2_keywords = ["Python", "SQL"]
    ranked_run.save()

    run_stage2(ranked_run)

    result = StageResult.objects.filter(run=ranked_run, stage="stage2").first()
    assert result.output_json["keywords_used"] == ["Python", "SQL"]


def test_stage2_only_ranks_stage1_survivors(fixed_semantics):
    run = stage2_run()
    survivor = add_candidate(run, "passed.pdf")
    rejected = add_candidate(run, "failed.pdf")
    pass_stage1(run, survivor, passed=True)
    pass_stage1(run, rejected, passed=False)
    fixed_semantics([0.5])

    run_stage2(run)

    results = StageResult.objects.filter(run=run, stage="stage2")
    assert [r.candidate.original_filename for r in results] == ["passed.pdf"]


def test_stage2_sends_anonymized_candidates_to_the_ranker(ranked_run, monkeypatch):
    captured = {}

    def spy(jd_text, texts):
        captured["texts"] = texts
        return [0.5] * len(texts)

    monkeypatch.setattr("screening.stage2_ranking.ranking.semantic_scores", spy)
    run_stage2(ranked_run)

    joined = "\n".join(captured["texts"])
    assert "Amina Benali" not in joined and "example.com" not in joined


def test_rerunning_stage2_replaces_the_previous_ranking(ranked_run, fixed_semantics):
    fixed_semantics([0.2, 0.9, 0.5])
    run_stage2(ranked_run)
    fixed_semantics([0.9, 0.2, 0.5])
    run_stage2(ranked_run)

    results = StageResult.objects.filter(run=ranked_run, stage="stage2").order_by("output_json__rank")
    assert results.count() == 3
    assert results.first().candidate.original_filename == "a.pdf"


def test_ties_break_on_upload_order_so_ranking_is_reproducible(ranked_run, fixed_semantics):
    fixed_semantics([0.5, 0.5, 0.5])

    run_stage2(ranked_run)

    results = StageResult.objects.filter(run=ranked_run, stage="stage2").order_by("output_json__rank")
    assert [r.candidate.original_filename for r in results] == ["a.pdf", "b.pdf", "c.pdf"]


# --- Preconditions --------------------------------------------------------------

def test_stage2_refuses_at_the_wrong_stage(ranked_run):
    ranked_run.current_stage = "stage1_pending"
    ranked_run.save()

    with pytest.raises(StageNotReady, match="not ready for Stage 2"):
        run_stage2(ranked_run)


def test_stage2_refuses_when_nobody_passed_stage1():
    run = stage2_run()
    pass_stage1(run, add_candidate(run, "failed.pdf"), passed=False)

    with pytest.raises(StageNotReady, match="nothing to rank"):
        run_stage2(run)


def test_stage2_command_records_failure_rather_than_hanging(ranked_run, monkeypatch):
    monkeypatch.setattr("screening.stage2_ranking.ranking.semantic_scores",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("model download failed")))

    with pytest.raises(RuntimeError):
        call_command("run_stage2", ranked_run.pk)

    ranked_run.refresh_from_db()
    assert ranked_run.job_status == "failed"
    assert "model download failed" in ranked_run.job_message


# --- Keyword seeding --------------------------------------------------------------

def test_confirming_stage1_seeds_the_keyword_list_from_the_jd(client):
    run = stage2_run(current_stage="stage1_pending")
    candidate = add_candidate(run, "cv.pdf")
    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(True)):
        run_stage1(run)

    client.post(reverse("runs:stage1_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.stage2_keywords == ["5+ years Python", "Kubernetes"]   # must_have + nice_to_have


def test_seeding_does_not_overwrite_keywords_the_user_already_set(client):
    run = stage2_run(current_stage="stage1_pending", stage2_keywords=["Only this one"])
    add_candidate(run, "cv.pdf")
    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=verdict(True)):
        run_stage1(run)

    client.post(reverse("runs:stage1_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.stage2_keywords == ["Only this one"]


def test_reseeding_is_explicit_and_restores_the_jd_keywords(client, ranked_run):
    ranked_run.stage2_keywords = []
    ranked_run.save()

    client.post(reverse("runs:stage2_reseed", args=[ranked_run.pk]))

    ranked_run.refresh_from_db()
    assert ranked_run.stage2_keywords == ["5+ years Python", "Kubernetes"]


# --- The checkpoint's two controls -------------------------------------------------

def test_saving_keywords_parses_one_per_line_and_deduplicates(client, ranked_run):
    client.post(reverse("runs:stage2_settings", args=[ranked_run.pk]),
                {"keywords": "Python\n  Django  \n\npython\nSQL\n", "cutoff_percent": "25"})

    ranked_run.refresh_from_db()
    assert ranked_run.stage2_keywords == ["Python", "Django", "SQL"]


def test_saving_the_cutoff_does_not_require_a_rerun(client, ranked_run, fixed_semantics):
    fixed_semantics([0.2, 0.9, 0.5])
    run_stage2(ranked_run)
    before = list(StageResult.objects.filter(run=ranked_run, stage="stage2")
                  .values_list("pk", "output_json__rank"))

    client.post(reverse("runs:stage2_settings", args=[ranked_run.pk]),
                {"keywords": "\n".join(ranked_run.stage2_keywords), "cutoff_percent": "67"})

    ranked_run.refresh_from_db()
    after = list(StageResult.objects.filter(run=ranked_run, stage="stage2")
                 .values_list("pk", "output_json__rank"))
    assert ranked_run.stage2_cutoff_percent == 67.0
    assert before == after                                    # same rows, untouched


def test_cutoff_change_reslices_the_shortlist_immediately(client, ranked_run, fixed_semantics):
    fixed_semantics([0.2, 0.9, 0.5])
    run_stage2(ranked_run)

    client.post(reverse("runs:stage2_settings", args=[ranked_run.pk]),
                {"keywords": "", "cutoff_percent": "50"})
    context = client.get(reverse("runs:stage2_review", args=[ranked_run.pk])).context

    # ceil(3 * 0.50) = 2. Rounding UP is deliberate: at Stage 2 an extra candidate
    # costs one Stage 3 scoring call, while a dropped one is unrecoverable.
    assert context["shortlist_count"] == 2
    assert [row["shortlisted"] for row in context["rows"]] == [True, True, False]


def test_an_out_of_range_cutoff_is_rejected(client, ranked_run):
    client.post(reverse("runs:stage2_settings", args=[ranked_run.pk]),
                {"keywords": "", "cutoff_percent": "150"})

    ranked_run.refresh_from_db()
    assert ranked_run.stage2_cutoff_percent == 25.0           # unchanged default


def test_a_non_numeric_cutoff_is_rejected(client, ranked_run):
    client.post(reverse("runs:stage2_settings", args=[ranked_run.pk]),
                {"keywords": "", "cutoff_percent": "twenty"})

    ranked_run.refresh_from_db()
    assert ranked_run.stage2_cutoff_percent == 25.0


def test_editing_keywords_marks_the_existing_ranking_stale(client, ranked_run, fixed_semantics):
    """Scores on screen were computed with the old keywords — say so rather than
    letting the recruiter confirm a ranking that does not match the controls."""
    fixed_semantics([0.2, 0.9, 0.5])
    ranked_run.stage2_keywords = ["Python"]
    ranked_run.save()
    run_stage2(ranked_run)

    assert client.get(reverse("runs:stage2_review", args=[ranked_run.pk])).context["stale"] is False

    client.post(reverse("runs:stage2_settings", args=[ranked_run.pk]),
                {"keywords": "Kubernetes", "cutoff_percent": "25"})

    response = client.get(reverse("runs:stage2_review", args=[ranked_run.pk]))
    assert response.context["stale"] is True
    assert "keyword list has changed" in response.content.decode()


# --- Review checkpoint -------------------------------------------------------------

def test_review_shows_the_full_ranked_list_not_just_the_shortlist(client, ranked_run, fixed_semantics):
    fixed_semantics([0.2, 0.9, 0.5])
    run_stage2(ranked_run)

    response = client.get(reverse("runs:stage2_review", args=[ranked_run.pk]))
    html = response.content.decode()

    assert len(response.context["rows"]) == 3
    assert response.context["shortlist_count"] == 1           # 25% of 3
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        assert name in html                                   # below-cutoff candidates still listed
    assert "below cutoff" in html


def test_review_shows_both_score_components(client, ranked_run, fixed_semantics):
    fixed_semantics([0.812, 0.4, 0.6])
    run_stage2(ranked_run)

    html = client.get(reverse("runs:stage2_review", args=[ranked_run.pk])).content.decode()

    assert "0.812" in html                                    # the semantic half is visible
    assert "Semantic" in html and "Keyword" in html


def test_review_page_works_before_stage2_has_run(client, ranked_run):
    response = client.get(reverse("runs:stage2_review", args=[ranked_run.pk]))

    assert response.status_code == 200
    assert response.context["has_results"] is False


def test_start_launches_the_stage2_command_in_a_subprocess(client, ranked_run):
    with patch("runs.views.subprocess.Popen") as popen:
        client.post(reverse("runs:stage2_start", args=[ranked_run.pk]))

    assert popen.call_args.args[0][1:] == ["manage.py", "run_stage2", str(ranked_run.pk)]
    ranked_run.refresh_from_db()
    assert ranked_run.job_status == "running"


# --- Confirm gate ---------------------------------------------------------------------

def test_confirming_stage2_advances_to_stage3(client, ranked_run, fixed_semantics):
    fixed_semantics([0.2, 0.9, 0.5])
    run_stage2(ranked_run)

    client.post(reverse("runs:stage2_confirm", args=[ranked_run.pk]))

    ranked_run.refresh_from_db()
    assert ranked_run.current_stage == "stage3_pending"


def test_running_stage2_alone_does_not_advance_the_run(ranked_run, fixed_semantics):
    fixed_semantics([0.2, 0.9, 0.5])

    run_stage2(ranked_run)

    ranked_run.refresh_from_db()
    assert ranked_run.current_stage == "stage2_pending"


def test_cannot_confirm_stage2_before_it_has_run(client, ranked_run):
    client.post(reverse("runs:stage2_confirm", args=[ranked_run.pk]))

    ranked_run.refresh_from_db()
    assert ranked_run.current_stage == "stage2_pending"


# =============================================================================
# Stage 3 — scoring
# =============================================================================

INFERRED = {"requirement": "Team leadership", "description": "Implied by seniority",
            "type": "inferred", "scale": "5=led teams, 3=mentored, 1=none", "suggested_weight": 1}
# Stage 2's NICE_TO_HAVE carries weight 0.5; Stage 3 uses 3 so the normalized weights
# are 0.75 / 0.25 and every expected score below can be checked by hand.
KUBERNETES = {**NICE_TO_HAVE, "suggested_weight": 3}


def stage3_run(requirements=None, **kwargs):
    jd = JobDescription.objects.create(
        input_mode="text", raw_text="Backend engineer.",
        requirements=requirements if requirements is not None else [MUST_HAVE, KUBERNETES, INFERRED],
        confirmed_by_user=True,
    )
    defaults = {"job_description": jd, "current_stage": "stage3_pending"}
    defaults.update(kwargs)
    return ScreeningRun.objects.create(**defaults)


def rank_stage2(run, candidate, rank, final_score=0.5):
    return StageResult.objects.create(
        run=run, candidate=candidate, stage="stage2", passed=None,
        output_json={"semantic_score": final_score, "keyword_score": final_score,
                     "final_score": final_score, "rank": rank, "keywords_used": []},
    )


@pytest.fixture
def scored_run():
    """A run at stage3_pending with four Stage 2-ranked candidates (25% cutoff -> 1)."""
    run = stage3_run()
    for index, name in enumerate(("a.pdf", "b.pdf", "c.pdf", "d.pdf"), start=1):
        rank_stage2(run, add_candidate(run, name), rank=index, final_score=1.0 - index / 10)
    return run


def criterion_scores(kubernetes=4, leadership=2, confidence="high"):
    return [
        {"criterion": "Kubernetes", "score": kubernetes, "evidence_quote": "Ran clusters",
         "reasoning": "Operated Kubernetes.", "confidence": confidence},
        {"criterion": "Team leadership", "score": leadership, "evidence_quote": "Led six engineers",
         "reasoning": "Led a small team.", "confidence": confidence},
    ]


# --- Running the stage ------------------------------------------------------------

def test_stage3_scores_only_the_stage2_shortlist(scored_run):
    """Four ranked candidates, 25% cutoff -> only the top one is scored. Stage 3 is the
    expensive stage; that is the whole point of Stage 2."""
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores()}) as mock:
        summary = run_stage3(scored_run)

    assert mock.call_count == 1
    assert summary["scored"] == 1
    result = StageResult.objects.get(run=scored_run, stage="stage3")
    assert result.candidate.original_filename == "a.pdf"


def test_raising_the_cutoff_widens_who_stage3_scores(scored_run):
    scored_run.stage2_cutoff_percent = 75.0
    scored_run.save()

    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores()}) as mock:
        run_stage3(scored_run)

    assert mock.call_count == 3                       # ceil(4 * 0.75)


def test_stage3_stores_the_aggregate_and_the_weights_that_produced_it(scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores(4, 2)}):
        run_stage3(scored_run)

    output = StageResult.objects.get(run=scored_run, stage="stage3").output_json
    # suggested weights 3 and 1 normalize to 0.75 / 0.25 -> (4*.75 + 2*.25)/5*100 = 70.0
    assert output["final_score"] == 70.0
    assert output["weights_used"] == {"Kubernetes": pytest.approx(0.75),
                                      "Team leadership": pytest.approx(0.25)}
    assert len(output["criterion_breakdown"]) == 2


def test_saved_weights_are_used_instead_of_the_jd_suggestions(scored_run):
    scored_run.rubric_weights = {"Kubernetes": 0.1, "Team leadership": 0.9}
    scored_run.save()

    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores(5, 1)}):
        run_stage3(scored_run)

    output = StageResult.objects.get(run=scored_run, stage="stage3").output_json
    assert output["final_score"] == pytest.approx(28.0)     # (5*.1 + 1*.9)/5*100


def test_stage3_results_have_no_pass_fail(scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores()}):
        run_stage3(scored_run)

    assert StageResult.objects.get(run=scored_run, stage="stage3").passed is None


def test_stage3_sends_anonymized_candidates(scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores()}) as mock:
        run_stage3(scored_run)

    sent = mock.call_args.args[1]
    assert "Amina Benali" not in sent and "example.com" not in sent


def test_low_confidence_scores_are_flagged_on_the_result(scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores(confidence="low")}):
        run_stage3(scored_run)

    output = StageResult.objects.get(run=scored_run, stage="stage3").output_json
    assert output["low_confidence_flags"] == ["Kubernetes", "Team leadership"]


def test_a_mis_named_criterion_is_recorded_not_fatal(scored_run):
    bad = criterion_scores() + [{"criterion": "K8s administration", "score": 5,
                                 "evidence_quote": "q", "reasoning": "r", "confidence": "high"}]

    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": bad}):
        summary = run_stage3(scored_run)

    output = StageResult.objects.get(run=scored_run, stage="stage3").output_json
    assert output["unknown_criteria"] == ["K8s administration"]
    assert output["final_score"] == 70.0
    assert summary["anomalies"] == 1


def test_rerunning_stage3_replaces_the_previous_scores(scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores(4, 2)}):
        run_stage3(scored_run)
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores(5, 5)}):
        run_stage3(scored_run)

    results = StageResult.objects.filter(run=scored_run, stage="stage3")
    assert results.count() == 1
    assert results.first().output_json["final_score"] == 100.0


# --- The empty-rubric edge case ------------------------------------------------------

def test_a_must_have_only_jd_blocks_stage3_with_a_skip_instruction():
    run = stage3_run(requirements=[MUST_HAVE])
    rank_stage2(run, add_candidate(run, "cv.pdf"), rank=1)

    with pytest.raises(StageNotReady, match="nothing .* to score|Skip Stage 3"):
        run_stage3(run)


def test_skipping_stage3_advances_to_final_review(client):
    """The specified handling: Stage 2's ranking stands as the final order."""
    run = stage3_run(requirements=[MUST_HAVE])
    rank_stage2(run, add_candidate(run, "cv.pdf"), rank=1)

    client.post(reverse("runs:stage3_skip", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "review"
    assert StageResult.objects.filter(run=run, stage="stage3").count() == 0


def test_the_stage2_ranking_survives_the_skip(client):
    run = stage3_run(requirements=[MUST_HAVE])
    first = add_candidate(run, "first.pdf")
    second = add_candidate(run, "second.pdf")
    rank_stage2(run, first, rank=1, final_score=0.9)
    rank_stage2(run, second, rank=2, final_score=0.4)

    client.post(reverse("runs:stage3_skip", args=[run.pk]))

    ranking = list(StageResult.objects.filter(run=run, stage="stage2")
                   .order_by("output_json__rank").values_list("candidate__original_filename", flat=True))
    assert ranking == ["first.pdf", "second.pdf"]


def test_a_jd_with_criteria_cannot_be_skipped(client, scored_run):
    """Skipping is only for the empty-rubric case — it must not become a way to
    bypass scoring on a JD that has scorable criteria."""
    client.post(reverse("runs:stage3_skip", args=[scored_run.pk]))

    scored_run.refresh_from_db()
    assert scored_run.current_stage == "stage3_pending"


def test_the_empty_rubric_page_offers_the_skip_and_hides_the_weight_editor(client):
    run = stage3_run(requirements=[MUST_HAVE])
    rank_stage2(run, add_candidate(run, "cv.pdf"), rank=1)

    response = client.get(reverse("runs:stage3_review", args=[run.pk]))
    html = response.content.decode()

    assert response.context["rubric_empty"] is True
    assert "no nice-to-have or inferred requirements" in html
    assert "Skip Stage 3" in html
    assert "Criterion weights" not in html


# --- Preconditions ----------------------------------------------------------------------

def test_stage3_refuses_at_the_wrong_stage(scored_run):
    scored_run.current_stage = "stage2_pending"
    scored_run.save()

    with pytest.raises(StageNotReady, match="not ready for Stage 3"):
        run_stage3(scored_run)


def test_stage3_refuses_when_nobody_is_above_the_cutoff():
    run = stage3_run()

    with pytest.raises(StageNotReady, match="above the Stage 2 cutoff"):
        run_stage3(run)


def test_stage3_command_records_failure_rather_than_hanging(scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               side_effect=RuntimeError("API down")):
        with pytest.raises(RuntimeError):
            call_command("run_stage3", scored_run.pk)

    scored_run.refresh_from_db()
    assert scored_run.job_status == "failed"
    assert "API down" in scored_run.job_message


# --- The weight-adjustment checkpoint -------------------------------------------------------

def test_the_checkpoint_prefills_the_jd_suggested_weights(client, scored_run):
    context = client.get(reverse("runs:stage3_review", args=[scored_run.pk])).context

    assert [c["requirement"] for c in context["rubric"]] == ["Kubernetes", "Team leadership"]
    assert [c["weight"] for c in context["rubric"]] == pytest.approx([0.75, 0.25])


def test_must_have_requirements_are_absent_from_the_checkpoint(client, scored_run):
    context = client.get(reverse("runs:stage3_review", args=[scored_run.pk])).context

    assert "5+ years Python" not in [c["requirement"] for c in context["rubric"]]


def test_saving_weights_stores_them_on_the_run(client, scored_run):
    client.post(reverse("runs:stage3_weights", args=[scored_run.pk]),
                {"weight::Kubernetes": "0.6", "weight::Team leadership": "0.4"})

    scored_run.refresh_from_db()
    assert scored_run.rubric_weights == {"Kubernetes": 0.6, "Team leadership": 0.4}


def test_weights_that_do_not_sum_to_one_are_flagged_as_normalized(client, scored_run):
    client.post(reverse("runs:stage3_weights", args=[scored_run.pk]),
                {"weight::Kubernetes": "8", "weight::Team leadership": "2"})

    response = client.get(reverse("runs:stage3_review", args=[scored_run.pk]))

    assert response.context["normalized"] is True
    assert "normalized to 1.0" in response.content.decode()
    assert [c["weight"] for c in response.context["rubric"]] == pytest.approx([0.8, 0.2])


def test_a_negative_weight_is_rejected(client, scored_run):
    client.post(reverse("runs:stage3_weights", args=[scored_run.pk]),
                {"weight::Kubernetes": "-1", "weight::Team leadership": "0.4"})

    scored_run.refresh_from_db()
    assert scored_run.rubric_weights == {}


def test_a_non_numeric_weight_is_rejected(client, scored_run):
    client.post(reverse("runs:stage3_weights", args=[scored_run.pk]),
                {"weight::Kubernetes": "heavy", "weight::Team leadership": "0.4"})

    scored_run.refresh_from_db()
    assert scored_run.rubric_weights == {}


def test_zeroing_every_weight_warns_and_scores_equally(client, scored_run):
    client.post(reverse("runs:stage3_weights", args=[scored_run.pk]),
                {"weight::Kubernetes": "0", "weight::Team leadership": "0"})
    scored_run.refresh_from_db()

    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores(5, 1)}):
        run_stage3(scored_run)

    output = StageResult.objects.get(run=scored_run, stage="stage3").output_json
    assert output["weights_used"] == {"Kubernetes": pytest.approx(0.5),
                                      "Team leadership": pytest.approx(0.5)}
    assert output["final_score"] == pytest.approx(60.0)      # (5*.5 + 1*.5)/5*100


def test_resetting_weights_restores_the_suggestions(client, scored_run):
    scored_run.rubric_weights = {"Kubernetes": 0.99}
    scored_run.save()

    client.post(reverse("runs:stage3_weights_reset", args=[scored_run.pk]))

    scored_run.refresh_from_db()
    assert scored_run.rubric_weights == {}


# --- Results display and confirm gate ----------------------------------------------------------

def test_review_shows_every_criterion_score_with_its_evidence(client, scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores()}):
        run_stage3(scored_run)

    html = client.get(reverse("runs:stage3_review", args=[scored_run.pk])).content.decode()

    assert "70.0" in html
    assert "Ran clusters" in html and "Led six engineers" in html      # evidence quotes
    assert "Operated Kubernetes." in html                               # reasoning


def test_review_flags_low_confidence_scores_visually(client, scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores(confidence="low")}):
        run_stage3(scored_run)

    response = client.get(reverse("runs:stage3_review", args=[scored_run.pk]))

    assert response.context["low_confidence_total"] == 2
    assert "low-confidence" in response.content.decode()


def test_review_orders_candidates_by_final_score(client, scored_run):
    scored_run.stage2_cutoff_percent = 100.0
    scored_run.save()
    payloads = [
        {"criterion_scores": criterion_scores(1, 1)},
        {"criterion_scores": criterion_scores(5, 5)},
        {"criterion_scores": criterion_scores(3, 3)},
        {"criterion_scores": criterion_scores(2, 2)},
    ]
    with patch("screening.stage3_scoring.scoring.call_structured", side_effect=payloads):
        run_stage3(scored_run)

    context = client.get(reverse("runs:stage3_review", args=[scored_run.pk])).context

    assert [r.output_json["final_score"] for r in context["results"]] == [100.0, 60.0, 40.0, 20.0]


def test_confirming_stage3_advances_to_final_review(client, scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores()}):
        run_stage3(scored_run)

    client.post(reverse("runs:stage3_confirm", args=[scored_run.pk]))

    scored_run.refresh_from_db()
    assert scored_run.current_stage == "review"


def test_running_stage3_alone_does_not_advance_the_run(scored_run):
    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": criterion_scores()}):
        run_stage3(scored_run)

    scored_run.refresh_from_db()
    assert scored_run.current_stage == "stage3_pending"


def test_cannot_confirm_stage3_before_it_has_run(client, scored_run):
    client.post(reverse("runs:stage3_confirm", args=[scored_run.pk]))

    scored_run.refresh_from_db()
    assert scored_run.current_stage == "stage3_pending"


def test_start_launches_the_stage3_command_in_a_subprocess(client, scored_run):
    with patch("runs.views.subprocess.Popen") as popen:
        client.post(reverse("runs:stage3_start", args=[scored_run.pk]))

    assert popen.call_args.args[0][1:] == ["manage.py", "run_stage3", str(scored_run.pk)]
    scored_run.refresh_from_db()
    assert scored_run.job_status == "running"


# =============================================================================
# Final Review — history, overrides, CSV export
# =============================================================================

def review_run(**kwargs):
    jd = JobDescription.objects.create(
        input_mode="text", raw_text="Backend engineer.",
        requirements=[MUST_HAVE, KUBERNETES, INFERRED], confirmed_by_user=True,
    )
    defaults = {"job_description": jd, "current_stage": "review"}
    defaults.update(kwargs)
    return ScreeningRun.objects.create(**defaults)


def score_stage3(run, candidate, final_score=70.0, low_confidence=()):
    return StageResult.objects.create(
        run=run, candidate=candidate, stage="stage3", passed=None,
        output_json={
            "final_score": final_score,
            "criterion_breakdown": [
                {"criterion": "Kubernetes", "score": 4, "evidence_quote": "Ran clusters",
                 "reasoning": "Operated Kubernetes.", "confidence": "high"},
            ],
            "low_confidence_flags": list(low_confidence),
            "unscored_criteria": [], "unknown_criteria": [],
            "weights_used": {"Kubernetes": 0.75, "Team leadership": 0.25},
        },
    )


def fail_stage1(run, candidate):
    return StageResult.objects.create(
        run=run, candidate=candidate, stage="stage1", passed=False,
        output_json={"evaluations": [{"requirement": "5+ years Python", "met": False,
                                      "evidence": "", "reason": "No Python experience stated."}]},
    )


@pytest.fixture
def recruiter(django_user_model):
    return django_user_model.objects.create_user(username="hr_amine", password="pw12345")


@pytest.fixture
def full_run():
    """A run covering every possible fate: scored, below cutoff, Stage 1 failure,
    and a CV rejected at upload that never entered the pipeline."""
    run = review_run(stage2_cutoff_percent=50.0)

    top = add_candidate(run, "top.pdf")
    middle = add_candidate(run, "middle.pdf", name="Youssef El Amrani")
    filtered = add_candidate(run, "filtered.pdf", name="Sofia Marchetti")
    unreadable = add_candidate(run, "scan.pdf", status="rejected")
    unreadable.rejection_reason = "no extractable text found"
    unreadable.save()

    for candidate in (top, middle, filtered):
        StageResult.objects.create(
            run=run, candidate=candidate, stage="stage1",
            passed=candidate is not filtered,
            output_json={"evaluations": [{
                "requirement": "5+ years Python",
                "met": candidate is not filtered,
                "evidence": "Built Python ETL pipelines" if candidate is not filtered else "",
                "reason": "6 years stated." if candidate is not filtered else "No Python experience stated.",
            }]},
        )
    rank_stage2(run, top, rank=1, final_score=0.9)
    rank_stage2(run, middle, rank=2, final_score=0.4)
    score_stage3(run, top, final_score=82.0)

    return {"run": run, "top": top, "middle": middle,
            "filtered": filtered, "unreadable": unreadable}


# --- Every candidate appears, not just survivors ---------------------------------

def test_history_includes_every_candidate_whatever_their_fate(full_run):
    """GDPR Art. 22: a filtered-out CV stays visible. A decision nobody can see is a
    decision nobody can review."""
    rows = candidate_history(full_run["run"])

    assert {r["candidate"].original_filename for r in rows} == {
        "top.pdf", "middle.pdf", "filtered.pdf", "scan.pdf",
    }


def test_history_ranks_scored_candidates_first(full_run):
    rows = candidate_history(full_run["run"])

    assert rows[0]["candidate"].original_filename == "top.pdf"
    assert rows[0]["effective_score"] == 82.0
    assert [r["position"] for r in rows] == [1, 2, 3, 4]


def test_history_carries_each_stage_reason_through(full_run):
    rows = {r["candidate"].original_filename: r for r in candidate_history(full_run["run"])}

    assert rows["filtered.pdf"]["stage1"].passed is False
    assert rows["filtered.pdf"]["stage1"].evaluations[0]["reason"] == "No Python experience stated."
    assert rows["middle.pdf"]["stage2_rank"] == 2
    assert rows["scan.pdf"]["candidate"].rejection_reason == "no extractable text found"


def test_a_candidate_rejected_at_upload_has_no_stage_results(full_run):
    rows = {r["candidate"].original_filename: r for r in candidate_history(full_run["run"])}

    scan = rows["scan.pdf"]
    assert scan["stage1"] is None and scan["stage2"] is None and scan["stage3"] is None
    assert scan["effective_score"] is None


def test_low_confidence_flags_are_surfaced_on_the_row(full_run):
    score_stage3(full_run["run"], full_run["middle"], final_score=55.0,
                 low_confidence=["Team leadership"])

    rows = {r["candidate"].original_filename: r for r in candidate_history(full_run["run"])}

    assert rows["middle.pdf"]["low_confidence_flags"] == ["Team leadership"]


def test_final_review_page_shows_survivors_and_filtered_alike(client, full_run):
    html = client.get(reverse("runs:final_review", args=[full_run["run"].pk])).content.decode()

    for name in ("top.pdf", "middle.pdf", "filtered.pdf", "scan.pdf"):
        assert name in html
    assert "No Python experience stated." in html          # the Stage 1 failure reason
    assert "no extractable text found" in html             # the upload rejection reason
    assert "Ran clusters" in html                          # Stage 3 evidence


# --- The override flow ------------------------------------------------------------

def test_an_override_requires_a_signed_in_user(client, full_run):
    """performed_by is non-null: an override with no named author is not an audit record."""
    response = client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                           {"candidate_id": full_run["filtered"].pk, "action": "reinstate",
                            "reason": "Equivalent experience in Go."})

    assert response.status_code == 302
    assert "/admin/login/" in response.url
    assert Override.objects.count() == 0


def test_an_override_requires_a_reason(client, full_run, recruiter):
    client.force_login(recruiter)

    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["filtered"].pk, "action": "reinstate", "reason": "   "})

    assert Override.objects.count() == 0


def test_reinstating_logs_the_action_reason_and_user(client, full_run, recruiter):
    client.force_login(recruiter)

    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["filtered"].pk, "action": "reinstate",
                 "reason": "Ten years of Go maps to the Python requirement."})

    override = Override.objects.get()
    assert override.action == "reinstate"
    assert override.candidate == full_run["filtered"]
    assert override.performed_by == recruiter
    assert override.reason == "Ten years of Go maps to the Python requirement."


def test_an_override_never_edits_the_stage_result(client, full_run, recruiter):
    """StageResult is the automated trail — nobody edits it after the fact. The
    effective outcome is StageResult + Override, computed on read."""
    client.force_login(recruiter)
    before = StageResult.objects.get(run=full_run["run"], candidate=full_run["filtered"], stage="stage1")

    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["filtered"].pk, "action": "reinstate",
                 "reason": "Equivalent experience."})

    after = StageResult.objects.get(pk=before.pk)
    assert after.passed is False                              # unchanged
    assert after.output_json == before.output_json


def test_reinstating_promotes_the_candidate_above_the_filtered_group(client, full_run, recruiter):
    client.force_login(recruiter)
    before = [r["candidate"].original_filename for r in candidate_history(full_run["run"])]
    assert before.index("filtered.pdf") > before.index("middle.pdf")

    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["filtered"].pk, "action": "reinstate",
                 "reason": "Equivalent experience."})

    rows = candidate_history(full_run["run"])
    names = [r["candidate"].original_filename for r in rows]
    assert names.index("filtered.pdf") < names.index("middle.pdf")
    reinstated = next(r for r in rows if r["candidate"].original_filename == "filtered.pdf")
    assert reinstated["reinstated"] is True
    assert "Reinstated by a recruiter" in reinstated["outcome"]


def test_adjusting_a_score_replaces_the_effective_score_but_keeps_the_automated_one(
        client, full_run, recruiter):
    client.force_login(recruiter)

    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["top"].pk, "action": "adjust_score",
                 "adjusted_score": "55", "reason": "Overstated leadership evidence."})

    row = next(r for r in candidate_history(full_run["run"])
               if r["candidate"].original_filename == "top.pdf")
    assert row["effective_score"] == 55.0
    assert row["automated_score"] == 82.0                     # what the pipeline said, preserved
    assert row["score_adjusted"] is True


def test_the_most_recent_score_adjustment_wins(client, full_run, recruiter):
    client.force_login(recruiter)
    url = reverse("runs:create_override", args=[full_run["run"].pk])

    client.post(url, {"candidate_id": full_run["top"].pk, "action": "adjust_score",
                      "adjusted_score": "55", "reason": "First pass."})
    client.post(url, {"candidate_id": full_run["top"].pk, "action": "adjust_score",
                      "adjusted_score": "64", "reason": "Reconsidered after a call."})

    row = next(r for r in candidate_history(full_run["run"])
               if r["candidate"].original_filename == "top.pdf")
    assert row["effective_score"] == 64.0
    assert len(row["overrides"]) == 2                         # both kept in the log


def test_an_out_of_range_adjusted_score_is_rejected(client, full_run, recruiter):
    client.force_login(recruiter)

    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["top"].pk, "action": "adjust_score",
                 "adjusted_score": "150", "reason": "Too good."})

    assert Override.objects.count() == 0


def test_a_non_numeric_adjusted_score_is_rejected(client, full_run, recruiter):
    client.force_login(recruiter)

    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["top"].pk, "action": "adjust_score",
                 "adjusted_score": "excellent", "reason": "Too good."})

    assert Override.objects.count() == 0


def test_an_unknown_action_is_rejected(client, full_run, recruiter):
    client.force_login(recruiter)

    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["top"].pk, "action": "delete_candidate",
                 "reason": "Not a real action."})

    assert Override.objects.count() == 0


def test_a_note_override_records_context_without_changing_anything(client, full_run, recruiter):
    client.force_login(recruiter)

    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["top"].pk, "action": "note",
                 "reason": "Referred by the engineering manager."})

    row = next(r for r in candidate_history(full_run["run"])
               if r["candidate"].original_filename == "top.pdf")
    assert row["effective_score"] == 82.0                     # untouched
    assert row["reinstated"] is False
    assert row["overrides"][0].action == "note"


def test_an_override_cannot_target_a_candidate_from_another_run(client, full_run, recruiter):
    client.force_login(recruiter)
    other = review_run()
    outsider = add_candidate(other, "outsider.pdf")

    response = client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                           {"candidate_id": outsider.pk, "action": "note", "reason": "Wrong run."})

    assert response.status_code == 404
    assert Override.objects.count() == 0


def test_overrides_appear_on_the_review_page_with_their_author(client, full_run, recruiter):
    client.force_login(recruiter)
    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["filtered"].pk, "action": "reinstate",
                 "reason": "Equivalent experience in Go."})

    html = client.get(reverse("runs:final_review", args=[full_run["run"].pk])).content.decode()

    assert "Equivalent experience in Go." in html
    assert "hr_amine" in html
    assert "reinstated" in html


# --- CSV export -------------------------------------------------------------------------

def read_csv(client, run):
    response = client.get(reverse("runs:export_csv", args=[run.pk]))
    assert response.status_code == 200
    return response, list(csv.DictReader(io.StringIO(response.content.decode())))


def test_csv_is_served_as_a_download(client, full_run):
    response, _ = read_csv(client, full_run["run"])

    assert response["Content-Type"] == "text/csv"
    assert f'filename="run-{full_run["run"].pk}-final-review.csv"' in response["Content-Disposition"]


def test_csv_exports_every_candidate_not_just_the_shortlist(client, full_run):
    _, rows = read_csv(client, full_run["run"])

    assert [r["filename"] for r in rows] == ["top.pdf", "middle.pdf", "filtered.pdf", "scan.pdf"]
    assert [r["position"] for r in rows] == ["1", "2", "3", "4"]


def test_csv_carries_the_rationale_for_each_stage(client, full_run):
    _, rows = read_csv(client, full_run["run"])
    by_name = {r["filename"]: r for r in rows}

    assert "No Python experience stated." in by_name["filtered.pdf"]["stage1_reasons"]
    assert by_name["filtered.pdf"]["stage1_unmet"] == "5+ years Python"
    assert by_name["filtered.pdf"]["stage1_passed"] == "fail"
    assert by_name["middle.pdf"]["stage2_rank"] == "2"
    assert "Kubernetes: 4/5 (high) — Ran clusters" in by_name["top.pdf"]["stage3_criteria"]
    assert by_name["scan.pdf"]["rejection_reason"] == "no extractable text found"


def test_csv_includes_the_candidate_name_for_the_recruiter(client, full_run):
    """Anonymization protects the LLM stages, not the recruiter's own review — a
    shortlist without names is unusable. The file is personal data."""
    _, rows = read_csv(client, full_run["run"])

    assert rows[0]["candidate_name"] == "Amina Benali"
    assert rows[0]["email"] == "amina@example.com"


def test_csv_shows_both_the_automated_and_the_overridden_score(client, full_run, recruiter):
    client.force_login(recruiter)
    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["top"].pk, "action": "adjust_score",
                 "adjusted_score": "55", "reason": "Overstated leadership evidence."})

    _, rows = read_csv(client, full_run["run"])

    assert rows[0]["final_score"] == "55.0"
    assert rows[0]["automated_score"] == "82.0"
    assert rows[0]["score_adjusted"] == "yes"
    assert "Overstated leadership evidence." in rows[0]["overrides"]
    assert "hr_amine" in rows[0]["overrides"]


def test_csv_records_low_confidence_flags(client, full_run):
    score_stage3(full_run["run"], full_run["middle"], final_score=55.0,
                 low_confidence=["Team leadership"])

    _, rows = read_csv(client, full_run["run"])
    by_name = {r["filename"]: r for r in rows}

    assert by_name["middle.pdf"]["low_confidence_flags"] == "Team leadership"


def test_csv_survives_a_run_where_nothing_has_been_scored(client):
    """A run abandoned after upload still exports — the record of what happened
    should not depend on the pipeline having finished."""
    run = review_run()
    add_candidate(run, "only.pdf")

    _, rows = read_csv(client, run)

    assert len(rows) == 1
    assert rows[0]["final_score"] == ""
    assert rows[0]["stage1_passed"] == ""


def test_csv_quotes_reasons_containing_commas(client, full_run, recruiter):
    """A reason is free text; an unescaped comma would silently shift every later
    column and corrupt the audit record."""
    client.force_login(recruiter)
    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["top"].pk, "action": "note",
                 "reason": "Strong, but expensive, and remote-only."})

    _, rows = read_csv(client, full_run["run"])

    assert "Strong, but expensive, and remote-only." in rows[0]["overrides"]
    assert rows[0]["filename"] == "top.pdf"                   # columns still aligned


# =============================================================================
# Excel summary export
# =============================================================================

def open_xlsx(client, run):
    """Download the workbook and load it back, as Excel would."""
    response = client.get(reverse("runs:export_xlsx", args=[run.pk]))
    assert response.status_code == 200
    return response, load_workbook(io.BytesIO(response.content))


def sheet_rows(sheet, header_row=4):
    """Data rows as dicts keyed by column title."""
    titles = [c.value for c in sheet[header_row]]
    out = []
    for row in sheet.iter_rows(min_row=header_row + 1, values_only=True):
        if row[1] is None:
            continue
        out.append(dict(zip(titles, row)))
    return out


def test_xlsx_is_served_as_a_spreadsheet_download(client, full_run):
    response, _ = open_xlsx(client, full_run["run"])

    assert response["Content-Type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert f'filename="run-{full_run["run"].pk}-summary.xlsx"' in response["Content-Disposition"]


def test_xlsx_has_both_sheets(client, full_run):
    _, book = open_xlsx(client, full_run["run"])

    assert book.sheetnames == ["Candidates", "Run summary"]


def test_every_candidate_gets_a_row_including_those_who_never_ran(client, full_run):
    """The point of the sheet: nobody disappears. A CV rejected at upload is listed
    beside the top-scored candidate."""
    _, book = open_xlsx(client, full_run["run"])

    rows = sheet_rows(book["Candidates"])
    assert [r["CV file"] for r in rows] == ["top.pdf", "middle.pdf", "filtered.pdf", "scan.pdf"]


def test_a_stage1_failure_shows_failed_and_a_brief_reason(client, full_run):
    _, book = open_xlsx(client, full_run["run"])
    rows = {r["CV file"]: r for r in sheet_rows(book["Candidates"])}

    failed = rows["filtered.pdf"]
    assert failed["Stage 1 — hard filter"] == "failed"
    assert "5+ years Python" in failed["Why they stopped"]
    assert "No Python experience stated." in failed["Why they stopped"]


def test_a_stage1_pass_shows_passed_with_no_failure_reason(client, full_run):
    _, book = open_xlsx(client, full_run["run"])
    rows = {r["CV file"]: r for r in sheet_rows(book["Candidates"])}

    assert rows["top.pdf"]["Stage 1 — hard filter"] == "passed"
    assert not rows["top.pdf"]["Why they stopped"]


def test_a_candidate_rejected_at_upload_shows_that_reason(client, full_run):
    """They never reached Stage 1, so the reason comes from preprocessing instead."""
    _, book = open_xlsx(client, full_run["run"])
    rows = {r["CV file"]: r for r in sheet_rows(book["Candidates"])}

    scan = rows["scan.pdf"]
    assert scan["Preprocessing"] == "rejected"
    assert scan["Stage 1 — hard filter"] == "not reached"
    assert scan["Why they stopped"] == "no extractable text found"


def test_only_candidates_who_reached_stage3_carry_a_score(client, full_run):
    _, book = open_xlsx(client, full_run["run"])
    rows = {r["CV file"]: r for r in sheet_rows(book["Candidates"])}

    assert rows["top.pdf"]["Stage 3 score"] == 82.0
    for name in ("middle.pdf", "filtered.pdf", "scan.pdf"):
        assert rows[name]["Stage 3 score"] in (None, "")


def test_the_score_is_a_number_excel_can_sort_not_text(client, full_run):
    """A score stored as text sorts 100 before 9 and breaks every filter."""
    _, book = open_xlsx(client, full_run["run"])
    sheet = book["Candidates"]

    score_cell = sheet.cell(row=5, column=10)          # first data row, Stage 3 score
    assert isinstance(score_cell.value, (int, float))
    assert score_cell.number_format == "0.0"


def test_stage2_rank_and_shortlist_status_are_shown(client, full_run):
    _, book = open_xlsx(client, full_run["run"])
    rows = {r["CV file"]: r for r in sheet_rows(book["Candidates"])}

    assert rows["top.pdf"]["Stage 2 rank"] == 1
    assert rows["top.pdf"]["Stage 2 — ranking"] == "shortlisted"
    assert rows["middle.pdf"]["Stage 2 — ranking"] == "below cutoff"
    assert rows["filtered.pdf"]["Stage 2 — ranking"] == "not reached"


def test_low_confidence_criteria_are_listed(client, full_run):
    score_stage3(full_run["run"], full_run["middle"], final_score=55.0,
                 low_confidence=["Team leadership"])

    _, book = open_xlsx(client, full_run["run"])
    rows = {r["CV file"]: r for r in sheet_rows(book["Candidates"])}

    assert rows["middle.pdf"]["Low-confidence criteria"] == "Team leadership"


def test_an_override_appears_with_its_author_and_reason(client, full_run, recruiter):
    client.force_login(recruiter)
    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["filtered"].pk, "action": "reinstate",
                 "reason": "Ten years of Go, equivalent."})

    _, book = open_xlsx(client, full_run["run"])
    rows = {r["CV file"]: r for r in sheet_rows(book["Candidates"])}

    cell = rows["filtered.pdf"]["Recruiter overrides"]
    assert "Reinstate" in cell and "hr_amine" in cell and "Ten years of Go" in cell


def test_an_adjusted_score_is_the_one_shown(client, full_run, recruiter):
    client.force_login(recruiter)
    client.post(reverse("runs:create_override", args=[full_run["run"].pk]),
                {"candidate_id": full_run["top"].pk, "action": "adjust_score",
                 "adjusted_score": "55", "reason": "Overstated evidence."})

    _, book = open_xlsx(client, full_run["run"])
    rows = {r["CV file"]: r for r in sheet_rows(book["Candidates"])}

    assert rows["top.pdf"]["Stage 3 score"] == 55.0
    assert "82" in rows["top.pdf"]["Recruiter overrides"] or "adjust" in rows["top.pdf"]["Recruiter overrides"].lower()


# --- The run summary sheet ---------------------------------------------------------

def summary_facts(book):
    return {row[0]: row[1] for row in book["Run summary"].iter_rows(values_only=True)
            if row[0] and row[1] is not None}


def test_the_summary_sheet_counts_each_stage(client, full_run):
    _, book = open_xlsx(client, full_run["run"])
    facts = summary_facts(book)

    assert facts["Candidates uploaded"] == 4
    assert facts["Passed Stage 1"] == 2
    assert facts["Filtered out at Stage 1"] == 1
    assert facts["Scored at Stage 3"] == 1


def test_the_summary_sheet_records_the_settings_the_stages_ran_with(client, full_run):
    """Without these, the scores on sheet 1 cannot be interpreted later."""
    full_run["run"].stage2_keywords = ["Python", "Django"]
    full_run["run"].save()

    _, book = open_xlsx(client, full_run["run"])
    facts = summary_facts(book)

    assert "50%" in str(facts["Shortlist cutoff"])
    assert facts["Stage 2 keywords"] == "Python, Django"


def test_the_summary_sheet_lists_the_job_requirements_with_their_types(client, full_run):
    _, book = open_xlsx(client, full_run["run"])
    values = [cell for row in book["Run summary"].iter_rows(values_only=True)
              for cell in row if cell]

    assert "5+ years Python" in values
    assert "must_have" in values
    assert "nice_to_have" in values


# --- Robustness --------------------------------------------------------------------

def test_the_export_works_on_a_run_where_nothing_has_run_yet(client):
    """A run abandoned after upload still exports — the record should not depend on
    the pipeline having finished."""
    run = review_run()
    add_candidate(run, "only.pdf")

    _, book = open_xlsx(client, run)
    rows = sheet_rows(book["Candidates"])

    assert len(rows) == 1
    assert rows[0]["Stage 1 — hard filter"] == "not reached"
    assert rows[0]["Stage 3 score"] in (None, "")


def test_the_export_works_on_a_run_with_no_candidates(client):
    run = review_run()

    _, book = open_xlsx(client, run)

    assert sheet_rows(book["Candidates"]) == []
    assert summary_facts(book)["Candidates uploaded"] == 0


def test_many_unmet_requirements_are_truncated_not_dumped(client):
    """A candidate failing ten must-haves should not produce an unreadable cell."""
    run = review_run()
    candidate = add_candidate(run, "many.pdf")
    StageResult.objects.create(
        run=run, candidate=candidate, stage="stage1", passed=False,
        output_json={"evaluations": [
            {"requirement": f"Requirement {i}", "met": False, "evidence": "",
             "reason": f"Reason {i}."} for i in range(10)]})

    _, book = open_xlsx(client, run)
    cell = sheet_rows(book["Candidates"])[0]["Why they stopped"]

    assert "Requirement 0" in cell
    assert "+7 more" in cell
    assert "Requirement 9" not in cell


def test_the_sheet_is_filterable_and_the_header_stays_visible(client, full_run):
    """Small things, but they are what make a 60-row sheet usable."""
    _, book = open_xlsx(client, full_run["run"])
    sheet = book["Candidates"]

    assert sheet.freeze_panes == "A5"
    assert sheet.auto_filter.ref == "A4:L8"
