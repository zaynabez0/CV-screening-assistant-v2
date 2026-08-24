"""Django-side Stage 1 tests: the job orchestration, the review page, the confirm gate.

The LLM is mocked throughout. The subprocess is mocked in the trigger-view tests and
bypassed in the work tests, which call the service directly — that way the real
evaluation logic is exercised without spawning a process the test runner can't see.
"""

import json
from unittest.mock import patch

import pytest
from django.core.management import CommandError, call_command
from django.urls import reverse

from candidates.models import Candidate
from jobs.models import JobDescription
from runs.models import ScreeningRun, StageResult
from runs.services import StageNotReady, run_stage1, run_stage2

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
