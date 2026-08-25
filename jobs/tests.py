"""JD Processing views: run creation, extraction, editing, and the confirm transition.

The LLM is mocked throughout. The transition these tests exist for is
`jd_pending -> cv_pending`, which had no code path at all before this app was built.
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from candidates.models import Candidate
from jobs.models import JobDescription
from runs.models import ScreeningRun

FIXTURES = Path(__file__).resolve().parent.parent / "screening" / "tests" / "fixtures"

pytestmark = pytest.mark.django_db

JD_TEXT = "Backend Engineer. Required: 5+ years Python. Django a plus."


@pytest.fixture(autouse=True)
def media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "media"
    return settings.MEDIA_ROOT


def extracted():
    return [
        {"requirement": "5+ years Python", "description": "Required",
         "type": "must_have", "scale": "", "suggested_weight": 0},
        {"requirement": "Django", "description": "A plus", "type": "nice_to_have",
         "scale": "5=deep, 3=some, 1=none", "suggested_weight": 3},
    ]


def make_run(requirements=None, confirmed=False, stage="jd_pending"):
    jd = JobDescription.objects.create(
        input_mode="text", raw_text=JD_TEXT,
        requirements=requirements if requirements is not None else [],
        confirmed_by_user=confirmed)
    return ScreeningRun.objects.create(job_description=jd, current_stage=stage)


def upload(name):
    return SimpleUploadedFile(name, (FIXTURES / name).read_bytes())


def flat(response):
    """HTML with runs of whitespace collapsed, so assertions survive line wrapping."""
    return re.sub(r"\s+", " ", response.content.decode())


# --- Creating a run (the application's entry point) ----------------------------

def test_creating_a_run_from_pasted_text(client):
    response = client.post(reverse("jobs:run_create"),
                           {"input_mode": "text", "raw_text": JD_TEXT})

    run = ScreeningRun.objects.get()
    assert response.status_code == 302
    assert run.current_stage == "jd_pending"           # nothing is skipped
    assert run.job_description.raw_text == JD_TEXT
    assert run.job_description.confirmed_by_user is False
    assert run.job_description.requirements == []      # creation does not extract


def test_creating_a_run_from_an_uploaded_file(client):
    client.post(reverse("jobs:run_create"),
                {"input_mode": "upload", "raw_file": upload("clean_cv.pdf")})

    jd = JobDescription.objects.get()
    assert jd.input_mode == "upload"
    assert jd.raw_file.name.endswith(".pdf")


def test_creation_does_not_call_the_llm(client):
    """Every stage is user-triggered — creating a run must not start extraction."""
    with patch("jobs.views.extract_requirements") as mock:
        client.post(reverse("jobs:run_create"), {"input_mode": "text", "raw_text": JD_TEXT})

    mock.assert_not_called()


def test_a_scanned_job_description_is_rejected_and_no_run_is_left_behind(client):
    client.post(reverse("jobs:run_create"),
                {"input_mode": "upload", "raw_file": upload("no_text_layer.pdf")})

    assert ScreeningRun.objects.count() == 0
    assert JobDescription.objects.count() == 0


def test_a_corrupted_job_description_is_rejected(client):
    client.post(reverse("jobs:run_create"),
                {"input_mode": "upload", "raw_file": upload("corrupted.pdf")})

    assert ScreeningRun.objects.count() == 0


def test_a_docx_renamed_to_pdf_is_caught_by_the_magic_check(client):
    client.post(reverse("jobs:run_create"),
                {"input_mode": "upload", "raw_file": upload("disguised.pdf")})

    assert ScreeningRun.objects.count() == 0


def test_an_unsupported_extension_is_rejected(client):
    client.post(reverse("jobs:run_create"),
                {"input_mode": "upload", "raw_file": upload("unsupported.txt")})

    assert ScreeningRun.objects.count() == 0


def test_empty_text_is_rejected(client):
    client.post(reverse("jobs:run_create"), {"input_mode": "text", "raw_text": "   "})

    assert ScreeningRun.objects.count() == 0


# --- Extraction ------------------------------------------------------------------

def test_extraction_stores_the_requirements(client):
    run = make_run()

    with patch("jobs.views.extract_requirements", return_value=extracted()) as mock:
        client.post(reverse("jobs:jd_extract", args=[run.pk]))

    mock.assert_called_once()
    run.job_description.refresh_from_db()
    assert len(run.job_description.requirements) == 2


def test_extraction_receives_the_job_description_text(client):
    run = make_run()

    with patch("jobs.views.extract_requirements", return_value=extracted()) as mock:
        client.post(reverse("jobs:jd_extract", args=[run.pk]))

    assert mock.call_args.args[0] == JD_TEXT


def test_extraction_does_not_advance_the_run(client):
    """No stage auto-runs the next: only confirming moves current_stage."""
    run = make_run()

    with patch("jobs.views.extract_requirements", return_value=extracted()):
        client.post(reverse("jobs:jd_extract", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "jd_pending"
    assert run.job_description.confirmed_by_user is False


def test_re_extraction_replaces_the_list_rather_than_appending(client):
    run = make_run(requirements=extracted())

    with patch("jobs.views.extract_requirements", return_value=extracted()[:1]):
        client.post(reverse("jobs:jd_extract", args=[run.pk]))

    run.job_description.refresh_from_db()
    assert len(run.job_description.requirements) == 1


def test_an_extraction_failure_leaves_the_run_usable(client):
    run = make_run()

    with patch("jobs.views.extract_requirements", side_effect=RuntimeError("API down")):
        client.post(reverse("jobs:jd_extract", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "jd_pending"
    assert run.job_description.requirements == []


def test_extraction_normalizes_must_have_rows(client):
    """The model may return a scale on a must_have; the rule is applied on the way in."""
    sloppy = [{"requirement": "Python", "description": "", "type": "must_have",
               "scale": "5=expert", "suggested_weight": 4}]
    run = make_run()

    with patch("jobs.views.extract_requirements", return_value=sloppy):
        client.post(reverse("jobs:jd_extract", args=[run.pk]))

    run.job_description.refresh_from_db()
    stored = run.job_description.requirements[0]
    assert stored["scale"] == "" and stored["suggested_weight"] == 0


# --- Editing: type, text, delete, add ---------------------------------------------

def base_form(requirements):
    data = {}
    for i, r in enumerate(requirements):
        data[f"req-{i}-requirement"] = r["requirement"]
        data[f"req-{i}-description"] = r["description"]
        data[f"req-{i}-type"] = r["type"]
        data[f"req-{i}-scale"] = r["scale"]
        data[f"req-{i}-suggested_weight"] = str(r["suggested_weight"])
    return data


def test_changing_a_requirements_type(client):
    """The main control at this checkpoint — how a miscategorised requirement gets fixed."""
    run = make_run(requirements=extracted())
    form = base_form(extracted())
    form["req-1-type"] = "must_have"                # Django: nice_to_have -> must_have

    client.post(reverse("jobs:jd_save", args=[run.pk]), form)

    run.job_description.refresh_from_db()
    assert run.job_description.requirements[1]["type"] == "must_have"


def test_switching_to_must_have_clears_scale_and_weight(client):
    run = make_run(requirements=extracted())
    form = base_form(extracted())
    form["req-1-type"] = "must_have"

    client.post(reverse("jobs:jd_save", args=[run.pk]), form)

    run.job_description.refresh_from_db()
    row = run.job_description.requirements[1]
    assert row["scale"] == "" and row["suggested_weight"] == 0


def test_editing_requirement_text_and_description(client):
    run = make_run(requirements=extracted())
    form = base_form(extracted())
    form["req-0-requirement"] = "6+ years Python"
    form["req-0-description"] = "Tightened after a calibration call"

    client.post(reverse("jobs:jd_save", args=[run.pk]), form)

    run.job_description.refresh_from_db()
    assert run.job_description.requirements[0]["requirement"] == "6+ years Python"
    assert "calibration" in run.job_description.requirements[0]["description"]


def test_deleting_a_requirement(client):
    run = make_run(requirements=extracted())
    form = base_form(extracted())
    form["req-0-delete"] = "on"

    client.post(reverse("jobs:jd_save", args=[run.pk]), form)

    run.job_description.refresh_from_db()
    assert [r["requirement"] for r in run.job_description.requirements] == ["Django"]


def test_deleting_every_requirement_is_allowed_but_blocks_confirming(client):
    run = make_run(requirements=extracted())
    form = base_form(extracted())
    form["req-0-delete"] = form["req-1-delete"] = "on"

    client.post(reverse("jobs:jd_save", args=[run.pk]), form)
    client.post(reverse("jobs:jd_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.job_description.requirements == []
    assert run.current_stage == "jd_pending"        # cannot confirm an empty list


def test_a_blank_requirement_name_is_rejected(client):
    run = make_run(requirements=extracted())
    form = base_form(extracted())
    form["req-0-requirement"] = "   "

    client.post(reverse("jobs:jd_save", args=[run.pk]), form)

    run.job_description.refresh_from_db()
    assert run.job_description.requirements[0]["requirement"] == "5+ years Python"


def test_an_invalid_type_is_rejected(client):
    run = make_run(requirements=extracted())
    form = base_form(extracted())
    form["req-0-type"] = "critical"

    client.post(reverse("jobs:jd_save", args=[run.pk]), form)

    run.job_description.refresh_from_db()
    assert run.job_description.requirements[0]["type"] == "must_have"


def test_a_non_numeric_weight_is_rejected(client):
    run = make_run(requirements=extracted())
    form = base_form(extracted())
    form["req-1-suggested_weight"] = "heavy"

    client.post(reverse("jobs:jd_save", args=[run.pk]), form)

    run.job_description.refresh_from_db()
    assert run.job_description.requirements[1]["suggested_weight"] == 3


def test_adding_a_requirement_manually(client):
    run = make_run(requirements=extracted())

    client.post(reverse("jobs:jd_add", args=[run.pk]),
                {"new-requirement": "Arabic fluency", "new-description": "Client-facing role",
                 "new-type": "inferred", "new-scale": "5=native, 3=professional, 1=none",
                 "new-suggested_weight": "2"})

    run.job_description.refresh_from_db()
    added = run.job_description.requirements[-1]
    assert added["requirement"] == "Arabic fluency"
    assert added["type"] == "inferred"
    assert added["suggested_weight"] == 2


def test_adding_a_nameless_requirement_is_rejected(client):
    run = make_run(requirements=extracted())

    client.post(reverse("jobs:jd_add", args=[run.pk]),
                {"new-requirement": "  ", "new-type": "inferred"})

    run.job_description.refresh_from_db()
    assert len(run.job_description.requirements) == 2


# --- Confirming: the jd_pending -> cv_pending transition ---------------------------

def test_confirming_advances_the_run_to_cv_pending(client):
    """The transition that did not exist before this app was built."""
    run = make_run(requirements=extracted())

    response = client.post(reverse("jobs:jd_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "cv_pending"
    assert run.job_description.confirmed_by_user is True
    assert response.url == reverse("candidates:upload", args=[run.pk])


def test_confirming_is_refused_with_no_requirements(client):
    run = make_run(requirements=[])

    client.post(reverse("jobs:jd_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "jd_pending"
    assert run.job_description.confirmed_by_user is False


def test_confirming_with_no_must_have_is_allowed_but_warned(client):
    """Legal — it just means Stage 1 has no hard gate — so warn, do not block."""
    only_nice = [r for r in extracted() if r["type"] != "must_have"]
    run = make_run(requirements=only_nice)

    response = client.post(reverse("jobs:jd_confirm", args=[run.pk]), follow=True)

    run.refresh_from_db()
    assert run.current_stage == "cv_pending"
    assert any("no hard gate" in str(m).lower() or "must-have" in str(m).lower()
               for m in response.context["messages"])


def test_a_confirmed_job_description_cannot_be_edited(client):
    """Editing after Stage 1 has run would leave stored verdicts judged against
    requirements that no longer exist."""
    run = make_run(requirements=extracted(), confirmed=True, stage="cv_pending")
    form = base_form(extracted())
    form["req-0-requirement"] = "tampered"

    client.post(reverse("jobs:jd_save", args=[run.pk]), form)

    run.job_description.refresh_from_db()
    assert run.job_description.requirements[0]["requirement"] == "5+ years Python"


def test_a_confirmed_job_description_cannot_be_re_extracted(client):
    run = make_run(requirements=extracted(), confirmed=True, stage="cv_pending")

    with patch("jobs.views.extract_requirements") as mock:
        client.post(reverse("jobs:jd_extract", args=[run.pk]))

    mock.assert_not_called()


def test_confirming_twice_does_not_skip_a_stage(client):
    run = make_run(requirements=extracted())
    client.post(reverse("jobs:jd_confirm", args=[run.pk]))
    client.post(reverse("jobs:jd_confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "cv_pending"        # not pushed on to stage1_pending


# --- The review page ----------------------------------------------------------------

def test_the_review_page_lists_every_requirement_with_a_type_control(client):
    run = make_run(requirements=extracted())

    html = client.get(reverse("jobs:jd_review", args=[run.pk])).content.decode()

    assert "5+ years Python" in html and "Django" in html
    assert 'name="req-0-type"' in html and 'name="req-1-type"' in html
    for requirement_type in ("must_have", "nice_to_have", "inferred"):
        assert requirement_type in html


def test_the_review_page_warns_when_there_are_no_must_haves(client):
    run = make_run(requirements=[r for r in extracted() if r["type"] != "must_have"])

    response = client.get(reverse("jobs:jd_review", args=[run.pk]))

    assert response.context["must_have_count"] == 0
    assert "no hard gate" in flat(response).lower()


def test_the_review_page_is_read_only_once_confirmed(client):
    run = make_run(requirements=extracted(), confirmed=True, stage="cv_pending")

    response = client.get(reverse("jobs:jd_review", args=[run.pk]))

    assert response.context["locked"] is True
    assert "confirmed and locked" in response.content.decode()


# --- The run list -----------------------------------------------------------------

def test_the_run_list_shows_each_run_and_links_to_its_current_stage(client):
    run = make_run(requirements=extracted())

    response = client.get(reverse("runs:run_list"))
    html = response.content.decode()

    assert f"#{run.pk}" in html
    assert reverse("jobs:jd_review", args=[run.pk]) in html
    assert response.context["rows"][0]["requirement_count"] == 2


def test_the_run_list_links_a_confirmed_run_to_cv_upload(client):
    run = make_run(requirements=extracted(), confirmed=True, stage="cv_pending")

    html = client.get(reverse("runs:run_list")).content.decode()

    assert reverse("candidates:review", args=[run.pk]) in html


def test_the_run_list_is_empty_without_runs(client):
    response = client.get(reverse("runs:run_list"))

    assert response.status_code == 200
    assert response.context["rows"] == []


# --- The handover into the existing pipeline ------------------------------------------

def test_after_confirming_cv_upload_accepts_files(client):
    """The whole point: JD confirm must leave the run in a state the next stage accepts."""
    run = make_run(requirements=extracted())
    client.post(reverse("jobs:jd_confirm", args=[run.pk]))

    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload("clean_cv.pdf")]})

    assert Candidate.objects.filter(run=run, status="pending").count() == 1
