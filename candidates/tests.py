"""Django-side tests for CV preprocessing: the upload view, structuring, review page.

The LLM is mocked everywhere. Uploads are written to a temporary MEDIA_ROOT so the
suite never touches the real media/ directory.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from candidates.models import Candidate
from jobs.models import JobDescription
from screening.common import ocr
from runs.models import ScreeningRun
from runs.services import run_cv_structuring

FIXTURES = Path(__file__).resolve().parent.parent / "screening" / "tests" / "fixtures"

pytestmark = pytest.mark.django_db

needs_ocr = pytest.mark.skipif(
    not ocr.ocr_available(), reason="Tesseract is not installed on this machine"
)


@pytest.fixture(autouse=True)
def media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "media"
    return settings.MEDIA_ROOT


@pytest.fixture
def run():
    """A run sitting at CV preprocessing — the only stage where these views work."""
    jd = JobDescription.objects.create(input_mode="text", raw_text="Backend engineer.")
    return ScreeningRun.objects.create(job_description=jd, current_stage="cv_pending")


def upload_file(name, fixture=None):
    data = (FIXTURES / (fixture or name)).read_bytes()
    return SimpleUploadedFile(name, data)


def fake_extraction(**overrides):
    base = {
        "full_name": "Amina Benali", "email": "a@example.com", "phone": "+212600000000",
        "languages": [{"language": "French", "proficiency": "fluent"}],
        "education": [{"degree": "MSc", "field": "CS", "institution": "ENSIAS", "end_year": 2021}],
        "experience": [{"title": "Backend Developer", "company": "Atlas Data",
                        "start_date": "2021-01", "end_date": "2024-01",
                        "description": "ETL pipelines.", "evidence_quote": "Built Python ETL pipelines"}],
        "skills": ["Python", "SQL"], "total_years_experience": 3, "certifications": [],
        "extraction_confidence": "high", "extraction_notes": "",
    }
    base.update(overrides)
    return base


def structure(run, **patch_kwargs):
    """Structure the run's pending CVs, the way the background command does.

    Structuring is a subprocess-launched management command now, so these tests call the
    service directly rather than posting to a view — the same arrangement runs/tests.py
    already uses for the three stages. Returns the mock so callers can assert on the call.
    """
    patch_kwargs.setdefault("return_value", fake_extraction())
    with patch("runs.services.structure_candidate", **patch_kwargs) as mock:
        run_cv_structuring(run)
    return mock


# --- Upload -------------------------------------------------------------------

def test_upload_accepts_a_clean_pdf(client, run):
    response = client.post(reverse("candidates:upload", args=[run.pk]),
                           {"cvs": [upload_file("clean_cv.pdf")]})

    assert response.status_code == 302
    candidate = Candidate.objects.get(run=run)
    assert candidate.status == "pending"          # validated, NOT yet structured
    assert len(candidate.file_hash) == 64
    assert candidate.extracted_json is None       # upload must not call the LLM


def test_upload_does_not_call_the_llm(client, run):
    with patch("runs.services.structure_candidate") as mock:
        client.post(reverse("candidates:upload", args=[run.pk]),
                    {"cvs": [upload_file("clean_cv.pdf")]})

    mock.assert_not_called()


def test_upload_records_a_rejection_reason_instead_of_crashing(client, run):
    """A corrupt file in the batch must be recorded and survived, not 500."""
    response = client.post(reverse("candidates:upload", args=[run.pk]),
                           {"cvs": [upload_file("corrupted.pdf"), upload_file("clean_cv.pdf")]})

    assert response.status_code == 302
    corrupt = Candidate.objects.get(run=run, original_filename="corrupted.pdf")
    clean = Candidate.objects.get(run=run, original_filename="clean_cv.pdf")
    assert corrupt.status == "rejected"
    assert "could not read file" in corrupt.rejection_reason
    assert clean.status == "pending"              # one bad file does not sink the batch


def test_upload_flags_a_file_with_nothing_readable_with_the_specified_reason(client, run):
    """Not a scan any more — OCR reads those. This fixture is a page of drawn shapes,
    so there is genuinely nothing to find, by text layer or by OCR."""
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("no_text_layer.pdf")]})

    candidate = Candidate.objects.get(run=run)
    assert candidate.status == "rejected"
    # startswith, not ==: on a server with no Tesseract the reason says so as well.
    assert candidate.rejection_reason.startswith("no extractable text found")


@needs_ocr
def test_upload_accepts_a_scanned_cv(client, run):
    """The headline change: a CV that is only a picture is validated like any other."""
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("scanned_cv.pdf")]})

    candidate = Candidate.objects.get(run=run)
    assert candidate.status == "pending"
    assert "KARIM ZAHRAOUI" in candidate.extracted_text


@needs_ocr
def test_upload_accepts_an_image_of_a_cv(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("scanned_cv.png")]})

    candidate = Candidate.objects.get(run=run)
    assert candidate.status == "pending"
    assert "Nador Steelworks" in candidate.extracted_text


def test_upload_keeps_the_text_it_read(client, run):
    """Validation reads the file to check it holds text; throwing that away would mean
    structuring re-runs OCR on every scan for a result that cannot have changed."""
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})

    assert "Amina Benali" in Candidate.objects.get(run=run).extracted_text


def test_upload_marks_a_second_copy_as_duplicate_not_rejected(client, run):
    url = reverse("candidates:upload", args=[run.pk])
    client.post(url, {"cvs": [upload_file("clean_cv.pdf")]})
    client.post(url, {"cvs": [upload_file("clean_cv.pdf")]})

    statuses = sorted(Candidate.objects.filter(run=run).values_list("status", flat=True))
    assert statuses == ["duplicate", "pending"]


def test_dedup_is_scoped_to_the_run(client, run):
    """The same CV may legitimately be submitted to two different job openings."""
    other_jd = JobDescription.objects.create(input_mode="text", raw_text="Other role.")
    other_run = ScreeningRun.objects.create(job_description=other_jd, current_stage="cv_pending")

    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    client.post(reverse("candidates:upload", args=[other_run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})

    assert Candidate.objects.get(run=other_run).status == "pending"


def test_upload_is_blocked_when_the_run_is_at_another_stage(client, run):
    run.current_stage = "stage1_pending"
    run.save()

    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})

    assert Candidate.objects.filter(run=run).count() == 0


# --- Structuring --------------------------------------------------------------

def test_structuring_is_a_separate_user_triggered_step(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})

    mock = structure(run, return_value=fake_extraction())

    mock.assert_called_once()
    candidate = Candidate.objects.get(run=run)
    assert candidate.status == "ready"
    assert candidate.extracted_json["full_name"] == "Amina Benali"


def test_structuring_receives_the_extracted_cv_text(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})

    mock = structure(run, return_value=fake_extraction())

    assert "Amina Benali" in mock.call_args.args[0]     # real text from the PDF, not the filename


def test_structuring_reuses_the_text_from_upload_instead_of_reading_the_file_again(client, run):
    """The whole point of storing it: on a batch of scans, re-extracting here would
    repeat seconds of OCR per page."""
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})

    with patch("runs.services.extract_text_from_file") as extract:
        mock = structure(run, return_value=fake_extraction())

    extract.assert_not_called()
    assert "Amina Benali" in mock.call_args.args[0]


def test_structuring_still_reads_the_file_when_no_text_was_stored(client, run):
    """Candidates recorded before the text was stored must keep working."""
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    Candidate.objects.filter(run=run).update(extracted_text="")

    mock = structure(run, return_value=fake_extraction())

    assert "Amina Benali" in mock.call_args.args[0]
    assert "Amina Benali" in Candidate.objects.get(run=run).extracted_text   # and stored now


def test_structuring_skips_rejected_candidates(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("no_text_layer.pdf"), upload_file("clean_cv.pdf")]})

    mock = structure(run, return_value=fake_extraction())

    assert mock.call_count == 1                        # only the valid one


def test_an_llm_failure_leaves_the_candidate_retryable(client, run):
    """An API error is not the candidate's fault — the row must stay retryable rather
    than being marked rejected, which would look like the CV was at fault."""
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})

    structure(run, side_effect=RuntimeError("API down"))

    candidate = Candidate.objects.get(run=run)
    assert candidate.status == "pending"
    assert "structuring failed" in candidate.rejection_reason
    assert candidate.extracted_json is None

    structure(run, return_value=fake_extraction())

    candidate.refresh_from_db()
    assert candidate.status == "ready"
    assert candidate.rejection_reason == ""


# --- Review checkpoint --------------------------------------------------------

def test_review_page_shows_the_extracted_json_and_confidence(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    structure(run, return_value=fake_extraction())

    html = client.get(reverse("candidates:review", args=[run.pk])).content.decode()

    assert "clean_cv.pdf" in html
    assert "confidence: high" in html
    assert "Atlas Data" in html                        # the JSON itself is rendered


def test_low_confidence_extraction_is_visually_flagged(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    payload = fake_extraction(extraction_confidence="low", extraction_notes="Two-column layout garbled.")
    structure(run, return_value=payload)

    response = client.get(reverse("candidates:review", args=[run.pk]))
    html = response.content.decode()

    assert response.context["low_confidence_count"] == 1
    assert "flagged" in html                            # the CSS class driving the highlight
    assert "Low confidence" in html
    assert "Two-column layout garbled." in html         # the model's own explanation


def test_high_confidence_extraction_is_not_flagged(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    structure(run, return_value=fake_extraction())

    response = client.get(reverse("candidates:review", args=[run.pk]))

    assert response.context["low_confidence_count"] == 0
    assert "Low confidence" not in response.content.decode()


def test_review_shows_rejected_candidates_too(client, run):
    """GDPR Art. 22: a filtered-out CV stays visible, never silently disappears."""
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("no_text_layer.pdf")]})

    html = client.get(reverse("candidates:review", args=[run.pk])).content.decode()

    assert "no_text_layer.pdf" in html
    assert "no extractable text found" in html


# --- Confirming the checkpoint ------------------------------------------------

def test_confirming_advances_the_run_to_stage1(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    structure(run, return_value=fake_extraction())

    client.post(reverse("candidates:confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "stage1_pending"


def test_the_confirmed_review_page_links_on_to_stage_1(client, run):
    """A confirmed checkpoint is a finished step, not a dead end.

    Once the run has moved on, this page can no longer do anything — so it has to say
    where to go instead of leaving a disabled button and a red banner behind.
    """
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    structure(run, return_value=fake_extraction())
    client.post(reverse("candidates:confirm", args=[run.pk]))

    html = client.get(reverse("candidates:review", args=[run.pk])).content.decode()

    assert reverse("runs:stage1_review", args=[run.pk]) in html
    assert "Confirm preprocessing and continue" not in html   # the dead button is gone
    assert "not CV preprocessing" not in html                 # and so is the wrong-stage banner


def test_the_unconfirmed_review_page_still_offers_the_button_and_no_link(client, run):
    """The forward link must not appear before the checkpoint is actually confirmed."""
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    structure(run, return_value=fake_extraction())

    html = client.get(reverse("candidates:review", args=[run.pk])).content.decode()

    assert "Confirm preprocessing and continue" in html
    assert reverse("runs:stage1_review", args=[run.pk]) not in html


def test_a_run_stuck_before_this_stage_still_gets_the_blocked_banner(client, run):
    """Suppressing the banner is only for a run that has moved *past* this checkpoint.
    One that has not reached it yet is genuinely blocked and must still be told so."""
    run.current_stage = "jd_pending"
    run.save(update_fields=["current_stage"])

    html = client.get(reverse("candidates:review", args=[run.pk])).content.decode()

    assert "not CV preprocessing" in html
    assert reverse("runs:stage1_review", args=[run.pk]) not in html


def test_structuring_alone_does_not_advance_the_run(client, run):
    """No stage auto-runs the next. Only the human confirmation moves current_stage."""
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    structure(run, return_value=fake_extraction())

    run.refresh_from_db()
    assert run.current_stage == "cv_pending"


def test_cannot_confirm_with_nothing_structured(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("no_text_layer.pdf")]})

    client.post(reverse("candidates:confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "cv_pending"


# --- Structuring is a background job now ---------------------------------------

def test_pressing_structure_returns_immediately_and_spawns_the_job(client, run):
    """It used to loop the LLM inline and hold the request open for the whole batch.
    Now it hands off, so the page can come back and show a progress bar."""
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("clean_cv.pdf")]})

    with patch("candidates.views.subprocess.Popen") as popen:
        with patch("runs.services.structure_candidate") as llm:
            response = client.post(reverse("candidates:structure", args=[run.pk]))

    assert response.status_code == 302
    llm.assert_not_called()                       # the work happens in the subprocess
    assert popen.call_args.args[0][2] == "run_cv_structuring"

    run.refresh_from_db()
    assert run.job_status == "running"
    assert run.job_stage == "cv_structuring"      # 14 chars — needs the widened column
    assert run.job_total == 1


def test_structuring_refuses_while_a_job_is_already_running(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("clean_cv.pdf")]})
    run.mark_job_started("cv_structuring", total=1)

    with patch("candidates.views.subprocess.Popen") as popen:
        client.post(reverse("candidates:structure", args=[run.pk]))

    popen.assert_not_called()


def test_the_review_page_shows_a_progress_bar_while_structuring_runs(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("clean_cv.pdf")]})
    run.mark_job_started("cv_structuring", total=4)
    run.mark_job_progress(1)

    html = client.get(reverse("candidates:review", args=[run.pk])).content.decode()

    assert 'class="bar-fill"' in html
    assert "width: 25%" in html
    assert "1 / 4" in html


def test_the_structure_button_is_hidden_while_the_job_runs(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("clean_cv.pdf")]})
    run.mark_job_started("cv_structuring", total=1)

    html = client.get(reverse("candidates:review", args=[run.pk])).content.decode()

    assert "validated CV(s) with the LLM" not in html


def test_confirming_is_refused_while_structuring_is_still_running(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("clean_cv.pdf")]})
    structure(run)
    run.mark_job_started("cv_structuring", total=1)

    client.post(reverse("candidates:confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "cv_pending"       # not advanced

