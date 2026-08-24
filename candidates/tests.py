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
from runs.models import ScreeningRun

FIXTURES = Path(__file__).resolve().parent.parent / "screening" / "tests" / "fixtures"

pytestmark = pytest.mark.django_db


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
    with patch("candidates.views.structure_candidate") as mock:
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


def test_upload_flags_a_scanned_pdf_with_the_specified_reason(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("no_text_layer.pdf")]})

    candidate = Candidate.objects.get(run=run)
    assert candidate.status == "rejected"
    assert candidate.rejection_reason == "no extractable text found"


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

    with patch("candidates.views.structure_candidate", return_value=fake_extraction()) as mock:
        client.post(reverse("candidates:structure", args=[run.pk]))

    mock.assert_called_once()
    candidate = Candidate.objects.get(run=run)
    assert candidate.status == "ready"
    assert candidate.extracted_json["full_name"] == "Amina Benali"


def test_structuring_receives_the_extracted_cv_text(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})

    with patch("candidates.views.structure_candidate", return_value=fake_extraction()) as mock:
        client.post(reverse("candidates:structure", args=[run.pk]))

    assert "Amina Benali" in mock.call_args.args[0]     # real text from the PDF, not the filename


def test_structuring_skips_rejected_candidates(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]),
                {"cvs": [upload_file("no_text_layer.pdf"), upload_file("clean_cv.pdf")]})

    with patch("candidates.views.structure_candidate", return_value=fake_extraction()) as mock:
        client.post(reverse("candidates:structure", args=[run.pk]))

    assert mock.call_count == 1                        # only the valid one


def test_an_llm_failure_leaves_the_candidate_retryable(client, run):
    """An API error is not the candidate's fault — the row must stay retryable rather
    than being marked rejected, which would look like the CV was at fault."""
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})

    with patch("candidates.views.structure_candidate", side_effect=RuntimeError("API down")):
        client.post(reverse("candidates:structure", args=[run.pk]))

    candidate = Candidate.objects.get(run=run)
    assert candidate.status == "pending"
    assert "structuring failed" in candidate.rejection_reason
    assert candidate.extracted_json is None

    with patch("candidates.views.structure_candidate", return_value=fake_extraction()):
        client.post(reverse("candidates:structure", args=[run.pk]))

    candidate.refresh_from_db()
    assert candidate.status == "ready"
    assert candidate.rejection_reason == ""


# --- Review checkpoint --------------------------------------------------------

def test_review_page_shows_the_extracted_json_and_confidence(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    with patch("candidates.views.structure_candidate", return_value=fake_extraction()):
        client.post(reverse("candidates:structure", args=[run.pk]))

    html = client.get(reverse("candidates:review", args=[run.pk])).content.decode()

    assert "clean_cv.pdf" in html
    assert "confidence: high" in html
    assert "Atlas Data" in html                        # the JSON itself is rendered


def test_low_confidence_extraction_is_visually_flagged(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    payload = fake_extraction(extraction_confidence="low", extraction_notes="Two-column layout garbled.")
    with patch("candidates.views.structure_candidate", return_value=payload):
        client.post(reverse("candidates:structure", args=[run.pk]))

    response = client.get(reverse("candidates:review", args=[run.pk]))
    html = response.content.decode()

    assert response.context["low_confidence_count"] == 1
    assert "flagged" in html                            # the CSS class driving the highlight
    assert "Low confidence" in html
    assert "Two-column layout garbled." in html         # the model's own explanation


def test_high_confidence_extraction_is_not_flagged(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    with patch("candidates.views.structure_candidate", return_value=fake_extraction()):
        client.post(reverse("candidates:structure", args=[run.pk]))

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
    with patch("candidates.views.structure_candidate", return_value=fake_extraction()):
        client.post(reverse("candidates:structure", args=[run.pk]))

    client.post(reverse("candidates:confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "stage1_pending"


def test_structuring_alone_does_not_advance_the_run(client, run):
    """No stage auto-runs the next. Only the human confirmation moves current_stage."""
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("clean_cv.pdf")]})
    with patch("candidates.views.structure_candidate", return_value=fake_extraction()):
        client.post(reverse("candidates:structure", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "cv_pending"


def test_cannot_confirm_with_nothing_structured(client, run):
    client.post(reverse("candidates:upload", args=[run.pk]), {"cvs": [upload_file("no_text_layer.pdf")]})

    client.post(reverse("candidates:confirm", args=[run.pk]))

    run.refresh_from_db()
    assert run.current_stage == "cv_pending"
