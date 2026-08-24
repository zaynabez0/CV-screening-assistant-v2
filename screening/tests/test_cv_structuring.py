"""Tests for CV structuring and anonymization, with the LLM mocked.

No test here makes a network call. `structure_candidate` is tested for what OUR code
does — the prompt it builds, the schema it sends, that it passes the CV text through
intact and returns the model's answer unmodified. Whether the model extracts *well* is
a different question, answered by the week-2 golden set against the real API.

The three synthetic CVs (career gap, non-standard degree, long CV) are real text files
under fixtures/synthetic_cvs/. They double as ready-made inputs for that manual check.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from screening.cv.anonymization import anonymize_candidate
from screening.cv.schema import CV_SCHEMA
from screening.cv.structuring import structure_candidate

SYNTHETIC = Path(__file__).parent / "fixtures" / "synthetic_cvs"


def load(name: str) -> str:
    return (SYNTHETIC / name).read_text(encoding="utf-8")


def fake_extraction(**overrides) -> dict:
    """A schema-shaped response, as the real model would return it."""
    base = {
        "full_name": "Nadia Cherkaoui",
        "email": "nadia.cherkaoui@example.com",
        "phone": "+212 6 00 00 00 00",
        "languages": [{"language": "Arabic", "proficiency": "native"}],
        "education": [{"degree": "MSc", "field": "Software Engineering",
                       "institution": "Universite Hassan II", "end_year": 2017}],
        "experience": [{"title": "Senior Backend Engineer", "company": "Kite Logistics",
                        "start_date": "2023-03", "end_date": None,
                        "description": "Shipment tracking API.",
                        "evidence_quote": "Design and maintain the shipment tracking API"}],
        "skills": ["Python", "Django"],
        "total_years_experience": 6,
        "certifications": [],
        "extraction_confidence": "high",
        "extraction_notes": "",
    }
    base.update(overrides)
    return base


# --- What structure_candidate sends to the LLM -------------------------------

def test_cv_text_is_passed_through_untouched():
    cv_text = load("career_gap.txt")

    with patch("screening.cv.structuring.call_structured", return_value=fake_extraction()) as mock:
        structure_candidate(cv_text)

    instructions, input_text, schema_name, schema = mock.call_args.args
    assert input_text == cv_text          # not trimmed, not reformatted, not wrapped
    assert schema_name == "candidate_data"
    assert schema is CV_SCHEMA


def test_instructions_come_first_and_are_identical_across_candidates():
    """Prompt caching depends on the repeated part being the PREFIX. If the per-candidate
    text leaked into `instructions`, every call would miss the cache."""
    calls = []
    with patch("screening.cv.structuring.call_structured",
               side_effect=lambda *a, **k: calls.append(a) or fake_extraction()):
        structure_candidate(load("career_gap.txt"))
        structure_candidate(load("non_standard_degree.txt"))

    assert calls[0][0] == calls[1][0]                     # instructions identical
    assert calls[0][1] != calls[1][1]                     # inputs differ
    assert "Cherkaoui" not in calls[0][0]                 # no candidate data in the prefix


def test_instructions_forbid_inventing_data():
    """The 'never infer' rule is what makes evidence quotes trustworthy downstream."""
    with patch("screening.cv.structuring.call_structured", return_value=fake_extraction()) as mock:
        structure_candidate("some cv text")

    instructions = mock.call_args.args[0]
    assert "never infer" in instructions
    assert "verbatim quote" in instructions
    assert "extraction_confidence" in instructions


def test_result_is_returned_unmodified():
    """Our code must not post-process the model's answer — the stored JSON is the audit
    trail, and quietly editing it would break traceability."""
    payload = fake_extraction(extraction_confidence="low", extraction_notes="garbled")

    with patch("screening.cv.structuring.call_structured", return_value=payload):
        assert structure_candidate("text") == payload


# --- The three synthetic CVs -------------------------------------------------

def test_long_cv_is_not_truncated():
    """A silent truncation would drop a candidate's most recent role and nobody
    would see an error — the worst kind of bug in this pipeline."""
    cv_text = load("long_cv.txt")
    assert len(cv_text) > 3000
    assert "Northgate Telecom" in cv_text                  # the oldest role, at the end

    with patch("screening.cv.structuring.call_structured", return_value=fake_extraction()) as mock:
        structure_candidate(cv_text)

    sent = mock.call_args.args[1]
    assert len(sent) == len(cv_text)
    assert "Northgate Telecom" in sent


def test_non_standard_degree_cv_keeps_accented_characters():
    """French credentials carry accents. If they are mangled before the LLM sees them,
    'Diplome d'Ingenieur d'Etat' is a different string to the embedding model in
    Stage 2 and to the LLM here — and nothing raises an error when it happens.
    """
    cv_text = load("non_standard_degree.txt")
    assert "Diplôme d'Ingénieur d'État" in cv_text
    assert "École Nationale Supérieure" in cv_text

    with patch("screening.cv.structuring.call_structured", return_value=fake_extraction()) as mock:
        structure_candidate(cv_text)

    assert "Diplôme d'Ingénieur d'État" in mock.call_args.args[1]


def test_career_gap_cv_reaches_the_llm_with_both_roles_and_their_dates():
    """The gap is only detectable if BOTH roles and their dates survive. Stage 1 may be
    asked about continuous experience, so losing a date here changes an outcome."""
    cv_text = load("career_gap.txt")

    with patch("screening.cv.structuring.call_structured", return_value=fake_extraction()) as mock:
        structure_candidate(cv_text)

    sent = mock.call_args.args[1]
    assert "March 2023" in sent and "January 2020" in sent   # the 3-year gap, either side
    assert "Numera Finance" in sent and "Kite Logistics" in sent


@pytest.mark.parametrize("name", ["career_gap.txt", "non_standard_degree.txt", "long_cv.txt"])
def test_every_synthetic_cv_is_json_serialisable_as_a_prompt(name):
    """Whatever is in these files must survive being sent as JSON to the API."""
    text = load(name)
    assert json.loads(json.dumps({"input": text}))["input"] == text


# --- Anonymization ------------------------------------------------------------

def test_anonymize_strips_exactly_the_three_pii_fields():
    anon = anonymize_candidate(fake_extraction())

    assert "full_name" not in anon and "email" not in anon and "phone" not in anon
    assert set(CV_SCHEMA["required"]) - {"full_name", "email", "phone"} == set(anon)


def test_anonymize_does_not_mutate_the_original():
    """Stages compute this on demand; if it mutated in place, the first stage to run
    would silently destroy the recruiter's view of the candidate."""
    original = fake_extraction()
    anonymize_candidate(original)

    assert original["full_name"] == "Nadia Cherkaoui"


def test_anonymize_leaves_evidence_and_scoring_fields_intact():
    anon = anonymize_candidate(fake_extraction())

    assert anon["experience"][0]["evidence_quote"]
    assert anon["skills"] == ["Python", "Django"]
    assert anon["total_years_experience"] == 6


def test_anonymize_is_a_shallow_copy_so_names_inside_free_text_survive():
    """Known limitation, pinned deliberately: only the three top-level keys are
    stripped. A name appearing inside an evidence_quote or extraction_notes is NOT
    removed, and would reach the LLM in Stages 1 and 3.
    """
    candidate = fake_extraction(extraction_notes="Nadia's dates were ambiguous.")
    anon = anonymize_candidate(candidate)

    assert "Nadia" in anon["extraction_notes"]
