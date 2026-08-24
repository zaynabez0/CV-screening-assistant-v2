"""Tests for Stage 1's pure logic, with the LLM mocked.

Two cases anchor everything: a clear pass (every must-have met) and a clear fail (one
must-have missing). The rest pin the properties that make a verdict defensible — the
full evaluation list survives, the prompt prefix stays constant, PII never reaches the
model.
"""

from unittest.mock import patch

import pytest

from screening.stage1_hard_filter.evaluation import evaluate_hard_filters, select_must_have
from screening.stage1_hard_filter.schema import HARD_FILTER_SCHEMA

MUST_HAVES = [
    {"requirement": "5+ years Python", "description": "Professional Python experience",
     "type": "must_have", "scale": "", "suggested_weight": 0},
    {"requirement": "Fluent French", "description": "Written and spoken",
     "type": "must_have", "scale": "", "suggested_weight": 0},
]

ANON_CANDIDATE = {
    "languages": [{"language": "French", "proficiency": "fluent"}],
    "skills": ["Python", "Django"],
    "total_years_experience": 6,
    "experience": [{"title": "Backend Engineer", "company": "Kite Logistics",
                    "start_date": "2018-01", "end_date": None, "description": "Python services.",
                    "evidence_quote": "Six years building Python services"}],
    "education": [], "certifications": [],
    "extraction_confidence": "high", "extraction_notes": "",
}


def all_met():
    return {"evaluations": [
        {"requirement": "5+ years Python", "met": True,
         "evidence": "Six years building Python services", "reason": "6 years stated, exceeds 5."},
        {"requirement": "Fluent French", "met": True,
         "evidence": "French (fluent)", "reason": "Listed as fluent."},
    ]}


def one_unmet():
    return {"evaluations": [
        {"requirement": "5+ years Python", "met": True,
         "evidence": "Six years building Python services", "reason": "6 years stated, exceeds 5."},
        {"requirement": "Fluent French", "met": False,
         "evidence": "", "reason": "No French listed anywhere in the candidate data."},
    ]}


# --- The two anchor cases ----------------------------------------------------

def test_clear_pass_when_every_must_have_is_met():
    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=all_met()):
        result = evaluate_hard_filters(ANON_CANDIDATE, MUST_HAVES)

    assert result["passed"] is True
    assert len(result["evaluations"]) == 2
    assert all(e["evidence"] for e in result["evaluations"])   # a pass must be quotable


def test_clear_fail_when_one_must_have_is_missing():
    """One unmet requirement fails the candidate — must-haves are ANDed, not scored."""
    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=one_unmet()):
        result = evaluate_hard_filters(ANON_CANDIDATE, MUST_HAVES)

    assert result["passed"] is False
    assert len(result["evaluations"]) == 2                     # the met one is kept too
    unmet = [e for e in result["evaluations"] if not e["met"]]
    assert unmet[0]["reason"] == "No French listed anywhere in the candidate data."


def test_failure_keeps_the_full_evaluation_list_as_the_audit_trail():
    """Storing only the boolean would leave a rejected candidate with no reason —
    exactly what GDPR Art. 22 defensibility requires us to keep."""
    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=one_unmet()):
        result = evaluate_hard_filters(ANON_CANDIDATE, MUST_HAVES)

    assert [e["requirement"] for e in result["evaluations"]] == [
        "5+ years Python", "Fluent French",
    ]
    assert all("reason" in e for e in result["evaluations"])


# --- What gets sent to the model ---------------------------------------------

def test_requirements_go_in_the_instructions_and_the_candidate_in_the_input():
    """Prompt caching depends on this split: instructions are the repeated prefix,
    the candidate is the per-call part."""
    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               return_value=all_met()) as mock:
        evaluate_hard_filters(ANON_CANDIDATE, MUST_HAVES)

    instructions, input_text, schema_name, schema = mock.call_args.args
    assert "5+ years Python" in instructions and "Fluent French" in instructions
    assert input_text.startswith("Candidate: ")
    assert schema_name == "hard_filter_result"
    assert schema is HARD_FILTER_SCHEMA


def test_instructions_are_identical_across_candidates_in_a_run():
    calls = []
    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               side_effect=lambda *a, **k: calls.append(a) or all_met()):
        evaluate_hard_filters(ANON_CANDIDATE, MUST_HAVES)
        evaluate_hard_filters({**ANON_CANDIDATE, "skills": ["Go"]}, MUST_HAVES)

    assert calls[0][0] == calls[1][0]        # cached prefix intact
    assert calls[0][1] != calls[1][1]


def test_instructions_forbid_inferring():
    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               return_value=all_met()) as mock:
        evaluate_hard_filters(ANON_CANDIDATE, MUST_HAVES)

    instructions = mock.call_args.args[0]
    assert "never assume or infer" in instructions
    assert "quote the evidence" in instructions


def test_no_pii_reaches_the_model():
    """Stage 1 receives an already-anonymized candidate. If a caller ever passes the
    raw JSON, this is the test that should start failing."""
    with patch("screening.stage1_hard_filter.evaluation.call_structured",
               return_value=all_met()) as mock:
        evaluate_hard_filters(ANON_CANDIDATE, MUST_HAVES)

    sent = mock.call_args.args[1]
    assert "full_name" not in sent and "email" not in sent and "phone" not in sent


# --- Selecting must-haves ------------------------------------------------------

def test_select_must_have_ignores_the_other_two_types():
    requirements = MUST_HAVES + [
        {"requirement": "Kubernetes", "type": "nice_to_have", "description": "", "scale": "", "suggested_weight": 0.5},
        {"requirement": "Team leadership", "type": "inferred", "description": "", "scale": "", "suggested_weight": 0.3},
    ]

    assert [r["requirement"] for r in select_must_have(requirements)] == [
        "5+ years Python", "Fluent French",
    ]


def test_select_must_have_on_an_empty_jd():
    assert select_must_have([]) == []


# --- Edge cases ----------------------------------------------------------------

def test_a_jd_with_no_must_haves_passes_everyone_without_calling_the_llm():
    """No hard gate means no rejections. Calling the model with an empty requirement
    list would invite it to invent requirements, and cost one call per candidate."""
    with patch("screening.stage1_hard_filter.evaluation.call_structured") as mock:
        result = evaluate_hard_filters(ANON_CANDIDATE, [])

    mock.assert_not_called()
    assert result == {"passed": True, "evaluations": []}


def test_a_single_unmet_requirement_among_many_still_fails():
    many = {"evaluations": [
        {"requirement": f"req {i}", "met": True, "evidence": "e", "reason": "r"} for i in range(9)
    ] + [{"requirement": "req 9", "met": False, "evidence": "", "reason": "missing"}]}

    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=many):
        result = evaluate_hard_filters(ANON_CANDIDATE, MUST_HAVES)

    assert result["passed"] is False


@pytest.mark.parametrize("returned, expected", [(all_met(), True), (one_unmet(), False)])
def test_passed_is_derived_in_python_not_taken_from_the_model(returned, expected):
    """The model reports per-requirement facts; the AND is ours. The schema has no
    top-level "passed" field precisely so the model cannot overrule the arithmetic."""
    assert "passed" not in HARD_FILTER_SCHEMA["properties"]

    with patch("screening.stage1_hard_filter.evaluation.call_structured", return_value=returned):
        assert evaluate_hard_filters(ANON_CANDIDATE, MUST_HAVES)["passed"] is expected
