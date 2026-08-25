"""Tests for JD processing's pure logic, with the LLM mocked. No network calls.

The prompt contract matters as much as the output here: this single categorisation
drives both Stage 1 (must_have) and Stage 3 (nice_to_have + inferred), so an
instruction that stops telling the model how to tag would silently mis-route every
requirement in the pipeline.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from screening.jd.extraction import extract_requirements, get_jd_text, normalize_requirement
from screening.jd.schema import JD_SCHEMA, REQUIREMENT_TYPES

FIXTURES = Path(__file__).parent / "fixtures"

JD_TEXT = """Backend Engineer — Casablanca
Required: 5+ years of professional Python experience.
Django experience is a plus."""


def extracted():
    return {"requirements": [
        {"requirement": "5+ years of professional Python experience",
         "description": "Stated as required", "type": "must_have",
         "scale": "", "suggested_weight": 0},
        {"requirement": "Django", "description": "Stated as a plus",
         "type": "nice_to_have", "scale": "5=deep experience, 3=some, 1=none",
         "suggested_weight": 3},
    ]}


class FakeUpload:
    """Stands in for a FileField — get_jd_text only reads `.path`."""
    def __init__(self, path):
        self.path = str(path)


# --- get_jd_text: the second caller of the shared extraction function ----------

def test_free_text_is_returned_as_given():
    assert get_jd_text(free_text=JD_TEXT) == JD_TEXT


def test_an_uploaded_file_goes_through_the_shared_extraction_function():
    """The JD file path and the CV path run the same code — one function, two callers."""
    text = get_jd_text(uploaded_file=FakeUpload(FIXTURES / "clean_cv.pdf"))

    assert "Amina Benali" in text          # proves real extraction ran, not a stub


def test_a_docx_job_description_also_works():
    text = get_jd_text(uploaded_file=FakeUpload(FIXTURES / "clean_cv.docx"))

    assert "Youssef El Amrani" in text


def test_neither_input_raises():
    with pytest.raises(ValueError, match="Either uploaded_file or free_text"):
        get_jd_text()


def test_a_file_wins_when_both_are_supplied():
    text = get_jd_text(uploaded_file=FakeUpload(FIXTURES / "clean_cv.pdf"), free_text="ignored")

    assert "Amina Benali" in text and "ignored" not in text


def test_an_unreadable_job_description_propagates_the_error():
    """A corrupted JD must surface, not silently become an empty requirements list."""
    with pytest.raises(RuntimeError):
        get_jd_text(uploaded_file=FakeUpload(FIXTURES / "corrupted.pdf"))


# --- extract_requirements: what goes to the model ------------------------------

def test_jd_text_is_the_input_and_the_schema_is_jd_schema():
    with patch("screening.jd.extraction.call_structured", return_value=extracted()) as mock:
        extract_requirements(JD_TEXT)

    instructions, input_text, schema_name, schema = mock.call_args.args
    assert input_text == JD_TEXT           # passed through untouched
    assert schema_name == "jd_requirements"
    assert schema is JD_SCHEMA


def test_instructions_define_all_three_types():
    """The type definitions are the whole point of this call."""
    with patch("screening.jd.extraction.call_structured", return_value=extracted()) as mock:
        extract_requirements(JD_TEXT)

    instructions = mock.call_args.args[0]
    for requirement_type in REQUIREMENT_TYPES:
        assert requirement_type in instructions


def test_instructions_bias_away_from_over_tagging_must_have():
    """A false must_have becomes an unrecoverable Stage 1 rejection, so the prompt
    deliberately pushes doubt towards the other two types."""
    with patch("screening.jd.extraction.call_structured", return_value=extracted()) as mock:
        extract_requirements(JD_TEXT)

    instructions = mock.call_args.args[0]
    assert "when in doubt" in instructions
    assert "Do not invent requirements" in instructions


def test_instructions_specify_the_fixed_scale_format():
    with patch("screening.jd.extraction.call_structured", return_value=extracted()) as mock:
        extract_requirements(JD_TEXT)

    assert "5=<strong evidence>, 3=<meets baseline>, 1=<no evidence>" in mock.call_args.args[0]


def test_the_requirements_list_is_returned_unmodified():
    payload = extracted()

    with patch("screening.jd.extraction.call_structured", return_value=payload):
        assert extract_requirements(JD_TEXT) == payload["requirements"]


def test_an_empty_extraction_is_returned_as_an_empty_list():
    """A JD the model finds nothing in must not crash — the checkpoint tells the user."""
    with patch("screening.jd.extraction.call_structured", return_value={"requirements": []}):
        assert extract_requirements("") == []


# --- normalize_requirement: the must_have rule ----------------------------------

def test_must_have_loses_its_scale_and_weight():
    """CLAUDE.md: must_have carries scale "" and suggested_weight 0 — both unused.
    Without this, a requirement switched to must_have keeps a stale scale that
    Stage 3 silently ignores."""
    result = normalize_requirement({
        "requirement": "Python", "description": "", "type": "must_have",
        "scale": "5=expert, 3=some, 1=none", "suggested_weight": 4,
    })

    assert result["scale"] == ""
    assert result["suggested_weight"] == 0


@pytest.mark.parametrize("requirement_type", ["nice_to_have", "inferred"])
def test_scorable_types_keep_their_scale_and_weight(requirement_type):
    item = {"requirement": "Kubernetes", "description": "", "type": requirement_type,
            "scale": "5=expert, 3=some, 1=none", "suggested_weight": 3}

    assert normalize_requirement(item) == item


def test_normalize_does_not_mutate_the_original():
    original = {"requirement": "Python", "description": "", "type": "must_have",
                "scale": "5=expert", "suggested_weight": 4}
    normalize_requirement(original)

    assert original["scale"] == "5=expert"


# --- The output shape the rest of the pipeline depends on -------------------------

def test_extracted_items_carry_every_key_the_schema_requires():
    """Stage 1 reads `requirement` and `type`; Stage 3 reads `scale` and
    `suggested_weight`. A missing key downstream is a KeyError, not a soft failure."""
    with patch("screening.jd.extraction.call_structured", return_value=extracted()):
        requirements = extract_requirements(JD_TEXT)

    required = set(JD_SCHEMA["properties"]["requirements"]["items"]["required"])
    for item in requirements:
        assert required <= set(item)


def test_downstream_selectors_read_the_extracted_list_correctly():
    """End-to-end on the data contract: Stage 1 and Stage 3 must partition the list."""
    from screening.stage1_hard_filter.evaluation import select_must_have
    from screening.stage3_scoring.rubric import build_rubric

    with patch("screening.jd.extraction.call_structured", return_value=extracted()):
        requirements = extract_requirements(JD_TEXT)

    assert [r["requirement"] for r in select_must_have(requirements)] == [
        "5+ years of professional Python experience"]
    assert [c["requirement"] for c in build_rubric(requirements)] == ["Django"]
