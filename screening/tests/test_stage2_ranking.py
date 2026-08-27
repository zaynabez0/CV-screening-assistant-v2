"""Tests for Stage 2 ranking.

Two layers, on purpose:

1. The call — that the prompt carries the job description and the keyword list, that
   the candidate is the only thing that varies between calls (so the prefix stays
   cacheable), and that no PII reaches the model. `call_structured` is mocked.
2. The arithmetic — every number Stage 2 stores is computed in Python from the model's
   judgements, so it is asserted against hand-computed values with the call stubbed out
   entirely.

There is deliberately no third "against the real model" layer. Stage 2's dependency
used to be a local embedding model that could be exercised for free; it is now a paid
API call, and the rule in this repo is that no test hits the network.
"""

from unittest.mock import patch

import pytest

from screening.stage2_ranking import ranking as ranking_module
from screening.stage2_ranking.keywords import seed_keywords
from screening.stage2_ranking.matching import match_candidate
from screening.stage2_ranking.ranking import (
    rank_candidates,
    resolve_keyword_matches,
    score_breakdown,
    score_candidate_match,
    shortlist_size,
)
from screening.stage2_ranking.schema import MATCH_SCHEMA
from screening.stage2_ranking.text_builders import build_jd_text


def candidate(skills=(), years=0, experience=()):
    return {"skills": list(skills), "total_years_experience": years, "experience": list(experience)}


PYTHON_DEV = candidate(
    skills=["Python", "Django", "PostgreSQL"], years=6,
    experience=[{"title": "Backend Engineer", "company": "Kite", "description": "Python APIs."}],
)
DESIGNER = candidate(
    skills=["Figma", "Illustrator"], years=4,
    experience=[{"title": "Product Designer", "company": "Studio", "description": "Design systems."}],
)

JD = [
    {"requirement": "Python", "description": "Backend development in Python", "type": "must_have"},
    {"requirement": "PostgreSQL", "description": "Relational databases", "type": "nice_to_have"},
]


def found(keyword, evidence="quoted from the CV"):
    return {"keyword": keyword, "matched": True, "evidence": evidence}


def missing(keyword, evidence="nothing found"):
    return {"keyword": keyword, "matched": False, "evidence": evidence}


def reply(fit=50, matches=(), reasoning="Scripted for the test."):
    """One model response, shaped like MATCH_SCHEMA."""
    return {"fit_score": fit, "fit_reasoning": reasoning, "keyword_matches": list(matches)}


@pytest.fixture
def scripted_model(monkeypatch):
    """Stub the LLM call so the arithmetic can be asserted exactly."""
    def install(*replies):
        calls = []

        def fake(jd_requirements, anon_candidate, keywords):
            calls.append(anon_candidate)
            return replies[len(calls) - 1] if len(calls) <= len(replies) else replies[-1]

        monkeypatch.setattr(ranking_module, "match_candidate", fake)
        return calls
    return install


# --- Layer 1: build_jd_text ----------------------------------------------------

def test_build_jd_text_pairs_each_requirement_with_its_description():
    assert build_jd_text(JD) == (
        "Python: Backend development in Python\nPostgreSQL: Relational databases"
    )


# --- Layer 1: seed_keywords ----------------------------------------------------

def test_keywords_seed_from_must_have_and_nice_to_have_only():
    requirements = JD + [{"requirement": "Team leadership", "description": "Implied by seniority",
                          "type": "inferred"}]

    assert seed_keywords(requirements) == ["Python", "PostgreSQL"]


def test_seeding_deduplicates_and_keeps_jd_order():
    requirements = [
        {"requirement": "Python", "description": "", "type": "must_have"},
        {"requirement": "Docker", "description": "", "type": "nice_to_have"},
        {"requirement": "python", "description": "", "type": "nice_to_have"},
    ]

    assert seed_keywords(requirements) == ["Python", "Docker"]


def test_seeding_an_empty_jd_gives_an_empty_list():
    assert seed_keywords([]) == []


# --- Layer 1: shortlist_size ---------------------------------------------------

@pytest.mark.parametrize("count, cutoff, expected", [
    (100, 25.0, 25),
    (10, 25.0, 3),        # rounds up: 2.5 -> 3
    (4, 25.0, 1),
    (3, 25.0, 1),         # 0.75 rounds up, and never below 1
    (1, 25.0, 1),
    (10, 100.0, 10),
    (10, 0.0, 1),         # a 0% cutoff still shortlists one, never nobody
    (0, 25.0, 0),         # no candidates, no shortlist
])
def test_shortlist_size(count, cutoff, expected):
    assert shortlist_size(count, cutoff) == expected


# --- Layer 2: the call ---------------------------------------------------------

def test_requirements_and_keywords_go_in_the_instructions_candidate_in_the_input():
    with patch("screening.stage2_ranking.matching.call_structured",
               return_value=reply()) as mock:
        match_candidate(JD, PYTHON_DEV, ["Python", "Kubernetes"])

    instructions, input_text, schema_name, schema = mock.call_args.args
    assert "Python: Backend development in Python" in instructions    # name AND description
    assert "Kubernetes" in instructions
    assert input_text.startswith("Candidate: ")
    assert schema_name == "candidate_match"
    assert schema is MATCH_SCHEMA


def test_instructions_are_identical_across_candidates_for_prompt_caching():
    """The JD and the keyword list are fixed for a run; only the candidate varies. If
    candidate data ever leaked into `instructions`, the cached prefix would miss on
    every single call and nothing else would look wrong."""
    calls = []

    with patch("screening.stage2_ranking.matching.call_structured",
               side_effect=lambda *a, **k: calls.append(a) or reply()):
        match_candidate(JD, PYTHON_DEV, ["Python"])
        match_candidate(JD, DESIGNER, ["Python"])

    assert calls[0][0] == calls[1][0]
    assert calls[0][1] != calls[1][1]


def test_instructions_forbid_inferring_and_demand_the_keyword_verbatim():
    with patch("screening.stage2_ranking.matching.call_structured",
               return_value=reply()) as mock:
        match_candidate(JD, PYTHON_DEV, ["Python"])

    instructions = mock.call_args.args[0]
    assert "never assume" in instructions
    assert "never infer" in instructions
    assert "EXACTLY as written" in instructions


def test_an_empty_keyword_list_does_not_invite_the_model_to_invent_keywords():
    """Handing a model an empty list to reason about invites it to fill one in. The
    fit score is still needed, so the call still happens."""
    with patch("screening.stage2_ranking.matching.call_structured",
               return_value=reply()) as mock:
        match_candidate(JD, PYTHON_DEV, [])

    instructions = mock.call_args.args[0]
    assert "Do not invent keywords" in instructions
    assert "Keywords: []" not in instructions
    assert mock.call_count == 1


def test_no_pii_reaches_the_matching_model():
    """Stage 2 gets the anonymized candidate. Nothing downstream re-adds names, and
    unlike the old ranker there is no constructed summary implicitly filtering them."""
    with patch("screening.stage2_ranking.matching.call_structured",
               return_value=reply()) as mock:
        match_candidate(JD, PYTHON_DEV, ["Python"])

    sent = mock.call_args.args[1]
    assert "full_name" not in sent and "email" not in sent and "phone" not in sent


def test_match_candidate_returns_the_model_payload_unmodified():
    """No arithmetic in the call layer — that all lives in ranking.py."""
    payload = reply(fit=73, matches=[found("Python")])

    with patch("screening.stage2_ranking.matching.call_structured", return_value=payload):
        assert match_candidate(JD, PYTHON_DEV, ["Python"]) == payload


# --- Layer 3: resolving the model's keyword verdicts ---------------------------

def test_resolution_keeps_the_recruiters_spelling_and_order():
    resolved = resolve_keyword_matches([found("python"), missing("django")], ["Django", "Python"])

    assert [m["keyword"] for m in resolved] == ["Django", "Python"]
    assert [m["matched"] for m in resolved] == [False, True]


def test_keyword_names_match_back_case_and_whitespace_insensitively():
    """Models recase freely. Treating "python" as an unrecognised name would silently
    lower the score and change who gets shortlisted."""
    resolved = resolve_keyword_matches([found("  POSTGRESQL ")], ["PostgreSQL"])

    assert resolved[0]["matched"] is True


def test_a_keyword_the_model_invented_is_dropped():
    resolved = resolve_keyword_matches([found("Python"), found("Rust")], ["Python"])

    assert [m["keyword"] for m in resolved] == ["Python"]


def test_a_keyword_the_model_skipped_counts_as_absent_and_says_so():
    resolved = resolve_keyword_matches([found("Python")], ["Python", "Terraform"])

    assert resolved[1] == {"keyword": "Terraform", "matched": False,
                           "evidence": "Not reported by the model."}


# --- Layer 3: the scores -------------------------------------------------------

def test_fit_score_is_rescaled_from_0_100_to_0_1(scripted_model):
    scripted_model(reply(fit=85))

    assert score_candidate_match(JD, PYTHON_DEV, [])["semantic_score"] == 0.85


@pytest.mark.parametrize("fit, expected", [(150, 1.0), (-10, 0.0)])
def test_a_fit_score_outside_the_schema_bounds_is_clamped(scripted_model, fit, expected):
    """Strict mode enforces 0-100, but a gateway may not. An out-of-range fit would push
    final_score outside [0, 1] and corrupt the whole ranking, not just one row."""
    scripted_model(reply(fit=fit))

    assert score_candidate_match(JD, PYTHON_DEV, [])["semantic_score"] == expected


def test_keyword_score_is_the_fraction_the_model_found(scripted_model):
    scripted_model(reply(matches=[found("Python"), missing("Kubernetes")]))

    breakdown = score_candidate_match(JD, PYTHON_DEV, ["Python", "Kubernetes"])

    assert breakdown["keyword_score"] == 0.5


def test_a_keyword_returned_twice_cannot_push_the_score_above_one(scripted_model):
    """The denominator is the recruiter's list, so a duplicated verdict must not
    inflate the numerator — that would put final_score above 1 and outrank everyone."""
    scripted_model(reply(matches=[found("Python"), found("Python")]))

    assert score_candidate_match(JD, PYTHON_DEV, ["Python"])["keyword_score"] == 1.0


def test_keyword_score_with_no_keywords_is_zero_not_a_crash(scripted_model):
    """A run with an empty keyword list still ranks — on job fit alone, at half weight.
    Worth knowing: clearing every keyword halves everyone's score rather than disabling
    the keyword half."""
    scripted_model(reply(fit=80))

    breakdown = score_candidate_match(JD, PYTHON_DEV, [])

    assert breakdown["keyword_score"] == 0.0
    assert breakdown["final_score"] == pytest.approx(0.4)


def test_final_score_blends_fifty_fifty(scripted_model):
    scripted_model(reply(fit=80, matches=[found("Python"), found("Django")]))

    breakdown = score_candidate_match(JD, PYTHON_DEV, ["Python", "Django"])

    assert breakdown["final_score"] == pytest.approx(0.5 * 0.8 + 0.5 * 1.0)


def test_the_keyword_half_can_overturn_the_fit_half(scripted_model):
    """This is why both halves exist: exact-term precision must be able to outrank a
    candidate the model simply likes the look of."""
    keywords = ["Python", "Django", "PostgreSQL"]
    scripted_model(
        reply(fit=30, matches=[found(k) for k in keywords]),      # weak fit, every keyword
        reply(fit=95, matches=[missing(k) for k in keywords]),    # strong fit, none of them
    )

    scores = rank_candidates(JD, [PYTHON_DEV, DESIGNER], keywords)

    assert scores[0] > scores[1]


def test_the_model_reasoning_and_evidence_survive_into_the_breakdown(scripted_model):
    """The stored reasoning is what makes a given ranking explainable months later."""
    scripted_model(reply(fit=70, matches=[found("Python", "Built Python APIs")],
                         reasoning="Six years of backend work."))

    breakdown = score_candidate_match(JD, PYTHON_DEV, ["Python"])

    assert breakdown["fit_reasoning"] == "Six years of backend work."
    assert breakdown["keyword_matches"][0]["evidence"] == "Built Python APIs"


# --- Layer 3: across a field of candidates -------------------------------------

def test_score_breakdown_makes_one_call_per_candidate_in_input_order(scripted_model):
    """Returns scores positionally, NOT sorted — the caller pairs them with candidates.
    A sort here would silently mis-attribute every score."""
    calls = scripted_model(reply(fit=10), reply(fit=90))

    breakdown = score_breakdown(JD, [PYTHON_DEV, DESIGNER], [])

    assert len(calls) == 2 and calls == [PYTHON_DEV, DESIGNER]
    assert breakdown[0]["final_score"] < breakdown[1]["final_score"]


def test_score_breakdown_on_an_empty_candidate_list_calls_nothing(scripted_model):
    calls = scripted_model(reply())

    assert score_breakdown(JD, [], ["Python"]) == []
    assert calls == []


def test_rank_candidates_agrees_with_score_breakdown(scripted_model):
    scripted_model(reply(fit=80, matches=[found("Python")]), reply(fit=40))

    breakdown = score_breakdown(JD, [PYTHON_DEV, DESIGNER], ["Python"])
    scripted_model(reply(fit=80, matches=[found("Python")]), reply(fit=40))
    combined = rank_candidates(JD, [PYTHON_DEV, DESIGNER], ["Python"])

    assert [b["final_score"] for b in breakdown] == pytest.approx(combined)
