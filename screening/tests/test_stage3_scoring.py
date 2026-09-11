"""Tests for Stage 3: rubric building, weight adjustment, scoring, aggregation.

The LLM is mocked throughout. The aggregation maths is asserted against hand-computed
numbers rather than against the implementation, so a change in the formula shows up as
a failing arithmetic assertion, not a silently different score.
"""

from unittest.mock import patch

import pytest

from screening.stage3_scoring.aggregation import aggregate_score
from screening.stage3_scoring.rubric import apply_user_weights, build_rubric, rubric_is_empty
from screening.stage3_scoring.schema import SCORE_SCHEMA
from screening.stage3_scoring.scoring import score_candidate

MUST_HAVE = {"requirement": "5+ years Python", "description": "Professional Python",
             "type": "must_have", "scale": "", "suggested_weight": 0}
KUBERNETES = {"requirement": "Kubernetes", "description": "Container orchestration",
              "type": "nice_to_have", "scale": "5=expert, 3=some, 1=none", "suggested_weight": 3}
LEADERSHIP = {"requirement": "Team leadership", "description": "Implied by seniority",
              "type": "inferred", "scale": "5=led teams, 3=mentored, 1=none", "suggested_weight": 1}

JD = [MUST_HAVE, KUBERNETES, LEADERSHIP]

ANON_CANDIDATE = {
    "skills": ["Python", "Kubernetes"], "total_years_experience": 6,
    "experience": [{"title": "Staff Engineer", "company": "Kite", "start_date": "2019-01",
                    "end_date": None, "description": "Ran the platform team.",
                    "evidence_quote": "Led a team of six engineers"}],
    "education": [], "certifications": [], "languages": [],
    "extraction_confidence": "high", "extraction_notes": "",
}


def scores(kubernetes=4, leadership=2, confidence="high"):
    return [
        {"criterion": "Kubernetes", "score": kubernetes,
         "evidence_quote": "Ran the platform team", "reasoning": "Operated clusters.",
         "confidence": confidence},
        {"criterion": "Team leadership", "score": leadership,
         "evidence_quote": "Led a team of six engineers", "reasoning": "Led a small team.",
         "confidence": confidence},
    ]


# --- build_rubric --------------------------------------------------------------

def test_rubric_contains_nice_to_have_and_inferred_but_not_must_have():
    """Must-haves were already gated pass/fail in Stage 1 — scoring them again would
    count the same requirement twice."""
    rubric = build_rubric(JD)

    assert [c["requirement"] for c in rubric] == ["Kubernetes", "Team leadership"]


def test_rubric_normalizes_suggested_weights_to_sum_to_one():
    rubric = build_rubric(JD)

    # raw weights 3 and 1 -> 0.75 and 0.25
    assert [c["suggested_weight"] for c in rubric] == pytest.approx([0.75, 0.25])
    assert sum(c["suggested_weight"] for c in rubric) == pytest.approx(1.0)


def test_rubric_leaves_zero_total_weights_alone_rather_than_dividing_by_zero():
    zeroed = [{**KUBERNETES, "suggested_weight": 0}, {**LEADERSHIP, "suggested_weight": 0}]

    rubric = build_rubric(zeroed)

    assert [c["suggested_weight"] for c in rubric] == [0, 0]


def test_rubric_keeps_the_scale_text_each_criterion_is_judged_against():
    rubric = build_rubric(JD)

    assert rubric[0]["scale"] == "5=expert, 3=some, 1=none"


# --- The empty-rubric edge case --------------------------------------------------

def test_a_jd_of_only_must_haves_produces_an_empty_rubric():
    assert build_rubric([MUST_HAVE]) == []
    assert rubric_is_empty([MUST_HAVE]) is True


def test_rubric_is_empty_is_false_when_any_scorable_criterion_exists():
    assert rubric_is_empty(JD) is False
    assert rubric_is_empty([MUST_HAVE, LEADERSHIP]) is False


def test_aggregating_against_an_empty_rubric_returns_zero_not_a_crash():
    """The specified handling is to skip Stage 3 entirely, but the maths must not
    explode if an empty rubric ever reaches it."""
    result = aggregate_score([], [])

    assert result["final_score"] == 0.0
    assert result["criterion_breakdown"] == []


# --- apply_user_weights ------------------------------------------------------------

def test_user_weights_override_the_suggested_ones():
    rubric = build_rubric(JD)

    weighted = apply_user_weights(rubric, {"Kubernetes": 0.5, "Team leadership": 0.5})

    assert [c["weight"] for c in weighted] == pytest.approx([0.5, 0.5])


def test_criteria_the_user_did_not_touch_keep_their_suggested_weight():
    rubric = build_rubric(JD)

    weighted = apply_user_weights(rubric, {"Kubernetes": 0.75})

    assert weighted[0]["weight"] == pytest.approx(0.75)
    assert weighted[1]["weight"] == pytest.approx(0.25)      # untouched suggestion


def test_weights_that_do_not_sum_to_one_are_normalized():
    """A recruiter thinking in "out of 10" gets a valid rubric rather than an error."""
    rubric = build_rubric(JD)

    weighted = apply_user_weights(rubric, {"Kubernetes": 8, "Team leadership": 2})

    assert [c["weight"] for c in weighted] == pytest.approx([0.8, 0.2])
    assert sum(c["weight"] for c in weighted) == pytest.approx(1.0)


def test_weights_already_summing_to_one_are_left_alone():
    rubric = build_rubric(JD)

    weighted = apply_user_weights(rubric, {"Kubernetes": 0.6, "Team leadership": 0.4})

    assert [c["weight"] for c in weighted] == pytest.approx([0.6, 0.4])


def test_zeroing_every_weight_falls_back_to_equal_weights():
    """The checkpoint lets a user type every weight, and dividing by a zero total
    would be a ZeroDivisionError one keystroke away."""
    rubric = build_rubric(JD)

    weighted = apply_user_weights(rubric, {"Kubernetes": 0, "Team leadership": 0})

    assert [c["weight"] for c in weighted] == pytest.approx([0.5, 0.5])


def test_apply_user_weights_does_not_mutate_the_rubric():
    rubric = build_rubric(JD)

    apply_user_weights(rubric, {"Kubernetes": 0.9, "Team leadership": 0.1})

    assert "weight" not in rubric[0]


# --- score_candidate ----------------------------------------------------------------

def test_criteria_and_scales_go_in_the_instructions_candidate_in_the_input():
    rubric = build_rubric(JD)

    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": scores()}) as mock:
        score_candidate(ANON_CANDIDATE, rubric)

    instructions, input_text, schema_name, schema = mock.call_args.args
    assert "Kubernetes" in instructions
    assert "5=expert, 3=some, 1=none" in instructions       # the scale the model judges against
    assert input_text.startswith("Candidate: ")
    assert schema_name == "candidate_scores"
    assert schema is SCORE_SCHEMA


def test_instructions_are_identical_across_candidates_for_prompt_caching():
    rubric = build_rubric(JD)
    calls = []

    with patch("screening.stage3_scoring.scoring.call_structured",
               side_effect=lambda *a, **k: calls.append(a) or {"criterion_scores": scores()}):
        score_candidate(ANON_CANDIDATE, rubric)
        score_candidate({**ANON_CANDIDATE, "skills": ["Go"]}, rubric)

    assert calls[0][0] == calls[1][0]
    assert calls[0][1] != calls[1][1]


def test_instructions_demand_evidence_and_forbid_inferring():
    rubric = build_rubric(JD)

    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": scores()}) as mock:
        score_candidate(ANON_CANDIDATE, rubric)

    instructions = mock.call_args.args[0]
    assert "MUST quote evidence" in instructions
    assert "never infer" in instructions


def test_score_candidate_returns_the_model_list_unmodified():
    rubric = build_rubric(JD)
    payload = scores(kubernetes=5, leadership=1)

    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": payload}):
        assert score_candidate(ANON_CANDIDATE, rubric) == payload


def test_no_pii_reaches_the_scoring_model():
    rubric = build_rubric(JD)

    with patch("screening.stage3_scoring.scoring.call_structured",
               return_value={"criterion_scores": scores()}) as mock:
        score_candidate(ANON_CANDIDATE, rubric)

    sent = mock.call_args.args[1]
    assert "full_name" not in sent and "email" not in sent and "phone" not in sent


# --- aggregate_score: the arithmetic ---------------------------------------------------

def test_final_score_is_the_hand_computed_weighted_percentage():
    """Kubernetes 4 at weight 0.75, leadership 2 at weight 0.25:
         weighted_sum = 4*0.75 + 2*0.25 = 3.5
         max_possible = 5 * (0.75 + 0.25) = 5
         final        = 3.5 / 5 * 100 = 70.0
    """
    weighted = apply_user_weights(build_rubric(JD), {})

    result = aggregate_score(scores(kubernetes=4, leadership=2), weighted)

    assert result["final_score"] == 70.0


def test_a_perfect_candidate_scores_one_hundred():
    weighted = apply_user_weights(build_rubric(JD), {})

    assert aggregate_score(scores(5, 5), weighted)["final_score"] == 100.0


def test_the_lowest_possible_score_is_twenty_not_zero():
    """The scale starts at 1, not 0 — so a candidate with no evidence anywhere still
    scores 20%. Worth knowing before reading a shortlist: 20 means "nothing", not "poor"."""
    weighted = apply_user_weights(build_rubric(JD), {})

    assert aggregate_score(scores(1, 1), weighted)["final_score"] == 20.0


def test_weights_actually_change_the_outcome():
    rubric = build_rubric(JD)
    candidate_scores = scores(kubernetes=5, leadership=1)

    kubernetes_heavy = aggregate_score(candidate_scores, apply_user_weights(rubric, {"Kubernetes": 0.9, "Team leadership": 0.1}))
    leadership_heavy = aggregate_score(candidate_scores, apply_user_weights(rubric, {"Kubernetes": 0.1, "Team leadership": 0.9}))

    # (5*.9 + 1*.1)/5*100 = 4.6/5*100 = 92.0
    assert kubernetes_heavy["final_score"] == pytest.approx(92.0)
    # (5*.1 + 1*.9)/5*100 = 1.4/5*100 = 28.0 — the same candidate, 64 points apart
    assert leadership_heavy["final_score"] == pytest.approx(28.0)


def test_final_score_is_rounded_to_one_decimal():
    weighted = apply_user_weights(build_rubric(JD), {"Kubernetes": 1/3, "Team leadership": 2/3})

    result = aggregate_score(scores(4, 3), weighted)

    assert result["final_score"] == round(result["final_score"], 1)


def test_breakdown_and_flags_are_carried_through():
    weighted = apply_user_weights(build_rubric(JD), {})
    candidate_scores = scores(confidence="low")

    result = aggregate_score(candidate_scores, weighted)

    assert result["criterion_breakdown"] == candidate_scores      # full audit trail
    assert result["low_confidence_flags"] == ["Kubernetes", "Team leadership"]


def test_only_low_confidence_is_flagged():
    weighted = apply_user_weights(build_rubric(JD), {})
    mixed = [{**scores()[0], "confidence": "medium"}, {**scores()[1], "confidence": "low"}]

    assert aggregate_score(mixed, weighted)["low_confidence_flags"] == ["Team leadership"]


# --- aggregate_score: defending against bad model output --------------------------------

def test_a_criterion_the_model_invented_does_not_crash_the_run():
    """`criterion` is free text in SCORE_SCHEMA. A direct weight_map[...] lookup
    raises KeyError on a reworded name and takes down the whole batch."""
    weighted = apply_user_weights(build_rubric(JD), {})
    with_extra = scores() + [{"criterion": "Kubernetes administration", "score": 5,
                              "evidence_quote": "q", "reasoning": "r", "confidence": "high"}]

    result = aggregate_score(with_extra, weighted)

    assert result["final_score"] == 70.0                     # unchanged by the stray item
    assert result["unknown_criteria"] == ["Kubernetes administration"]
    assert len(result["criterion_breakdown"]) == 3           # still recorded in full


def test_a_criterion_the_model_skipped_is_reported_and_scores_nothing():
    """Missing evidence is the conservative reading: it contributes 0 to the weighted
    sum while still counting in the maximum."""
    weighted = apply_user_weights(build_rubric(JD), {})
    only_one = [scores()[0]]                                 # Kubernetes 4, leadership absent

    result = aggregate_score(only_one, weighted)

    assert result["unscored_criteria"] == ["Team leadership"]
    assert result["final_score"] == 60.0                     # 4*0.75 / 5 * 100
