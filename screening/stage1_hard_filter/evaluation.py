"""Stage 1 — hard filter. LLM-based, not a rule engine.

The LLM judges each candidate's anonymized JSON against the JD's must-have
requirements directly, so requirements phrased in natural language work without a
taxonomy or matching layer. The cost is strict reproducibility; the mitigations are
mandatory evidence quotes and the golden-set recall check.
"""

from screening.common.llm import call_structured
from screening.stage1_hard_filter.schema import HARD_FILTER_SCHEMA


def select_must_have(requirements: list[dict]) -> list[dict]:
    """Stage 1 gates on must_have only. nice_to_have and inferred are Stage 3's job."""
    return [r for r in requirements if r.get("type") == "must_have"]


def evaluate_hard_filters(anon_candidate: dict, must_have_requirements: list[dict]) -> dict:
    # A JD with no must-have requirements has no hard gate, so everyone passes. Guarding
    # here rather than calling the LLM with an empty list: `all([])` is True either way,
    # but an empty requirement list invites the model to invent requirements to evaluate,
    # and it would be a wasted API call per candidate. Deviation from CLAUDE.md, which
    # does not cover this case for Stage 1 (it does cover the equivalent for Stage 3).
    if not must_have_requirements:
        return {"passed": True, "evaluations": []}

    # instructions is identical across every candidate in the run — put it first so
    # it's the (automatically cached) repeated prefix; the candidate itself is the input.
    instructions = f"""Evaluate whether this candidate meets each must-have requirement,
    based ONLY on what's explicitly stated in the candidate data — never assume or infer.
    When met=true, quote the evidence. When met=false, state what's missing.
    Must-have requirements: {[r["requirement"] for r in must_have_requirements]}"""
    result = call_structured(instructions, f"Candidate: {anon_candidate}", "hard_filter_result", HARD_FILTER_SCHEMA)
    evaluations = result["evaluations"]
    return {"passed": all(e["met"] for e in evaluations), "evaluations": evaluations}
