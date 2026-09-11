"""Stage 1 — the hard filter.

The LLM judges each candidate's anonymized JSON against the JD's must-have
requirements directly, so requirements written in plain language work without a
taxonomy or a matching layer. The cost is strict reproducibility; the mitigations are
the mandatory evidence quote on every verdict and the fact that Python, not the model,
decides whether the candidate passed.
"""

from screening.common.llm import call_structured
from screening.stage1_hard_filter.schema import HARD_FILTER_SCHEMA


def select_must_have(requirements: list[dict]) -> list[dict]:
    """Stage 1 gates on must_have only. nice_to_have and inferred are Stage 3's job."""
    return [r for r in requirements if r.get("type") == "must_have"]


def resolve_evaluations(model_evaluations: list[dict], must_have_requirements: list[dict]) -> list[dict]:
    """One entry per must-have that was asked about, in the JD's order.

    `requirement` is free text in the schema, so the model can reword one, invent one,
    or leave one out. A requirement it did not report counts as NOT met: a hard
    requirement is met only on stated evidence, and counting a silent response as a
    pass would let a candidate through a gate that checked nothing.

    Names are matched case-insensitively and the JD's own wording is what gets stored,
    so the review page shows the requirement as the recruiter wrote it.
    """
    reported = {}
    for evaluation in model_evaluations:
        # First mention wins: the list was asked for in order, so a later duplicate is
        # a repeat rather than a correction.
        reported.setdefault((evaluation.get("requirement") or "").strip().lower(), evaluation)

    resolved = []
    for requirement in must_have_requirements:
        name = requirement["requirement"]
        evaluation = reported.get(name.strip().lower())
        if evaluation is None:
            resolved.append({
                "requirement": name,
                "met": False,
                "evidence": "",
                "reason": "The model did not evaluate this requirement, so it cannot be "
                          "treated as met. Re-run the stage, or reinstate the candidate "
                          "if you can confirm it yourself.",
            })
        else:
            resolved.append({
                "requirement": name,
                "met": bool(evaluation.get("met")),
                "evidence": evaluation.get("evidence", ""),
                "reason": evaluation.get("reason", ""),
            })
    return resolved


def evaluate_hard_filters(anon_candidate: dict, must_have_requirements: list[dict]) -> dict:
    """Whether one candidate clears every must-have, with the reasoning behind each."""
    # A JD with no must-haves has no hard gate, so everyone passes. Guarding here rather
    # than calling the model with an empty list: an empty requirement list invites it to
    # invent requirements to evaluate, and it would be a wasted call per candidate.
    if not must_have_requirements:
        return {"passed": True, "evaluations": []}

    # instructions is identical for every candidate in the run, so it is the repeated
    # prefix that gets cached automatically. Only the candidate varies per call.
    instructions = f"""Evaluate whether this candidate meets each must-have requirement,
    based ONLY on what's explicitly stated in the candidate data — never assume or infer.
    When met=true, quote the evidence. When met=false, state what's missing.
    Must-have requirements: {[r["requirement"] for r in must_have_requirements]}"""
    result = call_structured(instructions, f"Candidate: {anon_candidate}", "hard_filter_result", HARD_FILTER_SCHEMA)

    evaluations = resolve_evaluations(result["evaluations"], must_have_requirements)
    # The AND is ours, not the model's — the schema has no "passed" field precisely so
    # the model cannot overrule it.
    return {"passed": all(e["met"] for e in evaluations), "evaluations": evaluations}
