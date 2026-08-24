"""The Stage 3 scoring call."""

from screening.common.llm import call_structured
from screening.stage3_scoring.schema import SCORE_SCHEMA


def score_candidate(anon_candidate: dict, rubric: list) -> list[dict]:
    # instructions is identical across every candidate scored against this rubric —
    # kept as the repeated prefix for automatic caching, same principle as Stage 1.
    instructions = f"""Score against each criterion. MUST quote evidence.
    If no evidence, score low and confidence=low — never infer.
    Criteria and scales: {[{"criterion": r["requirement"], "scale": r["scale"]} for r in rubric]}"""
    result = call_structured(instructions, f"Candidate: {anon_candidate}", "candidate_scores", SCORE_SCHEMA)
    return result["criterion_scores"]
