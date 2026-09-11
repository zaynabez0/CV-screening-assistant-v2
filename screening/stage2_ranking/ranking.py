"""Combined ranking: a fixed 50/50 blend of job fit and keyword match.

Both halves come from one LLM call per candidate (`matching.py`), but they stay two
separate numbers: the review checkpoint shows both, and a recruiter who cannot see
whether a candidate ranked high on overall fit or on keyword hits has no basis for
editing the keyword list — which is the whole point of the checkpoint.

The blend is not user-tunable: the levers at this checkpoint are the keyword list and
the cutoff, which keeps the page to two controls instead of three.

The model is asked for a 0-100 fit score and a boolean per keyword; Python does the
division, the counting and the blend. Arithmetic done inside the model is arithmetic
nobody can test.
"""

import math

from screening.stage2_ranking.matching import match_candidate


def resolve_keyword_matches(model_matches: list[dict], keywords: list[str]) -> list[dict]:
    """One entry per keyword the recruiter actually asked for, in their order.

    `keyword` is free text in the schema, so the model can reword a keyword, invent one,
    or skip one. Names are matched back case-insensitively; anything invented is dropped,
    and any keyword left unreported counts as unmatched rather than silently shrinking
    the denominator. The recruiter's own spelling is what gets stored, so the review page
    shows the list they typed.
    """
    reported = {}
    for match in model_matches:
        # First mention wins: the model was asked for the list in order, so a later
        # duplicate is a repeat rather than a correction.
        reported.setdefault((match.get("keyword") or "").strip().lower(), match)

    resolved = []
    for keyword in keywords:
        match = reported.get(keyword.strip().lower())
        if match is None:
            resolved.append({"keyword": keyword, "matched": False,
                             "evidence": "Not reported by the model."})
        else:
            resolved.append({"keyword": keyword, "matched": bool(match.get("matched")),
                             "evidence": match.get("evidence", "")})
    return resolved


def keyword_hit_rate(resolved_matches: list[dict], keywords: list[str]) -> float:
    """The fraction of the recruiter's keywords the candidate demonstrably has."""
    if not keywords:
        return 0.0
    return sum(1 for m in resolved_matches if m["matched"]) / len(keywords)


def score_candidate_match(jd_requirements: list[dict], candidate: dict, keywords: list[str]) -> dict:
    """One candidate's Stage 2 scores, plus the model's reasoning for the audit trail."""
    result = match_candidate(jd_requirements, candidate, keywords)

    # 0-100 from the model because models are steadier on that scale; 0-1 here because
    # that is the range the blend and the review page use. Clamped because minimum and
    # maximum are enforced by OpenAI's strict mode but not necessarily by a gateway, and
    # an out-of-range fit would push final_score outside [0, 1] and corrupt the whole
    # ranking rather than one row.
    semantic_score = min(100, max(0, result["fit_score"])) / 100
    resolved = resolve_keyword_matches(result["keyword_matches"], keywords)
    keyword_score = keyword_hit_rate(resolved, keywords)

    return {
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
        "final_score": 0.5 * semantic_score + 0.5 * keyword_score,
        "fit_reasoning": result["fit_reasoning"],
        "keyword_matches": resolved,
    }


def shortlist_size(candidate_count: int, cutoff_percent: float) -> int:
    """How many top-ranked candidates proceed to Stage 3.

    Rounds up and never returns zero for a non-empty list: a cutoff that silently
    shortlists nobody would end the run with no explanation, and the recruiter can
    always set the cutoff lower deliberately.
    """
    if candidate_count <= 0:
        return 0
    return max(1, min(candidate_count, math.ceil(candidate_count * cutoff_percent / 100)))
