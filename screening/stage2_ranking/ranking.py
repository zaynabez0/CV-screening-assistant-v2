"""Combined ranking: a fixed 50/50 blend of semantic similarity and keyword match.

Not user-tunable in v1 — the user's lever at this checkpoint is the keyword list and
the cutoff, which keeps the UI to two controls instead of three.
"""

import math

from screening.stage2_ranking.embeddings import semantic_scores
from screening.stage2_ranking.keywords import keyword_match_score
from screening.stage2_ranking.text_builders import build_candidate_text, build_jd_text


def rank_candidates(jd_requirements: list[dict], candidates: list[dict], keywords: list[str]) -> list[float]:
    jd_text = build_jd_text(jd_requirements)
    candidate_texts = [build_candidate_text(c) for c in candidates]
    sem_scores = semantic_scores(jd_text, candidate_texts)
    kw_scores = [keyword_match_score(c, keywords) for c in candidates]
    return [0.5 * s + 0.5 * k for s, k in zip(sem_scores, kw_scores)]


def score_breakdown(jd_requirements: list[dict], candidates: list[dict], keywords: list[str]) -> list[dict]:
    """The same computation, keeping both halves visible.

    The review checkpoint shows the full ranked list, and a recruiter who cannot see
    whether a candidate ranked high on semantic similarity or on keyword hits has no
    basis for editing the keyword list — which is the whole point of the checkpoint.
    """
    jd_text = build_jd_text(jd_requirements)
    candidate_texts = [build_candidate_text(c) for c in candidates]
    sem_scores = semantic_scores(jd_text, candidate_texts)
    kw_scores = [keyword_match_score(c, keywords) for c in candidates]
    return [
        {"semantic_score": s, "keyword_score": k, "final_score": 0.5 * s + 0.5 * k}
        for s, k in zip(sem_scores, kw_scores)
    ]


def shortlist_size(candidate_count: int, cutoff_percent: float) -> int:
    """How many top-ranked candidates proceed to Stage 3.

    Rounds up and never returns zero for a non-empty list: a cutoff that silently
    shortlists nobody would end the run with no explanation, and the recruiter can
    always set the cutoff lower deliberately.
    """
    if candidate_count <= 0:
        return 0
    return max(1, min(candidate_count, math.ceil(candidate_count * cutoff_percent / 100)))
