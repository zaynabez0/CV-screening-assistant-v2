"""Stage 2 tests against the REAL BAAI/bge-m3 model.

These are the tests that answer "does the semantic half actually work" — the stub
tests in test_stage2_ranking.py pin the arithmetic, but only real embeddings can show
that a paraphrased CV outranks an unrelated one.

They skip automatically when the model cannot be loaded (no network, no cache), so the
suite stays green in a sandbox. On a developer machine with HuggingFace reachable they
run for real; the first run pays a one-off ~2 GB download.

Data is kept deliberately tiny — a handful of short strings — so the encode cost is
seconds, not minutes.
"""

import pytest

from screening.stage2_ranking.embeddings import semantic_scores
from screening.stage2_ranking.ranking import rank_candidates

pytestmark = pytest.mark.needs_embedding_model

JD = [
    {"requirement": "Python backend development",
     "description": "Building server-side services in Python", "type": "must_have"},
]

BACKEND_DEV = {
    "skills": ["Python", "Django"], "total_years_experience": 5,
    "experience": [{"title": "Backend Engineer", "company": "Kite",
                    "description": "Built server-side APIs in Python."}],
}
PASTRY_CHEF = {
    "skills": ["Pastry", "Cake decorating"], "total_years_experience": 5,
    "experience": [{"title": "Pastry Chef", "company": "Boulangerie",
                    "description": "Prepared desserts and viennoiseries."}],
}


def test_a_relevant_candidate_outranks_an_irrelevant_one(embedding_model):
    scores = semantic_scores(
        "Python backend development: building server-side services",
        ["Skills: Python, Django\nBackend Engineer: server-side APIs in Python",
         "Skills: Pastry, Cake decorating\nPastry Chef: desserts and viennoiseries"],
    )

    assert scores[0] > scores[1]


def test_paraphrase_scores_close_to_the_literal_phrasing(embedding_model):
    """The reason the semantic half exists: a CV that never says "Python" verbatim but
    describes the same work should still rank near one that does. Keyword matching
    alone cannot do this."""
    literal, paraphrase, unrelated = semantic_scores(
        "Python backend development",
        ["Python backend development",
         "Server-side programming for web services",
         "Cake decorating and pastry preparation"],
    )

    assert paraphrase > unrelated
    assert literal > paraphrase                      # literal still wins, as it should


def test_scores_are_bounded_like_cosine_similarity(embedding_model):
    scores = semantic_scores("Python backend", ["Python backend", "Pastry chef"])

    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_french_and_english_describe_the_same_role_similarly(embedding_model):
    """BGE-M3 was chosen for multilingual coverage — French and English CVs land in one
    ranking. If this fails, mixed-language batches are being ranked by language rather
    than by content."""
    english, french, unrelated = semantic_scores(
        "Backend software engineer building web services",
        ["Backend software engineer building web services",
         "Ingénieur logiciel backend développant des services web",
         "Chef pâtissier préparant des desserts"],
    )

    assert french > unrelated
    assert french > 0.5                              # genuinely close, not merely ahead


def test_rank_candidates_end_to_end_with_the_real_model(embedding_model):
    scores = rank_candidates(JD, [BACKEND_DEV, PASTRY_CHEF], ["Python", "Django"])

    assert scores[0] > scores[1]
    assert all(isinstance(s, float) for s in scores)
