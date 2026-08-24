"""Tests for Stage 2 ranking.

Three layers, on purpose:

1. Pure functions (text builders, keyword_match_score, seed_keywords, shortlist_size)
   — tested directly, no model involved.
2. `semantic_scores` and `rank_candidates` blend arithmetic — tested against a stub
   model with hand-chosen vectors, so the cosine maths and the 50/50 blend are pinned
   deterministically rather than sampled from real embeddings.
3. Real BAAI/bge-m3 behaviour — marked `needs_embedding_model`, skipped automatically
   when the model isn't downloadable, so the suite stays green in a sandbox and these
   still run for real on a developer machine.

Layer 2 exists because a real-embedding assertion like "score > 0.5" is a weak test:
it passes for a broken blend. The stub makes the arithmetic exact.
"""

import numpy as np
import pytest

from screening.stage2_ranking import embeddings as embeddings_module
from screening.stage2_ranking import ranking as ranking_module
from screening.stage2_ranking.embeddings import semantic_scores
from screening.stage2_ranking.keywords import keyword_match_score, seed_keywords
from screening.stage2_ranking.ranking import rank_candidates, score_breakdown, shortlist_size
from screening.stage2_ranking.text_builders import build_candidate_text, build_jd_text


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


class StubModel:
    """Deterministic stand-in for BGE-M3: returns pre-set unit vectors by text."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, texts, normalize_embeddings=False):
        self.calls.append({"texts": texts, "normalize_embeddings": normalize_embeddings})
        if isinstance(texts, str):
            return np.array(self.vectors[texts], dtype=float)
        return np.array([self.vectors[t] for t in texts], dtype=float)


@pytest.fixture
def stub_model(monkeypatch):
    def install(vectors):
        model = StubModel(vectors)
        monkeypatch.setattr(embeddings_module, "get_embedding_model", lambda: model)
        return model
    return install


# --- Layer 1: text builders ---------------------------------------------------

def test_build_candidate_text_includes_skills_experience_and_roles():
    text = build_candidate_text(PYTHON_DEV)

    assert "Skills: Python, Django, PostgreSQL" in text
    assert "Total experience: 6 years" in text
    assert "Backend Engineer at Kite: Python APIs." in text


def test_build_candidate_text_survives_a_sparse_candidate():
    """A CV with nothing extracted must not crash the ranker — it should just rank low."""
    text = build_candidate_text({})

    assert "Skills: " in text
    assert "Total experience: 0 years" in text


def test_build_candidate_text_excludes_pii_fields():
    """Stage 2 sees the anonymized candidate; the builder must not reintroduce names."""
    text = build_candidate_text({**PYTHON_DEV, "full_name": "Amina Benali",
                                 "email": "a@b.c", "phone": "+212"})

    assert "Amina" not in text and "a@b.c" not in text and "+212" not in text


def test_build_jd_text_pairs_each_requirement_with_its_description():
    assert build_jd_text(JD) == (
        "Python: Backend development in Python\nPostgreSQL: Relational databases"
    )


# --- Layer 1: keyword_match_score ---------------------------------------------

def test_keyword_score_is_the_fraction_of_keywords_matched():
    assert keyword_match_score(PYTHON_DEV, ["Python", "Django"]) == 1.0
    assert keyword_match_score(PYTHON_DEV, ["Python", "Kubernetes"]) == 0.5
    assert keyword_match_score(PYTHON_DEV, ["Kubernetes", "Terraform"]) == 0.0


def test_keyword_score_is_case_insensitive():
    assert keyword_match_score(PYTHON_DEV, ["PYTHON", "django"]) == 1.0


def test_keyword_score_with_no_keywords_is_zero_not_a_crash():
    """A run with an empty keyword list still ranks — on semantic similarity alone,
    at half weight. Worth knowing: clearing every keyword halves everyone's score
    rather than disabling the keyword half."""
    assert keyword_match_score(PYTHON_DEV, []) == 0.0


def test_keyword_score_matches_near_spellings_but_not_unrelated_terms():
    near = candidate(skills=["PostgreSQL"])
    assert keyword_match_score(near, ["PostgreSQL"]) == 1.0
    assert keyword_match_score(near, ["Postgre SQL"]) == 1.0      # ratio >= 90
    assert keyword_match_score(near, ["MySQL"]) == 0.0            # different product


def test_keyword_score_only_looks_at_the_skills_field():
    """Known limitation, pinned deliberately: a skill demonstrated in a job description
    but absent from `skills` scores zero. Recruiters editing keywords should know the
    match is against the extracted skills list, not the whole CV.
    """
    experience_only = candidate(
        skills=[], experience=[{"title": "SRE", "company": "X", "description": "Ran Kubernetes clusters."}]
    )

    assert keyword_match_score(experience_only, ["Kubernetes"]) == 0.0


def test_a_candidate_with_no_skills_scores_zero_rather_than_erroring():
    assert keyword_match_score({}, ["Python"]) == 0.0


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


# --- Layer 2: semantic_scores against a stub model -----------------------------

def test_semantic_scores_are_dot_products_of_normalized_vectors(stub_model):
    """Identical vector -> 1.0, orthogonal -> 0.0, opposite -> -1.0."""
    stub_model({
        "jd": [1.0, 0.0],
        "same": [1.0, 0.0],
        "orthogonal": [0.0, 1.0],
        "opposite": [-1.0, 0.0],
    })

    scores = semantic_scores("jd", ["same", "orthogonal", "opposite"])

    assert scores == pytest.approx([1.0, 0.0, -1.0])


def test_semantic_scores_asks_the_model_to_normalize(stub_model):
    """The dot-product-as-cosine shortcut is only valid on normalized vectors. If this
    flag were dropped, scores would silently scale with text length."""
    model = stub_model({"jd": [1.0, 0.0], "c": [1.0, 0.0]})

    semantic_scores("jd", ["c"])

    assert all(call["normalize_embeddings"] is True for call in model.calls)


def test_semantic_scores_returns_plain_floats_not_numpy(stub_model):
    """The result is stored in a JSONField — numpy floats are not JSON serialisable."""
    stub_model({"jd": [1.0, 0.0], "c": [0.6, 0.8]})

    scores = semantic_scores("jd", ["c"])

    assert isinstance(scores, list) and isinstance(scores[0], float)


def test_semantic_scores_on_an_empty_candidate_list_skips_the_model(stub_model):
    model = stub_model({})

    assert semantic_scores("jd", []) == []
    assert model.calls == []


# --- Layer 2: rank_candidates blend arithmetic ---------------------------------

@pytest.fixture
def fixed_semantics(monkeypatch):
    """Pin semantic_scores so the blend can be asserted exactly."""
    def install(scores):
        monkeypatch.setattr(ranking_module, "semantic_scores", lambda jd, texts: list(scores))
    return install


def test_rank_candidates_blends_fifty_fifty(fixed_semantics):
    fixed_semantics([0.8, 0.4])

    scores = rank_candidates(JD, [PYTHON_DEV, DESIGNER], ["Python", "Django"])

    # PYTHON_DEV matches both keywords (1.0); DESIGNER matches neither (0.0).
    assert scores == pytest.approx([0.5 * 0.8 + 0.5 * 1.0, 0.5 * 0.4 + 0.5 * 0.0])


def test_keyword_half_can_overturn_the_semantic_half(fixed_semantics):
    """This is why both halves exist: exact-term precision must be able to outrank a
    semantically fluent CV that lacks the actual skill."""
    fixed_semantics([0.30, 0.95])                    # DESIGNER scores far higher semantically

    scores = rank_candidates(JD, [PYTHON_DEV, DESIGNER], ["Python", "Django", "PostgreSQL"])

    assert scores[0] > scores[1]                     # keyword hits pull PYTHON_DEV ahead


def test_rank_candidates_preserves_input_order(fixed_semantics):
    """Returns scores positionally, NOT sorted — the caller pairs them with candidates.
    A sort here would silently mis-attribute every score."""
    fixed_semantics([0.1, 0.9])

    scores = rank_candidates(JD, [PYTHON_DEV, DESIGNER], [])

    assert scores[0] < scores[1]                     # order follows the input, not the value


def test_rank_candidates_with_no_keywords_halves_every_score(fixed_semantics):
    fixed_semantics([0.8, 0.4])

    scores = rank_candidates(JD, [PYTHON_DEV, DESIGNER], [])

    assert scores == pytest.approx([0.4, 0.2])


def test_rank_candidates_on_an_empty_candidate_list(fixed_semantics):
    fixed_semantics([])

    assert rank_candidates(JD, [], ["Python"]) == []


def test_score_breakdown_keeps_both_halves_and_agrees_with_rank_candidates(fixed_semantics):
    fixed_semantics([0.8, 0.4])

    breakdown = score_breakdown(JD, [PYTHON_DEV, DESIGNER], ["Python", "Django"])
    combined = rank_candidates(JD, [PYTHON_DEV, DESIGNER], ["Python", "Django"])

    assert [b["final_score"] for b in breakdown] == pytest.approx(combined)
    assert breakdown[0]["semantic_score"] == 0.8
    assert breakdown[0]["keyword_score"] == 1.0
