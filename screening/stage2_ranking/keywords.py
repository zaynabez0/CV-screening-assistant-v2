"""Keyword matching — exact/near-exact, against a user-editable list.

The embedding half catches paraphrase; this half catches exact-term precision, which
is what a recruiter means by "must actually say Kubernetes". Fuzzy at ratio >= 90 so
"Postgres"/"PostgreSQL" style variants match but unrelated terms do not.
"""

from rapidfuzz import fuzz


def keyword_match_score(candidate: dict, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    candidate_skills = [s.lower() for s in candidate.get("skills", [])]
    matched = sum(
        1 for kw in keywords
        if any(fuzz.ratio(kw.lower(), s) >= 90 for s in candidate_skills)
    )
    return matched / len(keywords)


def seed_keywords(requirements: list[dict]) -> list[str]:
    """Pre-fill the keyword list from the JD's must_have + nice_to_have requirement names.

    A starting point only — the user adds and removes freely at the checkpoint, and
    that editing is the primary lever they have over Stage 2. `inferred` requirements
    are excluded: they are not stated in the JD, so their wording is the model's
    phrasing rather than the employer's, and it makes a poor literal search term.
    """
    seen, seeded = set(), []
    for requirement in requirements:
        if requirement.get("type") not in ("must_have", "nice_to_have"):
            continue
        name = (requirement.get("requirement") or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            seeded.append(name)
    return seeded
