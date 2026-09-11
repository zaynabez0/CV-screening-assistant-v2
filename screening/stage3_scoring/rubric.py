"""Rubric building and weight adjustment — pure Python, no LLM involved.

Criteria are exactly the JD's nice_to_have and inferred requirements. Must-haves are
excluded: they were already gated pass/fail in Stage 1, and scoring them again would
count the same requirement twice.
"""


def build_rubric(jd_requirements: list[dict]) -> list[dict]:
    criteria = [r for r in jd_requirements if r["type"] in ("nice_to_have", "inferred")]
    total = sum(c["suggested_weight"] for c in criteria)
    if total > 0:
        criteria = [{**c, "suggested_weight": c["suggested_weight"] / total} for c in criteria]
    return criteria


def apply_user_weights(rubric: list, user_weights: dict) -> list:
    updated = [{**c, "weight": user_weights.get(c["requirement"], c["suggested_weight"])} for c in rubric]
    total = sum(c["weight"] for c in updated)

    # The weight-adjustment checkpoint lets a recruiter type a weight per criterion, so
    # setting every one to zero is one keystroke away — and a plain `c["weight"] / total`
    # is a ZeroDivisionError at that point. Equal weights keep the stage runnable; the
    # view tells the user it happened.
    if total <= 0:
        equal = 1 / len(updated) if updated else 0
        return [{**c, "weight": equal} for c in updated]

    if abs(total - 1.0) > 0.01:
        updated = [{**c, "weight": c["weight"] / total} for c in updated]  # normalize, surfaced to the user
    return updated


def rubric_is_empty(jd_requirements: list[dict]) -> bool:
    """True when a JD is stated entirely as hard requirements.

    Stage 3 is then skipped for the run and Stage 2's ranking stands as the final
    order — an empty rubric must not reach the aggregation maths.
    """
    return not build_rubric(jd_requirements)
