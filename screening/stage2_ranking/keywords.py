"""The recruiter's keyword list — seeding it. Matching against it lives in `matching.py`.

The list is the primary lever the user has over Stage 2, and this seeds a first version
of it from the job description so the checkpoint opens with something to edit rather
than an empty box.
"""


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
