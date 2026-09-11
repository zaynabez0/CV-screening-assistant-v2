"""Seeding the recruiter's keyword list. Matching against it lives in `matching.py`."""


def seed_keywords(requirements: list[dict]) -> list[str]:
    """A first keyword list, from the JD's must_have and nice_to_have requirement names.

    A starting point only: editing this list at the checkpoint is the main lever the
    recruiter has over Stage 2, so it opens with something to change rather than an
    empty box. `inferred` requirements are excluded — they are not stated in the JD, so
    their wording is the model's rather than the employer's, which makes a poor search
    term.
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
