"""JD text in, a flat tagged requirements list out.

The file path here is the *second* caller of the shared text-extraction function —
CV upload is the first. Same operation, same code, deliberately not duplicated.
"""

from screening.common.llm import call_structured
from screening.common.text_extraction import extract_text_from_file
from screening.jd.schema import JD_SCHEMA


def get_jd_text(uploaded_file=None, free_text=None) -> str:
    if uploaded_file is not None:
        return extract_text_from_file(uploaded_file.path)  # shared function, see below
    if free_text is not None:
        return free_text
    raise ValueError("Either uploaded_file or free_text must be provided")


def extract_requirements(jd_text: str) -> list[dict]:
    instructions = """Extract hiring requirements from this job description as a flat list.
    Tag each requirement's type:
    - must_have: explicit hard requirement. Only tag as must_have what is genuinely
      required — when in doubt, tag as nice_to_have or inferred instead.
    - nice_to_have: explicitly stated as preferred/bonus/a plus.
    - inferred: reasonably implied by context (seniority, domain) but not explicitly stated.
    For nice_to_have and inferred items, write a 1-5 scale in the fixed format
    "5=<strong evidence>, 3=<meets baseline>, 1=<no evidence>" and a suggested_weight.
    For must_have items, set scale to "" and suggested_weight to 0 (unused for this type).
    Do not invent requirements not stated or clearly implied."""
    result = call_structured(instructions, jd_text, "jd_requirements", JD_SCHEMA)
    return result["requirements"]


def normalize_requirement(item: dict) -> dict:
    """Apply the must_have rule to a single requirement.

    A must_have item carries scale "" and suggested_weight 0 — both are unused for that
    type. Applied whenever the recruiter changes a row's type at the review
    checkpoint, so a requirement switched to must_have does not keep a stale scale that
    Stage 3 would then silently ignore.
    """
    if item.get("type") == "must_have":
        return {**item, "scale": "", "suggested_weight": 0}
    return item
