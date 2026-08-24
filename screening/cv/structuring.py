"""CV text -> CV_SCHEMA JSON, via the shared LLM helper."""

from screening.common.llm import call_structured
from screening.cv.schema import CV_SCHEMA


def structure_candidate(cv_text: str) -> dict:
    instructions = """Extract structured candidate data from this CV text.
    Only extract what is explicitly stated — never infer or guess missing fields (use ""
    or null as appropriate for the field type). For each work experience, include a short
    verbatim quote as evidence. If the CV is ambiguous, damaged, or unusually formatted,
    set extraction_confidence accordingly and explain why in extraction_notes."""
    return call_structured(instructions, cv_text, "candidate_data", CV_SCHEMA)
