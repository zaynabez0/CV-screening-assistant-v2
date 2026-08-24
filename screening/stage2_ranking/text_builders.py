"""What actually gets embedded — a constructed summary, not raw CV text.

Embedding the raw CV would mix headers, contact blocks and formatting noise into the
vector. These builders reduce a candidate to the parts a JD is actually about, from
the already-structured JSON.
"""


def build_candidate_text(candidate: dict) -> str:
    parts = [
        f"Skills: {', '.join(candidate.get('skills', []))}",
        f"Total experience: {candidate.get('total_years_experience', 0)} years",
    ]
    for exp in candidate.get("experience", []):
        parts.append(f"{exp.get('title','')} at {exp.get('company','')}: {exp.get('description','')}")
    return "\n".join(parts)


def build_jd_text(requirements: list[dict]) -> str:
    return "\n".join(f"{r['requirement']}: {r['description']}" for r in requirements)
