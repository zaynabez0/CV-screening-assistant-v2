"""How the job description is rendered into the Stage 2 prompt.

One requirement per line, name paired with its description, which is what the model
needs to judge fit against. There is no matching candidate builder: the candidate goes
into the call as its already-structured, already-anonymized dict, exactly as Stages 1
and 3 send it. The old builder reduced a candidate to skills + years + roles because
embedding raw text mixed formatting noise into the vector; an LLM has no such problem
and reads education, certifications and languages too.
"""


def build_jd_text(requirements: list[dict]) -> str:
    return "\n".join(f"{r['requirement']}: {r['description']}" for r in requirements)
