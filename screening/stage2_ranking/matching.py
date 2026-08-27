"""The Stage 2 matching call — one LLM call per candidate, doing both halves.

Stage 2 used to be the one stage that made no LLM call: a self-hosted embedding model
for the semantic half, fuzzy string matching for the keyword half. Both are now this
single structured call, which makes Stage 2 look like Stages 1 and 3 and removes the
project's only heavy local dependency.

Two things are asked for at once because they are judgements about the same candidate
against the same job, and splitting them would double the cost for no gain:

  * `fit_score`     — the overall match, replacing cosine similarity. An LLM reads
                      "server-side programming for web services" as Python backend
                      work for the same reason an embedding model did, but it can also
                      say why, which the vector never could.
  * `keyword_matches` — one verdict per keyword, replacing `fuzz.ratio >= 90`. This is
                      strictly better than the fuzzy match it replaces: it reads the
                      whole candidate rather than only the `skills` list, so a skill
                      demonstrated in a job description finally counts, and it tells
                      "Postgres"/"PostgreSQL" apart from "MySQL" by meaning rather than
                      by edit distance.
"""

from screening.common.llm import call_structured
from screening.stage2_ranking.schema import MATCH_SCHEMA
from screening.stage2_ranking.text_builders import build_jd_text


def match_candidate(jd_requirements: list[dict], anon_candidate: dict, keywords: list[str]) -> dict:
    """The raw model response. Turning it into scores is `ranking.py`'s job."""
    # An empty list gets its own wording rather than a literal "Keywords: []" to reason
    # about: handing a model an empty list invites it to invent some, the same failure
    # `evaluate_hard_filters` guards against by returning early on no must-haves.
    if keywords:
        keyword_task = f"""2. keyword_matches: return one entry for EVERY keyword listed below, in that order,
    repeating the keyword EXACTLY as written — never reword, translate, merge, split, drop
    or add one. Set matched=true only when the candidate genuinely has it, looking at the
    whole candidate rather than just the skills list, so a keyword demonstrated in a job
    description counts. A spelling or formatting variant of the same thing matches
    ("Postgres"/"PostgreSQL", "k8s"/"Kubernetes"); a related but different technology does
    not ("MySQL" is not "PostgreSQL"). A keyword may be a phrase rather than a single term;
    then judge whether the candidate data shows what it describes. Keep `evidence` to a
    short quote when matched=true, or a few words on what is missing when it is false.

    Keywords: {keywords}"""
    else:
        keyword_task = ("2. This run has no keywords. Return keyword_matches as an empty array. "
                        "Do not invent keywords.")

    # instructions is identical for every candidate in the run — the job description and
    # the keyword list do not vary — so it is the repeated prefix that gets cached
    # automatically, same principle as Stages 1 and 3. Only the candidate is per-call.
    instructions = f"""Judge how well this candidate matches a job description, and which
    of a recruiter's keywords they actually demonstrate. Two separate jobs. Base both ONLY
    on what the candidate data explicitly states — never assume, never infer a skill from a
    job title, never fill a gap with what is typical for the role.

    1. fit_score, 0-100: how well the candidate's skills and experience meet the
    requirements below, taken as a whole. Judge substance, not wording — a candidate who
    clearly did the work scores high even if they never use the job description's phrasing.
    A term listed with no experience behind it is weak evidence. Roughly: 0-20 a different
    field, 21-40 adjacent but not this job, 41-60 partial fit with clear gaps, 61-80 solid
    on most requirements, 81-100 strong evidence on essentially all of them. If the
    candidate data is empty or shows nothing relevant, fit_score must be in 0-20.
    Give one sentence of fit_reasoning naming what drove the number.

    {keyword_task}

    Job requirements:
    {build_jd_text(jd_requirements)}"""
    return call_structured(instructions, f"Candidate: {anon_candidate}", "candidate_match", MATCH_SCHEMA)
