"""MATCH_SCHEMA — one candidate's Stage 2 match against the job description.

Two jobs in one response, because Stage 2 blends two halves and both now come from
the same call: `fit_score` is the overall judgement of the candidate against the
requirements, and `keyword_matches` is a per-keyword presence check against the
recruiter's editable list.

`fit_score` is an integer 0-100 rather than a float 0-1: models are markedly more
consistent on a 0-100 scale, and the division back to 0-1 happens in Python where it
is exact. `keyword_matches` deliberately returns booleans, not a fraction — the
fraction is arithmetic, and arithmetic belongs in Python where it can be tested
(same reasoning as Stage 3's aggregation).

`keyword` is a free-text string: nothing in the schema binds it to the keyword list
that was asked for, which is why `ranking.py` matches the names back defensively
instead of trusting them.
"""

MATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "fit_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "fit_reasoning": {"type": "string"},
        "keyword_matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "matched": {"type": "boolean"},
                    "evidence": {"type": "string"}
                },
                "required": ["keyword", "matched", "evidence"],
                "additionalProperties": False
            }
        }
    },
    "required": ["fit_score", "fit_reasoning", "keyword_matches"]
}
