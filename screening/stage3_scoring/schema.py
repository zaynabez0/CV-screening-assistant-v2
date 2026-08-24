"""SCORE_SCHEMA — one candidate's per-criterion scores.

`evidence_quote` is required on every item including low scores, and `confidence` is
required so a guess is visibly a guess. Note `criterion` is a free-text string: nothing
in the schema binds it to a rubric name, which is why aggregation guards against
names it doesn't recognise.
"""

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "criterion_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "evidence_quote": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["criterion", "score", "evidence_quote", "reasoning", "confidence"]
            }
        }
    },
    "required": ["criterion_scores"]
}
