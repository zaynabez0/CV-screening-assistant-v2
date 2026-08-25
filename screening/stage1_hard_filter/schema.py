"""HARD_FILTER_SCHEMA — the shape of one candidate's must-have evaluation.

`evidence` and `reason` are required on every item, including passes. That is the
mitigation for using an LLM here instead of a rule engine: a decision without a
quotable basis is not a decision anyone can defend to a rejected candidate.
"""

HARD_FILTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "met": {"type": "boolean"},
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["requirement", "met", "evidence", "reason"],
                "additionalProperties": False
            }
        }
    },
    "required": ["evaluations"]
}
