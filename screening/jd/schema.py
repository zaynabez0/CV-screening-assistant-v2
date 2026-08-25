"""JD_SCHEMA — one flat list of requirements, each tagged with its type.

There is deliberately no separate schema per category. This single list drives both
Stage 1 (the must_have items, gated pass/fail) and Stage 3 (nice_to_have + inferred,
scored against their scales), so a requirement can be recategorised by the recruiter
at the review checkpoint without any other data changing shape.
"""

JD_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "description": {"type": "string"},
                    "type": {"type": "string", "enum": ["must_have", "nice_to_have", "inferred"]},
                    "scale": {"type": "string"},          # "5=<strong evidence>, 3=<meets baseline>, 1=<no evidence>" — used by Stage 3 only, ignored for must_have
                    "suggested_weight": {"type": "number"}  # used by Stage 3 only, ignored for must_have
                },
                "required": ["requirement", "description", "type", "scale", "suggested_weight"]
            }
        }
    },
    "required": ["requirements"]
}

REQUIREMENT_TYPES = ("must_have", "nice_to_have", "inferred")
