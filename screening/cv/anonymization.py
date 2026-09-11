"""Dropping direct identifiers before a candidate is sent to the model.

Stages 1, 2 and 3 all call this rather than each deriving their own version, and it is
computed on demand rather than stored — a second, anonymized copy of every candidate
would be one more place personal data can leak from.
"""

DIRECT_IDENTIFIERS = {"full_name", "email", "phone"}


def anonymize_candidate(candidate: dict) -> dict:
    return {k: v for k, v in candidate.items() if k not in DIRECT_IDENTIFIERS}
