"""Anonymization — applied once after preprocessing, reused by every selection stage.

Computed on demand wherever a stage needs it, never persisted as a second copy of the
data. Stage 1, 2 and 3 all call this rather than each deriving their own version.
"""


def anonymize_candidate(candidate: dict) -> dict:
    strip = {"full_name", "email", "phone"}
    return {k: v for k, v in candidate.items() if k not in strip}
