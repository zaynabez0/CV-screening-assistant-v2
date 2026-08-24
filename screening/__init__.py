"""Pure-Python pipeline package for CV screening.

Architecture rule (non-negotiable): nothing under `screening/` imports Django.
Django views and management commands call into this package, never the reverse.
"""
