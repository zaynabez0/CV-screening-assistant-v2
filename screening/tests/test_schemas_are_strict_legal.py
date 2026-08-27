"""Every schema must satisfy OpenAI's Structured Outputs `strict: true` rules.

This exists because of a real failure: all four schemas were written exactly as CLAUDE.md
specifies, which omits `additionalProperties: false`. The API rejects such a schema with a
400 — so *every* LLM call in the app failed, and no test caught it, because every other
test mocks `call_structured` and never inspects the schema.

Mocking the call is right (tests must not hit the network). Never validating the schema
was the gap. This closes it.
"""

import pytest

from screening.cv.schema import CV_SCHEMA
from screening.jd.schema import JD_SCHEMA
from screening.stage1_hard_filter.schema import HARD_FILTER_SCHEMA
from screening.stage2_ranking.schema import MATCH_SCHEMA
from screening.stage3_scoring.schema import SCORE_SCHEMA

ALL_SCHEMAS = {
    "JD_SCHEMA": JD_SCHEMA,
    "CV_SCHEMA": CV_SCHEMA,
    "HARD_FILTER_SCHEMA": HARD_FILTER_SCHEMA,
    "MATCH_SCHEMA": MATCH_SCHEMA,
    "SCORE_SCHEMA": SCORE_SCHEMA,
}


def object_nodes(node, path="root"):
    """Every object node in a schema, with a readable path to it."""
    if not isinstance(node, dict):
        return
    if node.get("type") == "object":
        yield path, node
    for key, child in (node.get("properties") or {}).items():
        yield from object_nodes(child, f"{path}.{key}")
    if "items" in node:
        yield from object_nodes(node["items"], f"{path}[]")


@pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
def test_every_object_forbids_additional_properties(name):
    """Strict mode requires `additionalProperties: false` on EVERY object, including
    objects nested inside arrays — which is the level that is easy to miss."""
    offenders = [
        path for path, node in object_nodes(ALL_SCHEMAS[name])
        if node.get("additionalProperties") is not False
    ]

    assert offenders == [], f"{name}: objects missing additionalProperties:false -> {offenders}"


@pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
def test_every_property_is_required(name):
    """Strict mode has no optional keys: everything in `properties` must be in `required`.
    Use a nullable type instead of omitting a field."""
    offenders = {
        path: sorted(set(node.get("properties", {})) - set(node.get("required", [])))
        for path, node in object_nodes(ALL_SCHEMAS[name])
        if set(node.get("properties", {})) - set(node.get("required", []))
    }

    assert offenders == {}, f"{name}: properties missing from required -> {offenders}"


@pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
def test_the_root_is_an_object(name):
    """Structured Outputs will not accept a bare array or scalar at the root."""
    assert ALL_SCHEMAS[name]["type"] == "object"


def test_the_audit_actually_catches_a_bad_schema():
    """Guard against the guard: a schema missing the property must be reported, or these
    tests would pass on anything."""
    bad = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}

    offenders = [p for p, n in object_nodes(bad) if n.get("additionalProperties") is not False]

    assert offenders == ["root"]


def test_the_audit_reaches_objects_nested_in_arrays():
    """The level that broke in practice — an array's `items` object."""
    bad = {
        "type": "object", "additionalProperties": False,
        "properties": {"rows": {"type": "array", "items": {
            "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}},
        "required": ["rows"],
    }

    offenders = [p for p, n in object_nodes(bad) if n.get("additionalProperties") is not False]

    assert offenders == ["root.rows[]"]
