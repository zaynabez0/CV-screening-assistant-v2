"""The one LLM call helper. Every LLM-calling stage uses it.

JD extraction, CV structuring, Stage 1 and Stage 3 all go through `call_structured`
instead of four slightly different implementations.

Prompt caching: OpenAI's Responses API caches long repeated prompt prefixes
automatically. Callers get that for free by putting the shared/reused part of the
prompt in `instructions` and the per-call unique part in `input_text` — which is
exactly what the two arguments are for.
"""

import json

from openai import OpenAI

MODEL = "gpt-5.6-luna"

_client = None


def get_client() -> OpenAI:
    """Build the client on first use, not at import time.

    Deviation from CLAUDE.md, which shows a module-level `client = OpenAI()`: that
    raises at *import* when OPENAI_API_KEY is unset, which would break `manage.py
    check`, the test suite, and a fresh checkout before anyone has a key. This is the
    same load-once-and-reuse pattern CLAUDE.md already uses for the embedding model.
    """
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment — never hardcode the key
    return _client


def call_structured(instructions: str, input_text: str, schema_name: str, schema: dict) -> dict:
    """instructions: the shared/reused part of the prompt (put first, so it's the
    cached prefix). input_text: the per-call unique part (e.g. one candidate)."""
    response = get_client().responses.create(
        model=MODEL,
        instructions=instructions,
        input=input_text,
        text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
    )
    return json.loads(response.output_text)
