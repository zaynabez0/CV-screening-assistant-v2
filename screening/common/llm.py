"""The one LLM call helper. Every LLM-calling stage uses it.

JD extraction, CV structuring, Stage 1 and Stage 3 all go through `call_structured`
instead of four slightly different implementations.

Configuration comes from the environment only — never Django settings, because this
package must stay importable with zero Django imports:

    OPENAI_API_KEY          the key. Standard name, wins if several are set.
    KEY_REGULATOR_API_KEY   alternative name, for a gateway-issued sub-key.
    OPENAI_BASE_URL         a gateway endpoint instead of api.openai.com. Optional.
    OPENAI_MODEL            override the model id. Optional — a gateway may expose the
                            same model under a different name.

Prompt caching: OpenAI's Responses API caches long repeated prompt prefixes
automatically. Callers get that for free by putting the shared/reused part of the
prompt in `instructions` and the per-call unique part in `input_text` — which is
exactly what the two arguments are for.
"""

import json
import os

from openai import OpenAI

DEFAULT_MODEL = "gpt-5.6-luna"

# Mirrors the SDK's own default endpoint, and we always pass it explicitly. Reason:
# if OPENAI_BASE_URL is set but *blank* — `OPENAI_BASE_URL=` in a .env, easy to do when
# uncommenting the line — the SDK resolves base_url to the empty string and never falls
# back to its default, whether you omit the argument or pass None. Every request then
# goes nowhere with a confusing error. Passing a concrete URL is the only way out.
# (This also overrides the SDK's mTLS/data-residency default, which this project does
# not configure.)
DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Checked in order. OPENAI_API_KEY first so the standard name always wins; the second
# exists for deployments whose key is issued by a gateway under its own variable name.
API_KEY_ENV_VARS = ("OPENAI_API_KEY", "KEY_REGULATOR_API_KEY")

_client = None


def get_base_url() -> str:
    """The endpoint to call: a gateway if OPENAI_BASE_URL is set, otherwise OpenAI's."""
    return (os.environ.get("OPENAI_BASE_URL") or "").strip() or DEFAULT_BASE_URL


def get_model() -> str:
    """The model id to call.

    A gateway often fronts the same model under a different id, so this is overridable
    without editing code. Defaults to CLAUDE.md's `gpt-5.6-luna`.
    """
    return os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def _resolve_api_key() -> str:
    """First key found among API_KEY_ENV_VARS.

    Raises with the accepted names spelled out: the generic SDK error ("Missing
    credentials…") sends people looking for OPENAI_API_KEY when this project may be
    configured with the other one.
    """
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        "No LLM API key found in the environment. Set one of: "
        + ", ".join(API_KEY_ENV_VARS)
        + ". If you use a gateway rather than api.openai.com, also set OPENAI_BASE_URL."
    )


def get_client() -> OpenAI:
    """Build the client on first use, not at import time.

    Deviation from CLAUDE.md, which shows a module-level `client = OpenAI()`: that
    raises at *import* when no key is set, which would break `manage.py check`, the
    test suite, and a fresh checkout before anyone has a key. This is the same
    load-once-and-reuse pattern CLAUDE.md already uses for the embedding model.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=_resolve_api_key(),  # read from the environment — never hardcode the key
            base_url=get_base_url(),
        )
    return _client


def reset_client() -> None:
    """Drop the cached client so a changed environment takes effect on the next call."""
    global _client
    _client = None


def call_structured(instructions: str, input_text: str, schema_name: str, schema: dict) -> dict:
    """instructions: the shared/reused part of the prompt (put first, so it's the
    cached prefix). input_text: the per-call unique part (e.g. one candidate)."""
    response = get_client().responses.create(
        model=get_model(),
        instructions=instructions,
        input=input_text,
        text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
    )
    return json.loads(response.output_text)
