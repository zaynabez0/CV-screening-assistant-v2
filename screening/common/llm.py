"""The one LLM call helper.

JD extraction, CV structuring and Stages 1, 2 and 3 all call `call_structured` rather
than each building their own request.

Settings come from the environment, never from Django settings, because this package
must stay importable without Django. `.env` is loaded here as well as in
config/settings.py, so the web server, the stage subprocesses, the tests and a plain
`python some_script.py` all see the same values.

    KEY_REGULATOR_API_KEY   the API key.
    MODEL                   the id of the model to call.
    BASE_URL                the endpoint to call it on.

All three are required, and `.env.example` lists exactly these. The `openai` package is
used as the client library, so the endpoint has to speak the OpenAI Responses API — but
nothing here assumes an OpenAI model or an OpenAI address.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# An explicit path rather than dotenv's search-upwards default, which would resolve
# relative to this file and could pick up a stray .env in some parent directory.
# Nothing to load is fine: a server may be configured with real environment variables.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# One name per setting. A second accepted spelling is a silent override waiting to
# happen, with a leftover shell variable beating the .env someone is reading.
API_KEY_ENV_VAR = "KEY_REGULATOR_API_KEY"
MODEL_ENV_VAR = "MODEL"
BASE_URL_ENV_VAR = "BASE_URL"

_client = None


def _env(name: str) -> str | None:
    """The value of `name`, or None if it is unset, empty or only whitespace.

    `.env.example` ships every line present and empty, so a blank value means the
    template was copied and not filled in. Whitespace is stripped, because a value
    pasted into a .env often arrives with a trailing space.
    """
    value = (os.environ.get(name) or "").strip()
    return value or None


def _require_env(name: str, consequence: str) -> str:
    """Like `_env`, but a missing value is an error naming the variable to fill in."""
    value = _env(name)
    if value is None:
        raise RuntimeError(
            f"{name} is not set, so {consequence}. "
            f"Copy .env.example to .env and fill in {name}."
        )
    return value


def get_base_url() -> str:
    """The endpoint to send requests to.

    No default, so that this project is not tied to one provider. Passing it explicitly
    also keeps the SDK from falling back to its own OpenAI address.
    """
    return _require_env(BASE_URL_ENV_VAR, "there is no endpoint to send requests to")


def get_model() -> str:
    """The id of the model to call.

    No default: which model to run is a deployment decision, and providers each spell
    their ids differently. Unset fails here rather than at the first API call, once per
    candidate, in the middle of a run.
    """
    return _require_env(MODEL_ENV_VAR, "there is no model to call")


def _resolve_api_key() -> str:
    return _require_env(API_KEY_ENV_VAR, "there is no API key to authenticate with")


def get_client() -> OpenAI:
    """The shared client, built on first use and reused afterwards.

    Not built at import, which would make importing this module fail whenever nothing
    is configured and break `manage.py check`, the tests and a fresh checkout.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=_resolve_api_key(), base_url=get_base_url())
    return _client


def reset_client() -> None:
    """Drop the cached client so a changed environment applies to the next call."""
    global _client
    _client = None


def call_structured(instructions: str, input_text: str, schema_name: str, schema: dict) -> dict:
    """One LLM call, returning the parsed JSON response.

    Put the part of the prompt that repeats across calls in `instructions` and the part
    that changes in `input_text`. Providers commonly cache a repeated prefix, so
    splitting the prompt this way can make every call after the first one cheaper.
    """
    response = get_client().responses.create(
        model=get_model(),
        instructions=instructions,
        input=input_text,
        text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
    )
    return json.loads(response.output_text)
