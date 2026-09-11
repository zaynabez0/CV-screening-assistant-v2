"""The one LLM call helper.

JD extraction, CV structuring and Stages 1, 2 and 3 all call `call_structured` rather
than each building their own request.

Settings come from the environment, never from Django settings, because this package
must stay importable without Django. `.env` is loaded here as well as in
config/settings.py, so the web server, the stage subprocesses, the tests and a plain
`python some_script.py` all see the same values.

    KEY_REGULATOR_API_KEY   the API key. Required.
    MODEL                   the id of the model to call. Required.
    BASE_URL                a gateway endpoint instead of api.openai.com. Optional.

`.env.example` lists exactly these, and nothing else.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
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

# The SDK's own default, which we always pass explicitly. If base_url is set but blank
# the SDK uses the empty string instead of falling back, and every request goes nowhere
# with a confusing error. Passing a real URL is the only way to avoid that.
DEFAULT_BASE_URL = "https://api.openai.com/v1"

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
    """The gateway endpoint, or OpenAI's own when no gateway is configured."""
    return _env(BASE_URL_ENV_VAR) or DEFAULT_BASE_URL


def get_model() -> str:
    """The id of the model to call.

    No default: which model to run is a deployment decision, and a gateway often
    exposes a model under an id of its own. Unset fails here rather than at the first
    API call, once per candidate, in the middle of a run.
    """
    return _require_env(MODEL_ENV_VAR, "there is no model to call")


def _resolve_api_key() -> str:
    return _require_env(API_KEY_ENV_VAR, "there is no API key to authenticate with")


def get_client() -> OpenAI:
    """The shared client, built on first use and reused afterwards.

    Not built at import, which would make importing this module fail whenever no key
    is set and break `manage.py check`, the tests and a fresh checkout.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=_resolve_api_key(), base_url=get_base_url())
    return _client


def reset_client() -> None:
    """Drop the cached client so a changed environment applies to the next call."""
    global _client
    _client = None


@dataclass(frozen=True)
class CallUsage:
    """What one `call_structured` cost, in tokens and wall-clock time.

    `schema_name` says which step made the call: the five call sites each pass their
    own, so it doubles as a label.

    Both cached and reasoning counts are subsets, not extras: `cached_input_tokens` is
    part of `input_tokens`, and `reasoning_tokens` is part of `output_tokens`. Price
    them accordingly.
    """

    schema_name: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    duration_ms: int


# Thread-local rather than a plain global: the dev server handles requests in threads,
# and two recruiters working at once must not land in each other's tally.
_active = threading.local()


@contextmanager
def collect_usage():
    """Collect a CallUsage for every `call_structured` made inside this block.

    The list is yielded straight away and filled as the calls happen, so a caller that
    handles its own exceptions still sees the calls that completed. Blocks nest: an
    inner block takes over, and the outer one resumes when it exits.
    """
    previous = getattr(_active, "sink", None)
    sink: list[CallUsage] = []
    _active.sink = sink
    try:
        yield sink
    finally:
        _active.sink = previous


def _token_count(node, *path) -> int:
    """Follow `path` down the response and return an int, or 0 if anything is missing.

    A gateway may leave `usage` out entirely, and a token count is not worth failing a
    run over. The isinstance check also keeps a stubbed client's mock attributes from
    being stored as counts.
    """
    for key in path:
        node = getattr(node, key, None)
    return node if isinstance(node, int) else 0


def call_structured(instructions: str, input_text: str, schema_name: str, schema: dict) -> dict:
    """One LLM call, returning the parsed JSON response.

    Put the part of the prompt that repeats across calls in `instructions` and the part
    that changes in `input_text`. The API caches repeated prefixes automatically, so
    splitting them this way makes every call after the first one cheaper.

    Token usage and timing go to the active `collect_usage()` block, if there is one.
    That is a side channel rather than a second return value, so callers are unaffected.
    """
    model = get_model()
    started = time.perf_counter()
    response = get_client().responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
    )
    duration_ms = round((time.perf_counter() - started) * 1000)

    sink = getattr(_active, "sink", None)
    if sink is not None:
        usage = getattr(response, "usage", None)
        sink.append(CallUsage(
            schema_name=schema_name,
            model=model,
            input_tokens=_token_count(usage, "input_tokens"),
            cached_input_tokens=_token_count(usage, "input_tokens_details", "cached_tokens"),
            output_tokens=_token_count(usage, "output_tokens"),
            reasoning_tokens=_token_count(usage, "output_tokens_details", "reasoning_tokens"),
            duration_ms=duration_ms,
        ))

    return json.loads(response.output_text)
