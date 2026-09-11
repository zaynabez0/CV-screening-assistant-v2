"""The one LLM call helper. Every LLM-calling stage uses it.

JD extraction, CV structuring, Stage 1, Stage 2 and Stage 3 all go through
`call_structured` instead of five slightly different implementations.

Configuration comes from the environment only — never Django settings, because this
package must stay importable with zero Django imports. The project's `.env` is loaded
here as well as in config/settings.py, so a plain `python some_script.py` that never
touches Django still finds the settings without exporting them in the shell first.

Three variables, one name each. They are spelled exactly the same way in `.env.example`,
which is the full list of what this project reads:

    KEY_REGULATOR_API_KEY   the API key. Required.
    OPENAI_MODEL            the id of the model to call. Required.
    OPENAI_BASE_URL         a gateway endpoint instead of api.openai.com. Optional.

Prompt caching: OpenAI's Responses API caches long repeated prompt prefixes
automatically. Callers get that for free by putting the shared/reused part of the
prompt in `instructions` and the per-call unique part in `input_text` — which is
exactly what the two arguments are for.

Cost and latency: every call is timed and its token usage read off the response, and
both are reported to whatever `collect_usage()` block is active. That is a *reporting*
channel, not a storage one — this package cannot touch the database, so the Django side
(`runs/usage.py`) opens the block and persists what comes out.
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

# Load the project's .env at import, so every entry point is configured the same way:
# the Django server, the stage subprocesses, pytest, and a bare script run by hand.
# Two deliberate choices:
#   * the path is explicit rather than dotenv's search-upwards default, which resolves
#     relative to *this file* and would quietly pick up a stray .env in some parent
#     directory if the package were ever installed elsewhere;
#   * load_dotenv does not overwrite variables that are already set, so a real shell
#     variable still wins over .env — the override documented in the README.
# A missing .env is not an error: on a machine configured with real environment
# variables there is nothing to load.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# One environment variable per setting, and no aliases: what `.env.example` lists is
# the whole contract. A second accepted name for the same setting is a silent override
# waiting to happen — a leftover variable in someone's shell beating the .env they are
# looking at while they debug.
API_KEY_ENV_VAR = "KEY_REGULATOR_API_KEY"
MODEL_ENV_VAR = "OPENAI_MODEL"
BASE_URL_ENV_VAR = "OPENAI_BASE_URL"

# Mirrors the SDK's own default endpoint, and we always pass it explicitly. Reason:
# if the base URL is set but *blank* — `OPENAI_BASE_URL=` in a .env, which is exactly
# how the template ships — the SDK resolves base_url to the empty string and never falls
# back to its default, whether you omit the argument or pass None. Every request then
# goes nowhere with a confusing error. Passing a concrete URL is the only way out.
# (This also overrides the SDK's mTLS/data-residency default, which this project does
# not configure.)
DEFAULT_BASE_URL = "https://api.openai.com/v1"

_client = None


def _env(name: str) -> str | None:
    """The value of `name`, or None when it is unset, empty or whitespace only.

    Blank counts as unset for all three settings. `.env.example` ships every line
    present and empty, so "filled in" has to mean more than "the line exists" —
    and a copy-pasted value often arrives with a trailing space attached.
    """
    value = (os.environ.get(name) or "").strip()
    return value or None


def _require_env(name: str, consequence: str) -> str:
    """`_env`, but a missing value is a startup error naming the variable to fill in.

    The SDK's own version of this ("Missing credentials…") names OPENAI_API_KEY, which
    this project does not read — so it sends people looking for the wrong variable.
    """
    value = _env(name)
    if value is None:
        raise RuntimeError(
            f"{name} is not set, so {consequence}. "
            f"Copy .env.example to .env and fill in {name}."
        )
    return value


def get_base_url() -> str:
    """The endpoint to call: the gateway named by OPENAI_BASE_URL, or OpenAI's own."""
    return _env(BASE_URL_ENV_VAR) or DEFAULT_BASE_URL


def get_model() -> str:
    """The model id to call, from OPENAI_MODEL.

    Deliberately has no default. Which model this project runs on is a deployment
    decision — and a gateway commonly fronts a model under an id of its own — so it is
    declared once in `.env` rather than guessed here. A wrong default would be worse
    than a missing one: it fails at the first API call, per candidate, mid-run.
    """
    return _require_env(MODEL_ENV_VAR, "there is no model to call")


def _resolve_api_key() -> str:
    return _require_env(API_KEY_ENV_VAR, "there is no API key to authenticate with")


def get_client() -> OpenAI:
    """Build the client on first use, not at import time.

    A module-level `OpenAI()` would raise at *import* whenever no key is set, which
    would break `manage.py check`, the test suite, and a fresh checkout before anyone
    has a key. Built once on the first call and reused for every call after that.
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


@dataclass(frozen=True)
class CallUsage:
    """What one `call_structured` cost, in tokens and wall-clock time.

    `schema_name` identifies which step made the call: the five call sites each pass a
    distinct one ("jd_requirements", "candidate_data", "hard_filter_result",
    "candidate_match", "candidate_scores"), so it doubles as the label telling recorded
    calls apart without anything extra being threaded through the screening layer.

    Two subtleties that matter when pricing this, both of them subsets rather than extras:
    `cached_input_tokens` is part of `input_tokens`, not additional to it, and
    `reasoning_tokens` is part of `output_tokens`. Bill accordingly.
    """

    schema_name: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    duration_ms: int


# Thread-local, not a plain module global: the dev server handles requests in threads,
# and two recruiters structuring CVs at once must not have their calls land in each
# other's tally.
_active = threading.local()


@contextmanager
def collect_usage():
    """Collect a CallUsage for every `call_structured` made inside this block.

    Yields the list, which is filled as the calls happen — so a caller that wraps this in
    its own `finally` still sees the calls that completed before an exception. Blocks
    nest: an inner block takes over, and the outer one resumes when it exits.
    """
    previous = getattr(_active, "sink", None)
    sink: list[CallUsage] = []
    _active.sink = sink
    try:
        yield sink
    finally:
        _active.sink = previous


def _token_count(node, *path) -> int:
    """Walk `path` off the response and return an int, or 0 if anything is missing.

    Defensive on purpose: a gateway may omit `usage` entirely, and a token count is not
    worth failing a run over. The isinstance check carries its own weight in the tests,
    where the client is stubbed with a bare `MagicMock` that answers every attribute
    with another mock rather than raising — without it, a mock is stored as a count.
    """
    for key in path:
        node = getattr(node, key, None)
    return node if isinstance(node, int) else 0


def call_structured(instructions: str, input_text: str, schema_name: str, schema: dict) -> dict:
    """instructions: the shared/reused part of the prompt (put first, so it's the
    cached prefix). input_text: the per-call unique part (e.g. one candidate).

    Returns the parsed response. Usage and latency go to the active `collect_usage()`
    block, if any — deliberately a side channel rather than a second return value, so
    every caller stays as it was.
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
