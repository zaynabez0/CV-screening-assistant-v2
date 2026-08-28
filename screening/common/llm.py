"""The one LLM call helper. Every LLM-calling stage uses it.

JD extraction, CV structuring, Stage 1 and Stage 3 all go through `call_structured`
instead of four slightly different implementations.

Configuration comes from the environment only — never Django settings, because this
package must stay importable with zero Django imports. The project's `.env` is loaded
here as well as in config/settings.py, so a plain `python some_script.py` that never
touches Django still finds the key without exporting it in the shell first:

    OPENAI_API_KEY          the standard name. Checked first.
    KEY_REGULATOR_API_KEY   alternative name, for a gateway-issued sub-key.
    OPENAI_BASE_URL         a gateway endpoint instead of api.openai.com. Optional.
    OPENAI_MODEL            override the model id. Optional — a gateway may expose the
                            same model under a different name.

Prompt caching: OpenAI's Responses API caches long repeated prompt prefixes
automatically. Callers get that for free by putting the shared/reused part of the
prompt in `instructions` and the per-call unique part in `input_text` — which is
exactly what the two arguments are for.

Cost and latency: every call is timed and its token usage read off the response, and
both are reported to whatever `collect_usage()` block is active. That is a *reporting*
channel, not a storage one — this package cannot touch the database, so the Django side
(`runs/usage.py`) opens the block and persists what comes out. `call_structured`'s
return value is unchanged, which is why none of the code or tests around it had to move.
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

DEFAULT_MODEL = "gpt-5.6-luna"

# Mirrors the SDK's own default endpoint, and we always pass it explicitly. Reason:
# if the base URL is set but *blank* — `OPENAI_BASE_URL=` in a .env, easy to do when
# uncommenting the line — the SDK resolves base_url to the empty string and never falls
# back to its default, whether you omit the argument or pass None. Every request then
# goes nowhere with a confusing error. Passing a concrete URL is the only way out.
# (This also overrides the SDK's mTLS/data-residency default, which this project does
# not configure.)
DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Checked in order. OPENAI_API_KEY first so the standard name always wins; the second
# exists for deployments whose key is issued by a gateway under its own variable name.
API_KEY_ENV_VARS = ("OPENAI_API_KEY", "KEY_REGULATOR_API_KEY")

# Same order rule for the endpoint: the SDK's own name first, then the bare name that
# gateway instructions often pair with a sub-key (KEY_REGULATOR_API_KEY + BASE_URL).
# Both are optional — with neither set we talk to api.openai.com directly.
BASE_URL_ENV_VARS = ("OPENAI_BASE_URL", "BASE_URL")

_client = None


def get_base_url() -> str:
    """The endpoint to call: a gateway if one is configured, otherwise OpenAI's.

    A blank or whitespace-only value counts as unset and falls through to the next
    name, so a leftover `OPENAI_BASE_URL=` in a .env can't shadow a filled-in BASE_URL.
    """
    for name in BASE_URL_ENV_VARS:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return DEFAULT_BASE_URL


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
        + ". If you use a gateway rather than api.openai.com, also set one of: "
        + ", ".join(BASE_URL_ENV_VARS)
        + "."
    )


def get_client() -> OpenAI:
    """Build the client on first use, not at import time.

    Deviation from CLAUDE.md, which shows a module-level `client = OpenAI()`: that
    raises at *import* when no key is set, which would break `manage.py check`, the
    test suite, and a fresh checkout before anyone has a key. Built once on the first
    call and reused for every call after that.
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

    `schema_name` identifies which step made the call — the five call sites already pass
    a distinct one ("jd_requirements", "candidate_data", "hard_filter_result",
    "candidate_match", "candidate_scores"), so nothing extra had to be threaded through
    the screening layer to tell them apart.

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

    Defensive on purpose. A gateway may omit `usage` entirely, and — the reason this is
    not merely theoretical — the client is stubbed with a bare `MagicMock` in
    test_llm_client.py, which answers every attribute with another mock rather than
    raising. The isinstance check is what stops a mock being stored as a token count.
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
