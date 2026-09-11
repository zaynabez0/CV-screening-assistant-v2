"""How the shared LLM client is configured from the environment. No network calls.

Three variables, one name each, all required: KEY_REGULATOR_API_KEY, MODEL and
BASE_URL. Every stage that calls an LLM goes through this one helper, so a mistake here
breaks all five at once.

Blank is treated as unset for all three, which matters because `.env.example` ships
every line present and empty — a copied-but-unfilled template must fail with a message
naming the variable, not send an empty key or an empty address to a provider.
"""

from unittest.mock import MagicMock, patch

import pytest

from screening.common import llm
from screening.common.llm import (
    API_KEY_ENV_VAR,
    BASE_URL_ENV_VAR,
    MODEL_ENV_VAR,
    call_structured,
    get_base_url,
    get_client,
    get_model,
    reset_client,
)

SCHEMA = {"type": "object", "properties": {}, "required": []}

DEFAULTS = {
    API_KEY_ENV_VAR: "sk-x",
    MODEL_ENV_VAR: "a-model",
    BASE_URL_ENV_VAR: "https://provider.example.com/v1",
}


@pytest.fixture(autouse=True)
def clean_client(monkeypatch):
    """No cached client leaks between tests, and nothing leaks in from the shell or .env.

    This runs after the root conftest fixture and deliberately undoes it: these tests own
    the environment, including the "nothing is configured" case.
    """
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    reset_client()
    yield
    reset_client()


def configure(monkeypatch, **overrides):
    """Set every required variable, so a test only has to name the one it is about."""
    for name, value in {**DEFAULTS, **overrides}.items():
        monkeypatch.setenv(name, value)


# --- The API key ----------------------------------------------------------------

def test_the_key_is_read_from_the_environment(monkeypatch):
    configure(monkeypatch, **{API_KEY_ENV_VAR: "sk-gateway"})

    assert get_client().api_key == "sk-gateway"


def test_no_key_names_the_variable_to_set(monkeypatch):
    """The SDK's own error names OPENAI_API_KEY, which this project does not read — so
    it sends someone hunting for a variable that would have no effect."""
    configure(monkeypatch)
    monkeypatch.delenv(API_KEY_ENV_VAR)

    with pytest.raises(RuntimeError) as exc:
        get_client()

    assert API_KEY_ENV_VAR in str(exc.value)


def test_an_empty_key_is_treated_as_unset(monkeypatch):
    """`KEY_REGULATOR_API_KEY=` is how the template ships. Copying it without filling it
    in must fail here, naming the variable — not at the API with a 401."""
    configure(monkeypatch, **{API_KEY_ENV_VAR: ""})

    with pytest.raises(RuntimeError) as exc:
        get_client()

    assert API_KEY_ENV_VAR in str(exc.value)


def test_a_whitespace_only_key_is_treated_as_unset(monkeypatch):
    configure(monkeypatch, **{API_KEY_ENV_VAR: "   "})

    with pytest.raises(RuntimeError):
        get_client()


def test_surrounding_whitespace_is_stripped_from_the_key(monkeypatch):
    """Copy-pasting a key into a .env often brings a trailing space with it."""
    configure(monkeypatch, **{API_KEY_ENV_VAR: "  sk-gateway  "})

    assert get_client().api_key == "sk-gateway"


# --- The endpoint ------------------------------------------------------------------

def test_the_endpoint_reaches_the_client(monkeypatch):
    configure(monkeypatch, **{BASE_URL_ENV_VAR: "https://my-provider.example.com/v1"})

    assert "my-provider.example.com" in str(get_client().base_url)


def test_no_endpoint_names_the_variable_to_set(monkeypatch):
    """There is no default endpoint, so that this project is not tied to one provider."""
    configure(monkeypatch)
    monkeypatch.delenv(BASE_URL_ENV_VAR)

    with pytest.raises(RuntimeError) as exc:
        get_base_url()

    assert BASE_URL_ENV_VAR in str(exc.value)


def test_an_empty_endpoint_is_treated_as_unset(monkeypatch):
    """`BASE_URL=` with nothing after it is how the template ships. Left blank, the SDK
    would resolve it to an empty address and every request would go nowhere with a
    confusing error — so it has to fail here instead, naming the variable."""
    configure(monkeypatch, **{BASE_URL_ENV_VAR: ""})

    with pytest.raises(RuntimeError) as exc:
        get_base_url()

    assert BASE_URL_ENV_VAR in str(exc.value)


def test_a_whitespace_only_endpoint_is_treated_as_unset(monkeypatch):
    configure(monkeypatch, **{BASE_URL_ENV_VAR: "   "})

    with pytest.raises(RuntimeError):
        get_base_url()


def test_surrounding_whitespace_is_stripped_from_the_endpoint(monkeypatch):
    configure(monkeypatch, **{BASE_URL_ENV_VAR: "  https://provider.example.com/v1  "})

    assert get_base_url() == "https://provider.example.com/v1"


def test_a_stray_openai_variable_cannot_reach_the_client(monkeypatch):
    """The SDK reads OPENAI_API_KEY and OPENAI_BASE_URL itself when they are not passed.
    Both are always passed, so a leftover variable in someone's shell is inert."""
    configure(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stray")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://stray.example.com/v1")

    client = get_client()

    assert client.api_key == "sk-x"
    assert "stray.example.com" not in str(client.base_url)


# --- Model id ----------------------------------------------------------------------

def test_the_model_is_read_from_the_environment(monkeypatch):
    configure(monkeypatch, **{MODEL_ENV_VAR: "internal-luna-v2"})

    assert get_model() == "internal-luna-v2"


def test_no_model_names_the_variable_to_set(monkeypatch):
    """There is no default on purpose: which model this runs on is a deployment
    decision. A wrong default would fail per candidate, mid-run, instead of up front."""
    configure(monkeypatch)
    monkeypatch.delenv(MODEL_ENV_VAR)

    with pytest.raises(RuntimeError) as exc:
        get_model()

    assert MODEL_ENV_VAR in str(exc.value)


def test_an_empty_model_is_treated_as_unset(monkeypatch):
    configure(monkeypatch, **{MODEL_ENV_VAR: ""})

    with pytest.raises(RuntimeError) as exc:
        get_model()

    assert MODEL_ENV_VAR in str(exc.value)


def test_surrounding_whitespace_is_stripped_from_the_model(monkeypatch):
    configure(monkeypatch, **{MODEL_ENV_VAR: "  internal-luna-v2  "})

    assert get_model() == "internal-luna-v2"


# --- The request ---------------------------------------------------------------------

def test_call_structured_sends_the_documented_request_shape(monkeypatch):
    """Structured Outputs, strict, with the configured model id."""
    configure(monkeypatch, **{MODEL_ENV_VAR: "internal-luna-v2"})

    fake = MagicMock()
    fake.responses.create.return_value = MagicMock(output_text='{"ok": true}')
    with patch.object(llm, "get_client", return_value=fake):
        result = call_structured("shared prefix", "per-call text", "my_schema", SCHEMA)

    assert result == {"ok": True}
    kwargs = fake.responses.create.call_args.kwargs
    assert kwargs["model"] == "internal-luna-v2"
    assert kwargs["instructions"] == "shared prefix"     # the cached prefix
    assert kwargs["input"] == "per-call text"
    assert kwargs["text"] == {"format": {"type": "json_schema", "name": "my_schema",
                                         "schema": SCHEMA, "strict": True}}


# --- Client caching -------------------------------------------------------------------

def test_the_client_is_built_once_and_reused(monkeypatch):
    configure(monkeypatch)

    assert get_client() is get_client()


def test_reset_client_picks_up_a_changed_environment(monkeypatch):
    configure(monkeypatch, **{API_KEY_ENV_VAR: "sk-first"})
    assert get_client().api_key == "sk-first"

    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-second")
    assert get_client().api_key == "sk-first"          # still cached, as intended

    reset_client()
    assert get_client().api_key == "sk-second"
