"""How the shared LLM client is configured from the environment. No network calls.

Three variables, one name each, all read straight from the environment:
KEY_REGULATOR_API_KEY, MODEL and the optional BASE_URL. Every stage that
calls an LLM goes through this one helper, so a mistake here breaks all five at once.

Blank is treated as unset for all three, which matters because `.env.example` ships
every line present and empty — a copied-but-unfilled template must fail with a message
naming the variable, not with an empty key or an empty model id sent to the API.
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


@pytest.fixture(autouse=True)
def clean_client(monkeypatch):
    """No cached client leaks between tests, and nothing leaks in from the shell or .env.

    This runs after the root conftest fixture and deliberately undoes it: these tests own
    the environment, including the "nothing is configured" case.
    """
    for name in (API_KEY_ENV_VAR, MODEL_ENV_VAR, BASE_URL_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    reset_client()
    yield
    reset_client()


# --- The API key ----------------------------------------------------------------

def test_the_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-gateway")

    assert get_client().api_key == "sk-gateway"


def test_no_key_names_the_variable_to_set(monkeypatch):
    """The SDK's own error names OPENAI_API_KEY, which this project does not read — so
    it sends someone hunting for a variable that would have no effect."""
    with pytest.raises(RuntimeError) as exc:
        get_client()

    assert API_KEY_ENV_VAR in str(exc.value)


def test_an_empty_key_is_treated_as_unset(monkeypatch):
    """`KEY_REGULATOR_API_KEY=` is how the template ships. Copying it without filling it
    in must fail here, naming the variable — not at the API with a 401."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "")

    with pytest.raises(RuntimeError) as exc:
        get_client()

    assert API_KEY_ENV_VAR in str(exc.value)


def test_a_whitespace_only_key_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "   ")

    with pytest.raises(RuntimeError):
        get_client()


def test_surrounding_whitespace_is_stripped_from_the_key(monkeypatch):
    """Copy-pasting a key into a .env often brings a trailing space with it."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "  sk-gateway  ")

    assert get_client().api_key == "sk-gateway"


# --- Base URL --------------------------------------------------------------------

def test_the_gateway_base_url_reaches_the_client(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-x")
    monkeypatch.setenv(BASE_URL_ENV_VAR, "https://gateway.example.com/v1")

    assert "gateway.example.com" in str(get_client().base_url)


def test_without_a_base_url_openai_itself_is_used(monkeypatch):
    """The base URL is the one optional setting: no gateway means talk to OpenAI."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-x")

    assert "api.openai.com" in str(get_client().base_url)


def test_a_blank_base_url_falls_back_to_the_default(monkeypatch):
    """`BASE_URL=` with nothing after it is how the template ships. The SDK
    resolves that to an EMPTY base URL and never applies its own default — whether the
    argument is omitted or passed as None — so every request would go nowhere. We pass
    a concrete URL to make that impossible."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-x")
    monkeypatch.setenv(BASE_URL_ENV_VAR, "")

    assert "api.openai.com" in str(get_client().base_url)


def test_a_whitespace_only_base_url_falls_back_too(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-x")
    monkeypatch.setenv(BASE_URL_ENV_VAR, "   ")

    assert "api.openai.com" in str(get_client().base_url)


def test_surrounding_whitespace_is_stripped_from_the_gateway_url(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV_VAR, "  https://gateway.example.com/v1  ")

    assert get_base_url() == "https://gateway.example.com/v1"


# --- Model id ----------------------------------------------------------------------

def test_the_model_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "internal-luna-v2")

    assert get_model() == "internal-luna-v2"


def test_no_model_names_the_variable_to_set():
    """There is no default on purpose: which model this runs on is a deployment
    decision. A wrong default would fail per candidate, mid-run, instead of up front."""
    with pytest.raises(RuntimeError) as exc:
        get_model()

    assert MODEL_ENV_VAR in str(exc.value)


def test_an_empty_model_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "")

    with pytest.raises(RuntimeError) as exc:
        get_model()

    assert MODEL_ENV_VAR in str(exc.value)


def test_surrounding_whitespace_is_stripped_from_the_model(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "  internal-luna-v2  ")

    assert get_model() == "internal-luna-v2"


def test_call_structured_sends_the_configured_model(monkeypatch):
    """The configured id has to actually reach the request, not just get_model()."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-x")
    monkeypatch.setenv(MODEL_ENV_VAR, "internal-luna-v2")

    fake = MagicMock()
    fake.responses.create.return_value = MagicMock(output_text='{"ok": true}')
    with patch.object(llm, "get_client", return_value=fake):
        result = call_structured("instructions", "input", "schema_name", SCHEMA)

    assert result == {"ok": True}
    assert fake.responses.create.call_args.kwargs["model"] == "internal-luna-v2"


def test_call_structured_still_sends_the_documented_request_shape(monkeypatch):
    """Structured Outputs, strict — unchanged by the configuration work."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-x")
    monkeypatch.setenv(MODEL_ENV_VAR, "internal-luna-v2")

    fake = MagicMock()
    fake.responses.create.return_value = MagicMock(output_text='{"ok": true}')
    with patch.object(llm, "get_client", return_value=fake):
        call_structured("shared prefix", "per-call text", "my_schema", SCHEMA)

    kwargs = fake.responses.create.call_args.kwargs
    assert kwargs["model"] == "internal-luna-v2"
    assert kwargs["instructions"] == "shared prefix"     # the cached prefix
    assert kwargs["input"] == "per-call text"
    assert kwargs["text"] == {"format": {"type": "json_schema", "name": "my_schema",
                                         "schema": SCHEMA, "strict": True}}


# --- Client caching -------------------------------------------------------------------

def test_the_client_is_built_once_and_reused(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-x")

    assert get_client() is get_client()


def test_reset_client_picks_up_a_changed_environment(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-first")
    assert get_client().api_key == "sk-first"

    monkeypatch.setenv(API_KEY_ENV_VAR, "sk-second")
    assert get_client().api_key == "sk-first"          # still cached, as intended

    reset_client()
    assert get_client().api_key == "sk-second"
