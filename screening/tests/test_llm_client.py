"""How the shared LLM client is configured from the environment. No network calls.

Covers the gateway case: a sub-key issued under its own variable name, routed through
a custom base URL, possibly exposing the model under a different id. Every stage that
calls an LLM goes through this one helper, so a mistake here breaks all four at once.
"""

from unittest.mock import MagicMock, patch

import pytest

from screening.common import llm
from screening.common.llm import (
    API_KEY_ENV_VARS,
    DEFAULT_MODEL,
    call_structured,
    get_base_url,
    get_client,
    get_model,
    reset_client,
)

SCHEMA = {"type": "object", "properties": {}, "required": []}


@pytest.fixture(autouse=True)
def clean_client(monkeypatch):
    """No cached client leaks between tests, and no real key leaks in from the shell."""
    for name in API_KEY_ENV_VARS + ("OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    reset_client()
    yield
    reset_client()


# --- Which environment variable supplies the key -------------------------------

def test_the_standard_variable_is_used(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-standard")

    assert get_client().api_key == "sk-standard"


def test_the_gateway_variable_is_used_when_the_standard_one_is_absent(monkeypatch):
    """The deployment case: the key is issued under the gateway's own variable name."""
    monkeypatch.setenv("KEY_REGULATOR_API_KEY", "sk-gateway")

    assert get_client().api_key == "sk-gateway"


def test_the_standard_variable_wins_when_both_are_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-standard")
    monkeypatch.setenv("KEY_REGULATOR_API_KEY", "sk-gateway")

    assert get_client().api_key == "sk-standard"


def test_an_empty_variable_is_treated_as_unset(monkeypatch):
    """`export OPENAI_API_KEY=` is a common mistake — fall through, don't send "" as a key."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("KEY_REGULATOR_API_KEY", "sk-gateway")

    assert get_client().api_key == "sk-gateway"


def test_no_key_at_all_names_every_accepted_variable(monkeypatch):
    """The SDK's own error only mentions OPENAI_API_KEY, which sends someone using the
    gateway variable hunting for the wrong thing."""
    with pytest.raises(RuntimeError) as exc:
        get_client()

    message = str(exc.value)
    for name in API_KEY_ENV_VARS:
        assert name in message
    assert "OPENAI_BASE_URL" in message          # points at the gateway setting too


# --- Base URL --------------------------------------------------------------------

def test_the_gateway_base_url_reaches_the_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")

    assert "gateway.example.com" in str(get_client().base_url)


def test_without_a_base_url_the_sdk_default_is_used(monkeypatch):
    """Anyone not using a gateway must be completely unaffected by this change."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    assert "api.openai.com" in str(get_client().base_url)


def test_a_blank_base_url_falls_back_to_the_default(monkeypatch):
    """`OPENAI_BASE_URL=` with nothing after it is easy to leave in a .env. The SDK
    resolves that to an EMPTY base URL and never applies its own default — whether the
    argument is omitted or passed as None — so every request would go nowhere. We pass
    a concrete URL to make that impossible."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    assert "api.openai.com" in str(get_client().base_url)


def test_a_whitespace_only_base_url_falls_back_too(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("OPENAI_BASE_URL", "   ")

    assert "api.openai.com" in str(get_client().base_url)


def test_surrounding_whitespace_is_stripped_from_the_gateway_url(monkeypatch):
    """Copy-pasting a URL into a .env often brings a trailing space with it."""
    monkeypatch.setenv("OPENAI_BASE_URL", "  https://gateway.example.com/v1  ")

    assert get_base_url() == "https://gateway.example.com/v1"


# --- Model id ----------------------------------------------------------------------

def test_the_model_defaults_to_the_one_claude_md_specifies():
    assert get_model() == DEFAULT_MODEL == "gpt-5.6-luna"


def test_the_model_can_be_overridden_for_a_gateway(monkeypatch):
    """A gateway commonly fronts the same model under its own id."""
    monkeypatch.setenv("OPENAI_MODEL", "internal-luna-v2")

    assert get_model() == "internal-luna-v2"


def test_an_empty_model_variable_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "")

    assert get_model() == DEFAULT_MODEL


def test_call_structured_sends_the_resolved_model(monkeypatch):
    """The override has to actually reach the request, not just get_model()."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("OPENAI_MODEL", "internal-luna-v2")

    fake = MagicMock()
    fake.responses.create.return_value = MagicMock(output_text='{"ok": true}')
    with patch.object(llm, "get_client", return_value=fake):
        result = call_structured("instructions", "input", "schema_name", SCHEMA)

    assert result == {"ok": True}
    assert fake.responses.create.call_args.kwargs["model"] == "internal-luna-v2"


def test_call_structured_still_sends_the_documented_request_shape(monkeypatch):
    """Structured Outputs, strict — unchanged by the configuration work."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    fake = MagicMock()
    fake.responses.create.return_value = MagicMock(output_text='{"ok": true}')
    with patch.object(llm, "get_client", return_value=fake):
        call_structured("shared prefix", "per-call text", "my_schema", SCHEMA)

    kwargs = fake.responses.create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["instructions"] == "shared prefix"     # the cached prefix
    assert kwargs["input"] == "per-call text"
    assert kwargs["text"] == {"format": {"type": "json_schema", "name": "my_schema",
                                         "schema": SCHEMA, "strict": True}}


# --- Client caching -------------------------------------------------------------------

def test_the_client_is_built_once_and_reused(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    assert get_client() is get_client()


def test_reset_client_picks_up_a_changed_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-first")
    assert get_client().api_key == "sk-first"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-second")
    assert get_client().api_key == "sk-first"          # still cached, as intended

    reset_client()
    assert get_client().api_key == "sk-second"
