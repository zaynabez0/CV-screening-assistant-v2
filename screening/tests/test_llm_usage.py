"""Tests for the usage collector in `screening.common.llm`.

The collector is how cost and latency escape a package that cannot touch the database.
Two things are worth pinning hard: that it never changes what `call_structured` returns —
68 mocks elsewhere depend on that — and that a response missing its `usage` block yields
zeros rather than garbage, because a MagicMock answers every attribute with another mock
and would otherwise be stored as a token count.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from screening.common import llm
from screening.common.llm import MODEL_ENV_VAR, call_structured, collect_usage

SCHEMA = {"type": "object", "additionalProperties": False, "properties": {}, "required": []}

MODEL = "recorded-model-id"


@pytest.fixture(autouse=True)
def pinned_model(monkeypatch):
    """The recorded model id is whatever the environment says, so pin it here rather
    than let a real OPENAI_MODEL in the shell or .env decide what this asserts."""
    monkeypatch.setenv(MODEL_ENV_VAR, MODEL)


def usage(input_tokens=1000, cached=200, output_tokens=300, reasoning=50):
    return SimpleNamespace(
        input_tokens=input_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=0),
        output_tokens=output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        total_tokens=input_tokens + output_tokens,
    )


def client_returning(*responses):
    """A stub OpenAI client whose `responses.create` yields each response in turn."""
    fake = MagicMock()
    fake.responses.create.side_effect = list(responses)
    return fake


def response(output_text='{"ok": true}', **usage_kwargs):
    return SimpleNamespace(output_text=output_text, usage=usage(**usage_kwargs))


def call(schema_name="hard_filter_result"):
    return call_structured("instructions", "input", schema_name, SCHEMA)


# --- what gets recorded --------------------------------------------------------

def test_a_call_inside_the_block_records_its_tokens_and_model():
    with patch.object(llm, "get_client", return_value=client_returning(response())):
        with collect_usage() as calls:
            call()

    assert len(calls) == 1
    recorded = calls[0]
    assert recorded.schema_name == "hard_filter_result"
    assert recorded.model == MODEL
    assert recorded.input_tokens == 1000
    assert recorded.cached_input_tokens == 200
    assert recorded.output_tokens == 300
    assert recorded.reasoning_tokens == 50


def test_the_return_value_is_unchanged_by_collection():
    """The whole reason this is a side channel: 68 test mocks and five call sites expect
    a bare parsed dict back."""
    with patch.object(llm, "get_client", return_value=client_returning(response())):
        with collect_usage():
            assert call() == {"ok": True}


def test_a_call_outside_any_block_records_nothing_and_does_not_raise():
    with patch.object(llm, "get_client", return_value=client_returning(response())):
        assert call() == {"ok": True}


def test_duration_is_measured_in_milliseconds():
    with patch.object(llm, "get_client", return_value=client_returning(response())):
        with collect_usage() as calls:
            call()

    assert isinstance(calls[0].duration_ms, int)
    assert calls[0].duration_ms >= 0


def test_every_call_in_the_block_is_recorded_in_order():
    responses = [response(), response(), response()]
    with patch.object(llm, "get_client", return_value=client_returning(*responses)):
        with collect_usage() as calls:
            call("candidate_data")
            call("candidate_match")
            call("candidate_scores")

    assert [c.schema_name for c in calls] == ["candidate_data", "candidate_match", "candidate_scores"]


# --- failure and absence -------------------------------------------------------

def test_calls_made_before_an_exception_are_still_recorded():
    """`record_calls` persists in a `finally`, and a call that succeeded was paid for even
    when the work around it is abandoned."""
    fake = MagicMock()
    fake.responses.create.side_effect = [response(), RuntimeError("upstream 503")]

    with patch.object(llm, "get_client", return_value=fake):
        with collect_usage() as calls:
            call()
            with pytest.raises(RuntimeError):
                call()

    assert len(calls) == 1


def test_a_response_with_no_usage_block_records_zeros():
    """A gateway may omit usage entirely. Zero is honest; a crash mid-run is not."""
    naked = SimpleNamespace(output_text='{"ok": true}', usage=None)
    with patch.object(llm, "get_client", return_value=client_returning(naked)):
        with collect_usage() as calls:
            call()

    assert calls[0].input_tokens == 0
    assert calls[0].output_tokens == 0


def test_a_magicmock_response_records_zeros_rather_than_mocks():
    """test_llm_client.py stubs the response as a bare MagicMock, which answers every
    attribute with another mock. Without the isinstance guard those land in the database
    as token counts."""
    mock_response = MagicMock(output_text='{"ok": true}')
    with patch.object(llm, "get_client", return_value=client_returning(mock_response)):
        with collect_usage() as calls:
            call()

    assert calls[0].input_tokens == 0
    assert calls[0].cached_input_tokens == 0
    assert calls[0].output_tokens == 0


# --- block scoping -------------------------------------------------------------

def test_the_block_stops_collecting_when_it_exits():
    with patch.object(llm, "get_client", return_value=client_returning(response(), response())):
        with collect_usage() as calls:
            call()
        call()                                  # outside now

    assert len(calls) == 1


def test_nested_blocks_do_not_leak_into_each_other():
    """Each stage wraps its own call; an inner block must not steal the outer one's
    entries, and the outer one must resume afterwards."""
    responses = [response(), response(), response()]
    with patch.object(llm, "get_client", return_value=client_returning(*responses)):
        with collect_usage() as outer:
            call("candidate_data")
            with collect_usage() as inner:
                call("candidate_match")
            call("candidate_scores")

    assert [c.schema_name for c in outer] == ["candidate_data", "candidate_scores"]
    assert [c.schema_name for c in inner] == ["candidate_match"]


def test_an_exception_escaping_the_block_still_restores_the_previous_collector():
    with patch.object(llm, "get_client", return_value=client_returning(response())):
        with pytest.raises(RuntimeError):
            with collect_usage():
                raise RuntimeError("boom")
        # If the collector had leaked, this call would be recorded somewhere invisible.
        with collect_usage() as calls:
            call()

    assert len(calls) == 1
