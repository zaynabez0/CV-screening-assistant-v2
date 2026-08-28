"""Tests for per-step cost recording and the progress bar.

Layer 1 pins `record_step`'s accumulation and the `step_cost` readout. Layer 2 goes
through a real service function, patching the OpenAI *client* rather than
`call_structured` — patching `call_structured` is the seam every other test in the suite
uses, and it short-circuits the very collector these tests exist to exercise. Layer 3
checks what actually reaches the page.
"""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from django.urls import reverse

from runs.services import run_stage1
from runs.usage import format_seconds, record_step, step_cost
from screening.common import llm
from screening.common.pricing import cost_usd

# The Stage 1 fixtures are the simplest full pipeline in the suite; reuse them rather
# than rebuilding a run, a JD and a candidate here.
from runs.tests import NICE_TO_HAVE, add_candidate, run, verdict  # noqa: F401  (pytest fixture)

pytestmark = pytest.mark.django_db


def usage(input_tokens=10_000, cached=4_000, output_tokens=2_000, reasoning=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=0),
        output_tokens=output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        total_tokens=input_tokens + output_tokens,
    )


def stub_llm(*payloads, **usage_kwargs):
    """Patch the OpenAI client so the real `call_structured` path runs, usage and all."""
    fake = MagicMock()
    fake.responses.create.side_effect = [
        SimpleNamespace(output_text=json.dumps(payload), usage=usage(**usage_kwargs))
        for payload in payloads
    ]
    return patch.object(llm, "get_client", return_value=fake)


def call(schema_name="hard_filter_result"):
    return llm.call_structured("instructions", "input", schema_name, {"type": "object"})


# --- Layer 1: recording --------------------------------------------------------

def test_a_step_totals_all_of_its_calls_into_one_entry(run):
    with stub_llm({"a": 1}, {"b": 2}, {"c": 3}):
        with record_step(run, "stage1"):
            call(); call(); call()

    run.refresh_from_db()
    entry = run.step_costs["stage1"]
    assert entry["calls"] == 3
    assert entry["input_tokens"] == 30_000
    assert entry["cached_input_tokens"] == 12_000
    assert entry["output_tokens"] == 6_000


def test_a_step_that_made_no_call_records_nothing(run):
    """Stage 1 short-circuits when the JD has no must-haves. That is a step which never
    ran, not a step that cost $0.00, and the two must not read the same on screen."""
    with record_step(run, "stage1"):
        pass

    run.refresh_from_db()
    assert run.step_costs == {}
    assert step_cost(run, "stage1") is None


def test_calls_made_before_a_failure_are_still_recorded(run):
    """The money was spent whether or not the work survived."""
    with stub_llm({"a": 1}):
        with pytest.raises(RuntimeError):
            with record_step(run, "stage2"):
                call()
                raise RuntimeError("upstream 503")

    run.refresh_from_db()
    assert run.step_costs["stage2"]["calls"] == 1


def test_re_running_a_step_replaces_its_cost_rather_than_adding_to_it(run):
    with stub_llm({"a": 1}, {"b": 2}):
        with record_step(run, "stage1"):
            call()
        with record_step(run, "stage1"):
            call()

    run.refresh_from_db()
    assert run.step_costs["stage1"]["calls"] == 1


def test_recording_one_step_does_not_disturb_another(run):
    with stub_llm({"a": 1}, {"b": 2}):
        with record_step(run, "jd_extraction"):
            call()
        with record_step(run, "stage1"):
            call()

    run.refresh_from_db()
    assert set(run.step_costs) == {"jd_extraction", "stage1"}


# --- Layer 1: the readout ------------------------------------------------------

def test_step_cost_prices_the_stored_tokens(run):
    with stub_llm({"a": 1}):
        with record_step(run, "stage1"):
            call()

    priced = step_cost(run, "stage1")
    assert priced["cost"] == cost_usd(10_000, 4_000, 2_000)
    assert priced["cost_display"] == "$0.0037"
    assert priced["calls"] == 1
    assert priced["input_display"] == "10,000"
    assert priced["cached_display"] == "4,000"


def test_step_cost_is_none_before_the_step_runs(run):
    assert step_cost(run, "stage3") is None


def test_changing_a_rate_reprices_a_run_that_already_finished(run, monkeypatch):
    """Cost is computed from stored tokens, never frozen — the reason the previous
    version's admin-entered rates could leave a finished run permanently unpriced."""
    with stub_llm({"a": 1}):
        with record_step(run, "stage1"):
            call()
    before = step_cost(run, "stage1")["cost"]

    monkeypatch.setattr("screening.common.pricing.USD_PER_MILLION_OUTPUT", Decimal("2.40"))

    assert step_cost(run, "stage1")["cost"] > before


@pytest.mark.parametrize("seconds, expected", [
    (0.0, "0.0 s"), (3.24, "3.2 s"), (59.9, "59.9 s"),
    (60, "1 min 00 s"), (131.4, "2 min 11 s"),
])
def test_seconds_read_as_minutes_once_past_a_minute(seconds, expected):
    assert format_seconds(seconds) == expected


# --- Layer 2: through a real stage ---------------------------------------------

def test_stage1_records_one_entry_covering_every_candidate(run):
    add_candidate(run, "a.pdf")
    add_candidate(run, "b.pdf")

    with stub_llm(verdict(True), verdict(False)):
        run_stage1(run)

    run.refresh_from_db()
    entry = run.step_costs["stage1"]
    assert entry["calls"] == 2
    assert entry["seconds"] >= 0
    assert step_cost(run, "stage1")["cost"] > 0


def test_a_jd_with_no_must_haves_costs_nothing_because_it_calls_nothing(run):
    run.job_description.requirements = [NICE_TO_HAVE]
    run.job_description.save()
    add_candidate(run, "a.pdf")

    with stub_llm():
        run_stage1(run)

    run.refresh_from_db()
    assert step_cost(run, "stage1") is None


# --- Layer 3: what reaches the page --------------------------------------------

def test_the_price_of_the_step_shows_above_the_confirm_button(run):
    add_candidate(run, "a.pdf")
    with stub_llm(verdict(True)):
        run_stage1(run)

    html = client_get(run, "runs:stage1_review")

    assert "This step cost" in html
    assert "$0.0037" in html
    assert "$0.20 in / $0.02 cached / $1.20 out per M tokens" in html  # checkable by hand


def test_no_price_line_before_the_step_has_run(run):
    add_candidate(run, "a.pdf")

    assert "This step cost" not in client_get(run, "runs:stage1_review")


def test_the_progress_bar_renders_at_the_jobs_actual_percentage(run):
    add_candidate(run, "a.pdf")
    run.mark_job_started("stage1", total=12)
    run.mark_job_progress(3)

    html = client_get(run, "runs:stage1_review")

    assert 'class="bar-fill"' in html
    assert "width: 25%" in html                      # 3 of 12
    assert 'aria-valuenow="25"' in html
    assert "3 / 12" in html


def test_the_bar_is_absent_when_no_job_is_running(run):
    add_candidate(run, "a.pdf")

    assert 'class="bar-fill"' not in client_get(run, "runs:stage1_review")


def test_a_failure_banner_belongs_only_to_the_step_that_failed(run):
    """Stage 1's banner used to have no job_stage guard, so a failed Stage 2 job showed
    up as 'Stage 1 failed' whenever someone navigated back."""
    add_candidate(run, "a.pdf")
    run.mark_job_started("stage2", total=1)
    run.mark_job_failed("upstream 503")

    assert "Stage 1 failed" not in client_get(run, "runs:stage1_review")


def client_get(run, url_name):
    from django.test import Client
    return Client().get(reverse(url_name, args=[run.pk])).content.decode()
