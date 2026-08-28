"""Tests for LLM cost and inference-time tracking.

Three layers:

1. The pricing arithmetic (`ModelPrice.cost_for`) — pure, asserted against hand-computed
   euros. The two subset traps live here: a cached token is part of the input count, not
   extra to it, and a reasoning token is part of the output count.
2. The roll-up (`run_cost_summary`) — built from `LlmCall` rows written directly, so the
   grouping and the per-CV average can be pinned without running a stage.
3. The recording, through the real service functions. These patch the OpenAI *client*
   rather than `call_structured`, because patching `call_structured` bypasses the
   collector — which is exactly what every other test in the suite wants and what these
   tests must not do.
"""

import csv
import io
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from django.urls import reverse

from runs.models import LlmCall, ModelPrice
from runs.services import run_stage1, run_stage2
from runs.usage import candidate_cost_rows, format_cost, format_seconds, record_calls, run_cost_summary
from screening.common import llm

# The Stage 1 fixtures are the simplest full pipeline in the suite; reuse them rather
# than rebuilding a run, a JD and a candidate here.
from runs.tests import MUST_HAVE, NICE_TO_HAVE, add_candidate, run, verdict  # noqa: F401  (pytest fixture)

pytestmark = pytest.mark.django_db


def price(model_name="gpt-5.6-luna", input_rate="1.0000", cached_rate="0.1000", output_rate="4.0000"):
    return ModelPrice.objects.create(
        model_name=model_name,
        input_per_million=Decimal(input_rate),
        cached_input_per_million=Decimal(cached_rate),
        output_per_million=Decimal(output_rate),
    )


def usage(input_tokens=1000, cached=200, output_tokens=300, reasoning=50):
    return SimpleNamespace(
        input_tokens=input_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=0),
        output_tokens=output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        total_tokens=input_tokens + output_tokens,
    )


def stub_llm(*payloads, **usage_kwargs):
    """Patch the OpenAI client so the real `call_structured` path runs, usage and all.

    Deliberately NOT a patch of `call_structured`: that is the seam every other test uses,
    and it short-circuits the very collector these tests exist to exercise.
    """
    fake = MagicMock()
    fake.responses.create.side_effect = [
        SimpleNamespace(output_text=json.dumps(payload), usage=usage(**usage_kwargs))
        for payload in payloads
    ]
    return patch.object(llm, "get_client", return_value=fake)


def add_call(run, step="stage1", candidate=None, input_tokens=1000, cached=200,
             output_tokens=300, reasoning=0, duration_ms=2000, model_name="gpt-5.6-luna"):
    return LlmCall.objects.create(
        run=run, candidate=candidate, step=step, model_name=model_name,
        input_tokens=input_tokens, cached_input_tokens=cached, output_tokens=output_tokens,
        reasoning_tokens=reasoning, duration_ms=duration_ms,
    )


# --- Layer 1: the pricing arithmetic -------------------------------------------

def test_cached_tokens_are_billed_at_the_cached_rate_not_the_input_rate():
    """input_tokens INCLUDES the cached ones. Adding the two instead of subtracting would
    overcharge every repeated prompt prefix — which this project deliberately maximises."""
    rate = price(input_rate="1.0000", cached_rate="0.1000", output_rate="4.0000")

    # 800 fresh @ €1/Mtok + 200 cached @ €0.10/Mtok + 300 out @ €4/Mtok
    expected = Decimal(800 + 20 + 1200) / Decimal(1_000_000)
    assert rate.cost_for(1000, 200, 300) == expected


def test_a_fully_cached_prompt_costs_only_the_cached_rate():
    rate = price(input_rate="1.0000", cached_rate="0.1000", output_rate="0.0000")

    assert rate.cost_for(1000, 1000, 0) == Decimal(100) / Decimal(1_000_000)


def test_no_cached_tokens_bills_the_whole_input_at_the_full_rate():
    rate = price(input_rate="2.0000", cached_rate="0.2000", output_rate="0.0000")

    assert rate.cost_for(500, 0, 0) == Decimal(1000) / Decimal(1_000_000)


def test_a_cached_count_larger_than_the_input_count_cannot_go_negative():
    """Shouldn't happen, but a gateway reporting nonsense must not produce a credit."""
    rate = price(input_rate="1.0000", cached_rate="0.0000", output_rate="0.0000")

    assert rate.cost_for(100, 500, 0) == Decimal(0)


def test_reasoning_tokens_do_not_inflate_the_cost(run):
    """They are a subset of output_tokens and are already billed there. Storing them is
    for visibility only."""
    price()
    add_call(run, output_tokens=300, reasoning=0)
    without = run_cost_summary(run)["total"]["cost"]

    LlmCall.objects.all().delete()
    add_call(run, output_tokens=300, reasoning=250)

    assert run_cost_summary(run)["total"]["cost"] == without


# --- Layer 2: the roll-up ------------------------------------------------------

def test_steps_appear_in_pipeline_order_not_insertion_order(run):
    add_call(run, step="stage3")
    add_call(run, step="jd_extraction")
    add_call(run, step="stage1")

    assert [r["step"] for r in run_cost_summary(run)["rows"]] == [
        "jd_extraction", "stage1", "stage3"
    ]


def test_a_step_that_has_not_run_is_omitted_rather_than_shown_as_zero(run):
    add_call(run, step="stage1")

    assert [r["step"] for r in run_cost_summary(run)["rows"]] == ["stage1"]


def test_per_cv_time_averages_over_distinct_candidates(run):
    a, b = add_candidate(run, "a.pdf"), add_candidate(run, "b.pdf")
    add_call(run, step="stage1", candidate=a, duration_ms=3000)
    add_call(run, step="stage1", candidate=b, duration_ms=5000)

    row = run_cost_summary(run)["rows"][0]
    assert row["seconds"] == 8.0
    assert row["seconds_per_cv"] == 4.0
    assert row["per_cv_display"] == "4.0 s"


def test_jd_extraction_has_no_per_cv_figure(run):
    """It is one call for the whole run, not one per CV, so an average would be a lie."""
    add_call(run, step="jd_extraction", candidate=None)

    row = run_cost_summary(run)["rows"][0]
    assert row["candidates"] == 0
    assert row["seconds_per_cv"] is None
    assert row["per_cv_display"] == "—"


def test_with_no_rate_configured_cost_is_none_not_zero(run):
    """€0.00 reads as "this was free". It wasn't — the rate is simply unknown."""
    add_call(run)
    summary = run_cost_summary(run)

    assert summary["rows"][0]["cost"] is None
    assert summary["rows"][0]["cost_display"] == "—"
    assert summary["total"]["cost"] is None
    assert summary["unpriced_models"] == ["gpt-5.6-luna"]


def test_entering_a_rate_afterwards_prices_a_run_that_already_finished(run):
    """The reason cost is computed on read rather than stored: you cannot know the rate
    before making a call, so the normal order is run first, price second."""
    add_call(run, input_tokens=1000, cached=0, output_tokens=0)
    assert run_cost_summary(run)["total"]["cost"] is None

    price(input_rate="3.0000")

    assert run_cost_summary(run)["total"]["cost"] == Decimal(3000) / Decimal(1_000_000)
    assert run_cost_summary(run)["unpriced_models"] == []


def test_the_total_sums_every_step(run):
    price()
    add_call(run, step="stage1", duration_ms=1000)
    add_call(run, step="stage2", duration_ms=2500)
    summary = run_cost_summary(run)

    assert summary["total"]["calls"] == 2
    assert summary["total"]["seconds"] == 3.5
    assert summary["total"]["cost"] == sum(r["cost"] for r in summary["rows"])


def test_an_unpriced_model_makes_the_total_unpriced_too(run):
    """A total that silently omits one step's cost looks like a real number and is not."""
    price(model_name="gpt-5.6-luna")
    add_call(run, step="stage1", model_name="gpt-5.6-luna")
    add_call(run, step="stage2", model_name="some-other-model")
    summary = run_cost_summary(run)

    assert summary["rows"][0]["cost"] is not None
    assert summary["rows"][1]["cost"] is None
    assert summary["total"]["cost"] is None


def test_an_empty_run_reports_no_calls(run):
    summary = run_cost_summary(run)

    assert summary["has_calls"] is False
    assert summary["rows"] == []


def test_per_candidate_rows_carry_seconds_and_cost_per_step(run):
    price()
    candidate = add_candidate(run, "a.pdf")
    add_call(run, step="stage1", candidate=candidate, duration_ms=2000)
    add_call(run, step="stage2", candidate=candidate, duration_ms=4000)

    rows = candidate_cost_rows(run)
    assert rows[candidate.pk]["stage1"]["seconds"] == 2.0
    assert rows[candidate.pk]["stage2"]["seconds"] == 4.0
    assert rows[candidate.pk]["stage1"]["cost"] > 0


# --- Layer 2: display formatting -----------------------------------------------

@pytest.mark.parametrize("seconds, expected", [
    (0.0, "0.0 s"), (3.24, "3.2 s"), (59.9, "59.9 s"),
    (60, "1 min 00 s"), (131.4, "2 min 11 s"),
])
def test_seconds_read_as_minutes_once_past_a_minute(seconds, expected):
    assert format_seconds(seconds) == expected


def test_cost_shows_four_decimals_because_one_call_costs_a_fraction_of_a_cent():
    assert format_cost(Decimal("0.00432")) == "€0.0043"
    assert format_cost(None) == "—"


# --- Layer 3: recording through the real service path --------------------------

def test_stage1_records_one_call_per_candidate(run):
    add_candidate(run, "a.pdf")
    add_candidate(run, "b.pdf")

    with stub_llm(verdict(True), verdict(False)):
        run_stage1(run)

    calls = LlmCall.objects.filter(run=run, step="stage1")
    assert calls.count() == 2
    assert {c.candidate.original_filename for c in calls} == {"a.pdf", "b.pdf"}
    assert all(c.input_tokens == 1000 and c.output_tokens == 300 for c in calls)
    assert all(c.duration_ms >= 0 for c in calls)


def test_a_jd_with_no_must_haves_records_no_calls_at_all(run):
    """`evaluate_hard_filters` returns early without calling the model. Recording a call
    there would invent a cost that was never incurred."""
    run.job_description.requirements = [NICE_TO_HAVE]
    run.job_description.save()
    add_candidate(run, "a.pdf")

    with stub_llm():
        run_stage1(run)

    assert LlmCall.objects.filter(run=run).count() == 0


def test_the_recorded_model_is_the_one_actually_called(run, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gateway-luna-x")
    add_candidate(run, "a.pdf")

    with stub_llm(verdict(True)):
        run_stage1(run)

    assert LlmCall.objects.get(run=run).model_name == "gateway-luna-x"


def test_calls_that_succeeded_before_a_failure_are_still_billed(run):
    """Stage 2 discards every score it collected when one candidate fails, but those calls
    were paid for. Recording inside the atomic block would roll them back and under-report
    what the run actually cost."""
    for name in ("a.pdf", "b.pdf"):
        add_candidate(run, name)
    with stub_llm(verdict(True), verdict(True)):
        run_stage1(run)                      # get two Stage 1 survivors into Stage 2
    run.current_stage = "stage2_pending"
    run.save()
    LlmCall.objects.filter(step="stage1").delete()

    fake = MagicMock()
    match = {"fit_score": 50, "fit_reasoning": "ok", "keyword_matches": []}
    fake.responses.create.side_effect = [
        SimpleNamespace(output_text=json.dumps(match), usage=usage()),
        RuntimeError("upstream 503"),
    ]

    with patch.object(llm, "get_client", return_value=fake):
        with pytest.raises(RuntimeError):
            run_stage2(run)

    assert LlmCall.objects.filter(run=run, step="stage2").count() == 1


def test_record_calls_persists_even_when_the_block_raises(run):
    candidate = add_candidate(run, "a.pdf")

    with stub_llm({"ok": True}):
        with pytest.raises(RuntimeError):
            with record_calls(run, "cv_structuring", candidate):
                llm.call_structured("i", "t", "candidate_data", {"type": "object"})
                raise RuntimeError("something after the call")

    assert LlmCall.objects.filter(run=run, step="cv_structuring").count() == 1


# --- The UI panel and the exports ----------------------------------------------

def stage1_html(client, run):
    return client.get(reverse("runs:stage1_review", args=[run.pk])).content.decode()


def test_the_cost_panel_shows_each_step_with_its_time_and_price(client, run):
    price()
    candidate = add_candidate(run, "a.pdf")
    add_call(run, step="stage1", candidate=candidate, duration_ms=3200)

    html = stage1_html(client, run)

    assert "Run cost" in html
    assert "Stage 1 — hard filter" in html
    assert "3.2 s" in html                       # both the step total and its per-CV figure
    assert "€" in html


def test_the_panel_names_the_unpriced_model_instead_of_showing_zero(client, run):
    """The failure mode this avoids: a recruiter reading €0.00 and believing it."""
    add_call(run, step="stage1", candidate=add_candidate(run, "a.pdf"))

    html = stage1_html(client, run)

    assert "No rate is set for" in html
    assert "gpt-5.6-luna" in html
    assert "€0.00" not in html


def test_the_panel_is_absent_until_a_call_has_been_made(client, run):
    add_candidate(run, "a.pdf")

    assert "Run cost" not in stage1_html(client, run)


def test_the_panel_appears_on_every_review_page(client, run):
    price()
    add_call(run, step="stage1", candidate=add_candidate(run, "a.pdf"))
    run.current_stage = "review"
    run.save()

    for name in ("runs:stage2_review", "runs:stage3_review", "runs:final_review"):
        html = client.get(reverse(name, args=[run.pk])).content.decode()
        assert "Run cost" in html, name


def test_the_csv_carries_per_candidate_seconds_and_cost(client, run):
    """The panel averages per step; this is the per-CV detail behind it."""
    price()
    candidate = add_candidate(run, "a.pdf")
    add_call(run, step="stage1", candidate=candidate, duration_ms=3500)

    response = client.get(reverse("runs:export_csv", args=[run.pk]))
    rows = list(csv.DictReader(io.StringIO(response.content.decode())))

    assert rows[0]["stage1_seconds"] == "3.5"
    assert float(rows[0]["stage1_cost"]) > 0
    assert rows[0]["stage3_seconds"] == ""        # never ran for this candidate


def test_the_csv_keeps_its_existing_columns_in_place(client, run):
    """The new columns are appended; anything that counts columns must not shift."""
    add_candidate(run, "a.pdf")

    response = client.get(reverse("runs:export_csv", args=[run.pk]))
    header = next(csv.reader(io.StringIO(response.content.decode())))

    assert header[:5] == ["position", "filename", "candidate_name", "email", "outcome"]
    assert header[20] == "overrides"             # the last of the original 21
    assert header[21] == "structuring_seconds"   # the first of the new ones
