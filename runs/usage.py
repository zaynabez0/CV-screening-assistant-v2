"""LLM cost and latency: recording it, and rolling it up for display.

The screening package times every call and counts its tokens but cannot store any of it —
it has no Django imports by design. This module is the other half: it opens a
`collect_usage()` block around each call site and writes what falls out to `LlmCall`.

Cost is never stored, only computed. Rates live in `ModelPrice`, editable in the admin,
and pricing a run from its stored token counts means entering a rate for the first time
prices every run that already finished — which is the normal order of events, since you
cannot know what a call cost until you have made one. The cost of that choice is that
correcting a rate rewrites history; the token counts, which are the facts, never move.
"""

from contextlib import contextmanager
from decimal import Decimal

from django.db.models import Count, Sum

from screening.common.llm import collect_usage

from .models import LlmCall, ModelPrice

# Rendered in the panel and the exports. Pipeline order, not alphabetical: the point of
# the table is to show where the time and money go as a run progresses.
STEP_LABELS = dict(LlmCall.STEP_CHOICES)


@contextmanager
def record_calls(run, step, candidate=None):
    """Persist one LlmCall for every LLM call made inside this block.

    Writing happens in a `finally`, and deliberately outside any surrounding transaction:
    a call that succeeded was paid for even when the work is later discarded. Stage 2
    throws away every score it collected if one candidate fails, and those calls must
    still appear on the bill.
    """
    with collect_usage() as calls:
        try:
            yield
        finally:
            LlmCall.objects.bulk_create([
                LlmCall(
                    run=run,
                    candidate=candidate,
                    step=step,
                    model_name=call.model,
                    input_tokens=call.input_tokens,
                    cached_input_tokens=call.cached_input_tokens,
                    output_tokens=call.output_tokens,
                    reasoning_tokens=call.reasoning_tokens,
                    duration_ms=call.duration_ms,
                )
                for call in calls
            ])


def format_tokens(count: int) -> str:
    return f"{count:,}"


def format_seconds(seconds: float) -> str:
    """Seconds under a minute, minutes and seconds above it.

    A stage over a large field runs into minutes, and "184.3 s" is a number the reader has
    to divide in their head.
    """
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes} min {remainder:02d} s"


def format_cost(cost) -> str:
    """Four decimal places: one screening call costs a fraction of a cent, and rounding to
    two would show most of this table as €0.00."""
    return "—" if cost is None else f"€{cost:.4f}"


def _cost_of(token_groups, prices):
    """Price one step's calls, grouped by model.

    Returns None if any model involved has no rate — a partially priced total looks like a
    real number and is not one. Grouping by model matters because OPENAI_MODEL can change
    between runs, or mid-run, and the rows would then be billed at the wrong rate.
    """
    total = Decimal(0)
    for group in token_groups:
        price = prices.get(group["model_name"])
        if price is None:
            return None
        total += price.cost_for(
            group["input_tokens"], group["cached_input_tokens"], group["output_tokens"]
        )
    return total


def run_cost_summary(run) -> dict:
    """Per-step calls, tokens, time and cost for one run, plus a total.

    Two queries regardless of how many steps have run: one grouped by step for the
    counters, one grouped by step and model for the pricing.
    """
    aggregates = {
        "calls": Count("pk"),
        # distinct, and NULLs are not counted — so JD extraction, which has no candidate,
        # correctly reports zero and gets no per-CV figure.
        "candidates": Count("candidate", distinct=True),
        "input_tokens": Sum("input_tokens"),
        "cached_input_tokens": Sum("cached_input_tokens"),
        "output_tokens": Sum("output_tokens"),
        "duration_ms": Sum("duration_ms"),
    }
    by_step = {
        row["step"]: row
        for row in LlmCall.objects.filter(run=run).values("step").annotate(**aggregates)
    }
    by_step_and_model = {}
    for row in (LlmCall.objects.filter(run=run)
                .values("step", "model_name")
                .annotate(input_tokens=Sum("input_tokens"),
                          cached_input_tokens=Sum("cached_input_tokens"),
                          output_tokens=Sum("output_tokens"))):
        by_step_and_model.setdefault(row["step"], []).append(row)

    prices = {price.model_name: price for price in ModelPrice.objects.all()}
    models_used = sorted({row["model_name"] for rows in by_step_and_model.values() for row in rows})
    unpriced = [name for name in models_used if name not in prices]

    rows, totals = [], {"calls": 0, "tokens": 0, "cached": 0, "seconds": 0.0, "cost": Decimal(0)}
    any_unpriced = False

    for step in LlmCall.STEP_ORDER:
        data = by_step.get(step)
        if data is None:
            continue  # a step that has not run yet is left out rather than shown as zero
        tokens = data["input_tokens"] + data["output_tokens"]
        seconds = data["duration_ms"] / 1000
        candidates = data["candidates"]
        cost = _cost_of(by_step_and_model.get(step, []), prices)

        rows.append({
            "step": step,
            "label": STEP_LABELS[step],
            "calls": data["calls"],
            "candidates": candidates,
            "tokens": tokens,
            "cached_tokens": data["cached_input_tokens"],
            "seconds": seconds,
            "seconds_per_cv": seconds / candidates if candidates else None,
            "cost": cost,
            "tokens_display": format_tokens(tokens),
            "seconds_display": format_seconds(seconds),
            "per_cv_display": format_seconds(seconds / candidates) if candidates else "—",
            "cost_display": format_cost(cost),
        })

        totals["calls"] += data["calls"]
        totals["tokens"] += tokens
        totals["cached"] += data["cached_input_tokens"]
        totals["seconds"] += seconds
        if cost is None:
            any_unpriced = True
        else:
            totals["cost"] += cost

    total_cost = None if (any_unpriced or not rows) else totals["cost"]

    return {
        "rows": rows,
        "has_calls": bool(rows),
        "models_used": models_used,
        "unpriced_models": unpriced,
        "total": {
            "calls": totals["calls"],
            "tokens": totals["tokens"],
            "cached_tokens": totals["cached"],
            "seconds": totals["seconds"],
            "cost": total_cost,
            "tokens_display": format_tokens(totals["tokens"]),
            "seconds_display": format_seconds(totals["seconds"]),
            "cost_display": format_cost(total_cost),
        },
    }


def candidate_cost_rows(run) -> dict:
    """Per-candidate seconds and cost for each step, keyed by candidate id.

    The panel shows per-step averages; this is the per-CV detail behind them, for the
    exports. Shape: {candidate_id: {step: {"seconds": float, "cost": Decimal|None}}}.
    """
    prices = {price.model_name: price for price in ModelPrice.objects.all()}
    rows: dict[int, dict[str, dict]] = {}
    for call in LlmCall.objects.filter(run=run).exclude(candidate=None):
        entry = rows.setdefault(call.candidate_id, {}).setdefault(
            call.step, {"seconds": 0.0, "cost": Decimal(0), "priced": True}
        )
        entry["seconds"] += call.duration_ms / 1000
        price = prices.get(call.model_name)
        if price is None:
            entry["priced"] = False
        elif entry["priced"]:
            entry["cost"] += price.cost_for(
                call.input_tokens, call.cached_input_tokens, call.output_tokens
            )
    for steps in rows.values():
        for entry in steps.values():
            if not entry["priced"]:
                entry["cost"] = None
    return rows
