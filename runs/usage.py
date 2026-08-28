"""What each LLM-calling step of a run spent: recording it, and reading it back.

The screening package times every call and counts its tokens but cannot store any of it —
it has no Django imports by design. This module is the other half: it opens a
`collect_usage()` block around a step and totals what falls out into
`ScreeningRun.step_costs`.

One entry per step, not one row per call. The number the recruiter is shown is a per-step
total sitting next to the button that moves them on, so per-call rows would be storage
nobody reads.

Only tokens and seconds are stored. Price is computed on read from the rates in
`screening/common/pricing.py`, which means correcting a rate corrects every run already
finished — and, unlike the version this replaces, there is no model id to match and
therefore nothing that can silently fail to price.
"""

from contextlib import contextmanager

from screening.common.llm import collect_usage
from screening.common.pricing import RATE_SUMMARY, cost_usd, format_usd

# Pipeline order. The keys are what `record_step` is called with and what `step_costs`
# is keyed by; the labels are what the pages show.
STEP_LABELS = {
    "jd_extraction": "JD extraction",
    "cv_structuring": "CV structuring",
    "stage1": "Stage 1 — hard filter",
    "stage2": "Stage 2 — ranking",
    "stage3": "Stage 3 — scoring",
}


@contextmanager
def record_step(run, step):
    """Total every LLM call made inside this block into `run.step_costs[step]`.

    Saved in a `finally`: a step that fails half way still spent what it spent, and the
    recruiter pressing the button again deserves to see that it cost something.
    """
    with collect_usage() as calls:
        try:
            yield
        finally:
            _save_step(run, step, calls)


def _save_step(run, step, calls) -> None:
    if not calls:
        # No call, no entry. `evaluate_hard_filters` short-circuits when the JD has no
        # must-haves — that is a step which never ran, not a step that cost $0.00, and
        # the two must not read the same on screen.
        return

    # Re-read first: another step of this run may have been written by a different
    # process since this object was loaded, and a whole-dict assignment would drop it.
    run.refresh_from_db(fields=["step_costs"])
    costs = dict(run.step_costs or {})
    costs[step] = {
        "calls": len(calls),
        "input_tokens": sum(call.input_tokens for call in calls),
        "cached_input_tokens": sum(call.cached_input_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        "reasoning_tokens": sum(call.reasoning_tokens for call in calls),
        "seconds": round(sum(call.duration_ms for call in calls) / 1000, 3),
        "model": calls[-1].model,
    }
    # Overwrites rather than accumulates: re-running a step replaces its work, so it must
    # replace its cost too.
    run.step_costs = costs
    run.save(update_fields=["step_costs"])


def format_seconds(seconds: float) -> str:
    """Seconds under a minute, minutes and seconds above it — a step over a large field
    runs into minutes, and "184.3 s" is a number the reader has to divide in their head."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes} min {remainder:02d} s"


def step_cost(run, step) -> dict | None:
    """What one step cost, ready to render. None when the step has not run."""
    data = (run.step_costs or {}).get(step)
    if not data:
        return None

    input_tokens = data.get("input_tokens", 0)
    cached = data.get("cached_input_tokens", 0)
    output_tokens = data.get("output_tokens", 0)
    seconds = data.get("seconds", 0)
    cost = cost_usd(input_tokens, cached, output_tokens)

    return {
        "step": step,
        "label": STEP_LABELS.get(step, step),
        "calls": data.get("calls", 0),
        "seconds": seconds,
        "seconds_display": format_seconds(seconds),
        "cost": cost,
        "cost_display": format_usd(cost),
        "input_display": f"{input_tokens:,}",
        "cached_display": f"{cached:,}",
        "output_display": f"{output_tokens:,}",
        "model": data.get("model", ""),
        # Printed next to the figure so the number is checkable rather than asserted.
        "rates": RATE_SUMMARY,
    }
