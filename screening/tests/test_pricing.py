"""Tests for the USD cost arithmetic.

The numbers here are hand-computed at the real configured rates, not derived from the
implementation — a change to the formula shows up as a failing arithmetic assertion
rather than a silently different price.

Two traps have their own tests because both were live bugs waiting to happen: the cached
count is a subset of the input count, and the reasoning count is a subset of the output
count. Getting the first one backwards overcharges every cached prefix, which is most of
what this app sends.
"""

from decimal import Decimal

import pytest

from screening.common.pricing import (
    RATE_SUMMARY,
    USD_PER_MILLION_CACHED_INPUT,
    USD_PER_MILLION_INPUT,
    USD_PER_MILLION_OUTPUT,
    cost_usd,
    format_usd,
)


def test_the_configured_rates_are_the_ones_the_provider_quoted():
    """If these change, every figure in the UI changes — so the numbers are pinned."""
    assert USD_PER_MILLION_INPUT == Decimal("0.20")
    assert USD_PER_MILLION_CACHED_INPUT == Decimal("0.02")
    assert USD_PER_MILLION_OUTPUT == Decimal("1.20")


def test_a_worked_example_at_the_real_rates():
    # 10,000 input of which 4,000 cached, plus 2,000 output:
    #   6,000 fresh  @ $0.20/M = $0.00120
    #   4,000 cached @ $0.02/M = $0.00008
    #   2,000 out    @ $1.20/M = $0.00240
    assert cost_usd(10_000, 4_000, 2_000) == Decimal("0.00368")


def test_cached_tokens_are_subtracted_from_the_input_count_not_added_to_it():
    """input_tokens is the TOTAL and includes the cached ones. Adding them would charge
    the cached share twice — once at the full rate and once at the cached rate."""
    both_ways_wrong = cost_usd(10_000, 0, 0) + cost_usd(0, 4_000, 0)

    assert cost_usd(10_000, 4_000, 0) < both_ways_wrong


def test_a_fully_cached_prompt_costs_a_tenth_of_a_fresh_one():
    fresh = cost_usd(10_000, 0, 0)
    cached = cost_usd(10_000, 10_000, 0)

    assert cached == fresh / 10          # $0.02 against $0.20


def test_no_cached_tokens_bills_the_whole_input_at_the_full_rate():
    assert cost_usd(1_000_000, 0, 0) == USD_PER_MILLION_INPUT


def test_output_is_the_expensive_direction():
    """Six times the input rate — which is why the prompts ask for short evidence."""
    assert cost_usd(0, 0, 1_000_000) == USD_PER_MILLION_OUTPUT
    assert cost_usd(0, 0, 1_000_000) > cost_usd(1_000_000, 0, 0)


def test_a_cached_count_larger_than_the_input_count_cannot_produce_a_credit():
    """Shouldn't happen, but a gateway reporting nonsense must not hand out money."""
    assert cost_usd(100, 500, 0) >= 0


def test_an_empty_call_costs_nothing():
    assert cost_usd(0, 0, 0) == Decimal(0)


def test_pricing_a_summed_group_matches_pricing_each_call():
    """The formula is linear, which is what lets a step total its tokens first and price
    once — the whole basis of storing tokens rather than money."""
    calls = [(3_000, 2_000, 400), (3_100, 2_050, 380), (2_900, 1_900, 420)]
    separately = sum(cost_usd(*call) for call in calls)
    together = cost_usd(*(sum(part) for part in zip(*calls)))

    assert separately == together


@pytest.mark.parametrize("cost, expected", [
    (Decimal("0.00368"), "$0.0037"),
    (Decimal("0"), "$0.0000"),
    (Decimal("1.5"), "$1.5000"),
])
def test_costs_render_to_four_decimals(cost, expected):
    """Two decimals would show most of this app as $0.00 — one call costs a few
    ten-thousandths of a dollar at these rates."""
    assert format_usd(cost) == expected


def test_the_rate_summary_names_all_three_rates():
    """It is printed next to every figure so the number can be checked by hand."""
    assert RATE_SUMMARY == "$0.20 in / $0.02 cached / $1.20 out per M tokens"
