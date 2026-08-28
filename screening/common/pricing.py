"""What a call costs, in USD.

The rates live here as constants, applied to every call whatever model id the gateway
reports. That is deliberate, and it is the second attempt: the first version kept rates
in a database table keyed on the model name, which silently priced nothing whenever the
gateway exposed the model under an id different from the one typed into the admin. A
constant cannot be mistyped, cannot fail to match, and cannot be blank.

Edit the three numbers below when the provider's rates change. Nothing else needs to
move — cost is computed from stored token counts every time it is displayed, so a rate
change re-prices runs that have already finished.

No Django imports: this sits next to llm.py and stays usable from a bare script.
"""

from decimal import Decimal

# USD per million tokens.
USD_PER_MILLION_INPUT = Decimal("0.20")
USD_PER_MILLION_CACHED_INPUT = Decimal("0.02")
USD_PER_MILLION_OUTPUT = Decimal("1.20")

RATE_SUMMARY = (
    f"${USD_PER_MILLION_INPUT:g} in / ${USD_PER_MILLION_CACHED_INPUT:g} cached / "
    f"${USD_PER_MILLION_OUTPUT:g} out per M tokens"
)

_PER_MILLION = Decimal(1_000_000)


def cost_usd(input_tokens: int, cached_input_tokens: int, output_tokens: int) -> Decimal:
    """USD for one call, or for a summed group of them — the formula is linear, so
    totalling tokens first and pricing once gives the same answer as pricing each call.

    Two counts are subsets, not extras, and both are easy to get wrong:

      * `input_tokens` is the TOTAL and includes the cached ones, so the cached share is
        subtracted before the full rate applies. Adding them instead would overcharge
        every repeated prompt prefix — which this project deliberately maximises, so the
        error would be large rather than marginal.
      * `reasoning_tokens` (not a parameter here, on purpose) is part of `output_tokens`
        and is already billed by it. It is stored for visibility and never priced again.
    """
    fresh_input = max(0, input_tokens - cached_input_tokens)
    total = (
        fresh_input * USD_PER_MILLION_INPUT
        + cached_input_tokens * USD_PER_MILLION_CACHED_INPUT
        + output_tokens * USD_PER_MILLION_OUTPUT
    )
    return total / _PER_MILLION


def format_usd(cost: Decimal) -> str:
    """Four decimals: at these rates a single screening call costs a few ten-thousandths
    of a dollar, and two decimals would render most of the app as $0.00."""
    return f"${cost:.4f}"
