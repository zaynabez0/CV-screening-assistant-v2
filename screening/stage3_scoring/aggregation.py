"""Aggregation — deterministic, outside the LLM call.

The model produces per-criterion judgements; the weighted arithmetic is ours. Keeping
the maths in Python means the final score is reproducible from the stored breakdown,
which is what makes a Stage 3 result auditable.
"""


def aggregate_score(criterion_scores: list, weighted_rubric: list) -> dict:
    weight_map = {r["requirement"]: r["weight"] for r in weighted_rubric}

    # SCORE_SCHEMA types `criterion` as a free-text string, so nothing stops the model
    # returning a reworded or invented criterion name. A direct weight_map[cs["criterion"]]
    # lookup raises KeyError there and kills the whole run over one bad response.
    # Unrecognised names are excluded from the arithmetic and reported instead, so the
    # score stays computable and the anomaly stays visible.
    returned = {cs["criterion"] for cs in criterion_scores}
    scored = [cs for cs in criterion_scores if cs["criterion"] in weight_map]
    unknown = [cs["criterion"] for cs in criterion_scores if cs["criterion"] not in weight_map]
    unscored = [name for name in weight_map if name not in returned]

    weighted_sum = sum(cs["score"] * weight_map[cs["criterion"]] for cs in scored)
    max_possible = 5 * sum(weight_map.values())

    return {
        # A criterion the model skipped contributes nothing to weighted_sum while still
        # counting in max_possible — i.e. treated as "no evidence", the conservative reading.
        "final_score": round(weighted_sum / max_possible * 100, 1) if max_possible else 0.0,
        "criterion_breakdown": criterion_scores,
        "low_confidence_flags": [cs["criterion"] for cs in criterion_scores if cs["confidence"] == "low"],
        "unscored_criteria": unscored,
        "unknown_criteria": unknown,
    }
