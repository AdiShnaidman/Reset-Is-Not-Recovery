"""Metric helpers used by the paper analyses."""

from __future__ import annotations

import math
from collections.abc import Mapping


def hysteresis_gap(reset_p_wrong: float, clean_p_wrong: float) -> float:
    return float(reset_p_wrong) - float(clean_p_wrong)


def wrong_following(predicted_label: str, advocated_wrong_label: str) -> bool:
    return str(predicted_label) == str(advocated_wrong_label)


def entropy(probabilities: Mapping[str, float]) -> float:
    total = float(sum(probabilities.values()))
    if total <= 0:
        return 0.0
    value = 0.0
    for probability in probabilities.values():
        normalized = float(probability) / total
        if normalized > 0:
            value -= normalized * math.log(normalized)
    return value


def max_probability(probabilities: Mapping[str, float]) -> float:
    if not probabilities:
        return 0.0
    total = float(sum(probabilities.values()))
    if total <= 0:
        return 0.0
    return max(float(value) / total for value in probabilities.values())
