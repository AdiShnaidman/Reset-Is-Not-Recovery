"""Clean-counterfactual tube projection diagnostic helpers."""

from __future__ import annotations


def linear_interpolate(clean_value: float, pressure_value: float, alpha: float) -> float:
    return (1.0 - float(alpha)) * float(clean_value) + float(alpha) * float(pressure_value)


def alpha_star(clean_value: float, pressure_value: float, target_value: float) -> float | None:
    denominator = float(pressure_value) - float(clean_value)
    if denominator == 0.0:
        return None
    return (float(target_value) - float(clean_value)) / denominator
