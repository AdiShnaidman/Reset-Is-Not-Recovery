"""Recovery criteria used by the paper analyses."""

from __future__ import annotations


def mro_success(mro_value: float | None, threshold: float = 0.95) -> bool:
    return mro_value is not None and float(mro_value) >= threshold


def restored_clean_context_like(reset_gap: float, tolerance: float = 0.05) -> bool:
    return abs(float(reset_gap)) <= tolerance


def contamination_rate(clean_correct: bool, reset_wrong_following: bool) -> bool:
    return bool(clean_correct and reset_wrong_following)
