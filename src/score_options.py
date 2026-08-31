"""Option-label likelihood scoring utilities.

The original experiments score answer-option letters by next-token likelihood,
considering both bare labels such as ``A`` and space-prefixed labels such as
`` A`` where tokenizer behavior requires it, then normalizing over displayed
answer options. This module documents the public scoring contract; the archived
research scripts contain the exact model-calling implementation used for the
paper runs.
"""

from __future__ import annotations


def normalize_option_scores(raw_scores: dict[str, float], option_labels: list[str]) -> dict[str, float]:
    selected = {label: float(raw_scores.get(label, 0.0)) for label in option_labels}
    total = sum(selected.values())
    if total <= 0:
        uniform = 1.0 / len(option_labels)
        return {label: uniform for label in option_labels}
    return {label: value / total for label, value in selected.items()}


def label_token_variants(label: str) -> tuple[str, str]:
    return label, f" {label}"
