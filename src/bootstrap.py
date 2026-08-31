"""Bootstrap confidence interval helpers."""

from __future__ import annotations

import numpy as np


def percentile_interval(values, alpha: float = 0.05) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    low = 100.0 * alpha / 2.0
    high = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(array, low)), float(np.percentile(array, high))


def bootstrap_mean(values, samples: int = 10000, seed: int = 12345) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = percentile_interval(draws)
    return float(array.mean()), low, high
