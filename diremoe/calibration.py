"""Validation-fitted scalar bias for calibrating binary logit margins."""
from __future__ import annotations

import numpy as np


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values.")
    return array


def sigmoid(value) -> np.ndarray:
    """Overflow-safe float64 sigmoid, including extreme finite margins."""
    x = np.asarray(value, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Sigmoid input must be finite.")
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def binary_nll(margin, y, bias: float = 0.0) -> float:
    """Compute mean binary cross-entropy from logit margins."""
    s = _vector(margin, "margin")
    labels = _vector(y, "y")
    if labels.shape != s.shape or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("y must have the margin shape and contain only 0/1.")
    if not np.isfinite(bias):
        raise ValueError("bias must be finite.")
    shifted = s + float(bias)
    losses = np.logaddexp(0.0, np.where(labels == 1, -shifted, shifted))
    return float(losses.mean())


def fit_bias(margin, y, l2: float = 0.1, bound: float = 3.0) -> float:
    """Fit bias by bisection on mean BCE + ``l2 * b**2 / 2`` in ``[-bound, bound]``."""
    s = _vector(margin, "margin")
    labels = _vector(y, "y")
    if labels.shape != s.shape or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("y must have the margin shape and contain only 0/1.")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative.")
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be finite and positive.")

    def gradient(b: float) -> float:
        return float(np.mean(sigmoid(s + b) - labels) + l2 * b)

    if abs(gradient(0.0)) <= 1e-14:
        return 0.0
    low, high = -float(bound), float(bound)
    if gradient(low) >= 0:
        return low
    if gradient(high) <= 0:
        return high
    for _ in range(80):
        middle = (low + high) / 2.0
        derivative = gradient(middle)
        if abs(derivative) < 1e-14 or high - low < 1e-12:
            return float(middle)
        if derivative < 0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)
