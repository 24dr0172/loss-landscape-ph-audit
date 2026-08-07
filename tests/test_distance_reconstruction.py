#!/usr/bin/env python3
"""Standalone numerical checks for the shared increment-Gram distance path."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_common import (  # noqa: E402
    distance_from_step_gram,
    gram_reconstruction_error,
    step_lengths_from_gram,
)


def direct_pairwise_distance(X: np.ndarray) -> np.ndarray:
    """Direct coordinate reference using extended-precision accumulation."""
    X_long = np.asarray(X, dtype=np.longdouble)
    delta = X_long[:, None, :] - X_long[None, :, :]
    squared = np.sum(delta * delta, axis=2, dtype=np.longdouble)
    return np.sqrt(squared).astype(np.float64)


def decaying_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """A drifting, convergent path that stresses short-interval recovery."""
    rng = np.random.default_rng(20260805)
    n_steps, dimension = 199, 96
    directions = rng.standard_normal((n_steps, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    magnitudes = np.geomspace(1.0, 1.0e-7, n_steps)
    steps = directions * magnitudes[:, None]
    X = np.vstack([np.zeros((1, dimension)), np.cumsum(steps, axis=0)])
    # The production pipeline builds its Gram matrix from the stored
    # checkpoints, not from the pre-cumsum source increments.  Re-differencing
    # here reproduces that exact floating-point contract.
    return X, np.diff(X, axis=0)


def assert_matches_direct(X: np.ndarray, steps: np.ndarray, perm: np.ndarray | None) -> None:
    G = steps @ steps.T
    ordered_steps = steps if perm is None else steps[perm]
    X_ref = np.vstack([
        np.zeros((1, X.shape[1]), dtype=np.longdouble),
        np.cumsum(np.asarray(ordered_steps, dtype=np.longdouble), axis=0, dtype=np.longdouble),
    ])
    D = distance_from_step_gram(G, perm=perm)
    D_ref = direct_pairwise_distance(X_ref)
    error = np.abs(D - D_ref)
    mask = D_ref > 0.0
    max_rel = float(np.max(error[mask] / D_ref[mask]))
    if max_rel > 5e-12:
        raise AssertionError(f"all-pair reconstruction error too large: {max_rel:.3e}")

    expected_steps = step_lengths_from_gram(G if perm is None else G[np.ix_(perm, perm)])
    if not np.array_equal(np.diag(D, k=1), expected_steps):
        raise AssertionError("lag-one distances do not exactly match sqrt(diag(G))")
    if gram_reconstruction_error(G, D, perm=perm) != 0.0:
        raise AssertionError("lag-one reconstruction invariant is not exact")


def main() -> None:
    X, steps = decaying_trajectory()
    assert_matches_direct(X, steps, perm=None)
    permutation = np.random.default_rng(17).permutation(len(steps))
    assert_matches_direct(X, steps, perm=permutation)

    G = steps @ steps.T
    try:
        distance_from_step_gram(G, perm=np.arange(len(steps) - 1))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid increment permutation was accepted")

    print("Distance reconstruction test passed: direct, permuted, and invariant checks are exact.")


if __name__ == "__main__":
    main()
