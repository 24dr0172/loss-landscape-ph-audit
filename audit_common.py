#!/usr/bin/env python3
"""Shared utilities for the topological-recurrence reliability audit.

This module is the single source of truth for audit statistics, permutation
null generators, SHA-256 hashing, H1 lifetime extraction, and graph-geodesic
construction.

Formal decision rule:
    formal_trigger = (pperm < alpha)

The strict-separation flag is diagnostic only.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

try:  # optional until a geodesic routine is called
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import kneighbors_graph
except Exception:  # pragma: no cover
    csr_matrix = None
    shortest_path = None
    kneighbors_graph = None

try:  # optional until PH is called
    from ripser import ripser
except Exception:  # pragma: no cover
    ripser = None


@dataclass(frozen=True)
class AuditThresholds:
    alpha: float = 0.05
    delta_min: float = 1e-3
    lobs_mult: float = 5.0
    eps: float = 1e-12
    eps_sig: float = 1e-6
    mad_scale: float = 1.4826
    null_collapse_zero_frac: float = 0.7
    null_collapse_lobs_min: float = 0.05


def configure_single_thread_blas() -> None:
    """Pin BLAS/OpenMP thread pools to avoid joblib × BLAS oversubscription."""
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def mad(a: Iterable[float]) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.median(np.abs(a - np.median(a))))


def compute_stats(
    Lobs: float,
    Lnull: Iterable[float],
    *,
    alpha: float = 0.05,
    delta_min: float = 1e-3,
    lobs_mult: float = 5.0,
    eps: float = 1e-12,
    eps_sig: float = 1e-6,
    mad_scale: float = 1.4826,
    null_collapse_zero_frac: float = 0.7,
    null_collapse_lobs_min: float = 0.05,
) -> dict:
    """Return manuscript-consistent audit statistics.

    Formal inference uses only ``formal_trigger = pperm < alpha``.
    ``separation_flag`` is diagnostic only. ``nominal_trigger`` is retained only to read archived CSV schemas and must not be used for manuscript recurrence decisions.
    """
    L = np.asarray(Lnull, dtype=float)
    n = int(L.size)
    if n == 0:
        raise ValueError("Lnull must contain at least one null lifetime.")

    Lobs = float(Lobs)
    med = float(np.median(L))
    mx = float(np.max(L))
    pperm = (1.0 + float(np.sum(L >= Lobs))) / (n + 1)
    pct = 100.0 * float(np.sum(L < Lobs)) / n
    zrob = (Lobs - med) / max(mad_scale * mad(L), eps_sig)
    delta = Lobs - mx
    tsr = Lobs / (med + eps)
    zero_frac = float(np.sum(L <= eps)) / n

    separation_flag = bool((delta >= delta_min) and (Lobs >= lobs_mult * mx))
    formal_trigger = bool(pperm < alpha)
    # Diagnostic field only. Do not use this as a manuscript recurrence
    # decision; formal inference is pperm < alpha.
    nominal_trigger = bool(formal_trigger or separation_flag)
    null_collapsed = bool((zero_frac >= null_collapse_zero_frac) and (Lobs < null_collapse_lobs_min))
    zrob_collapsed = bool(mad(L) <= eps_sig / max(mad_scale, eps))

    return {
        "Lobs": Lobs,
        "null_med": med,
        "null_max": mx,
        "TSR": tsr,
        "pperm": pperm,
        "pct": pct,
        "zrob": zrob,
        "delta": delta,
        "zero_frac": zero_frac,
        "null_collapsed": null_collapsed,
        "zrob_collapsed": zrob_collapsed,
        "fallback": separation_flag,
        "separation_flag": separation_flag,
        "formal_trigger": formal_trigger,
        "nominal_trigger": nominal_trigger,
    }


def _as_rng(rng_or_seed: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if isinstance(rng_or_seed, np.random.Generator):
        return rng_or_seed
    if rng_or_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(rng_or_seed))


def matched_step_null(
    X: np.ndarray,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Single matched-step permutation null preserving start and step multiset."""
    X = np.asarray(X)
    if len(X) < 2:
        return X.copy()
    rg = _as_rng(rng)
    steps = np.diff(X, axis=0)
    perm = rg.permutation(len(steps))
    start = X[0:1]
    return np.vstack([start, start + np.cumsum(steps[perm], axis=0)])


def matched_step_nulls(
    X: np.ndarray,
    n_nulls: int,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> list[np.ndarray]:
    """Matched-step permutation nulls preserving start and increment multiset."""
    rg = _as_rng(rng)
    return [matched_step_null(X, rg) for _ in range(int(n_nulls))]


def block_permutation_indices(n_steps: int, block_size: int, rng: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
    """Permutation of step indices by contiguous blocks; b=1 is matched-step."""
    rg = _as_rng(rng)
    n_steps = int(n_steps)
    if n_steps <= 0:
        return np.empty(0, dtype=int)
    b = max(1, int(block_size))
    blocks = [np.arange(i, min(i + b, n_steps), dtype=int) for i in range(0, n_steps, b)]
    order = rg.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])

def block_null(
    X: np.ndarray,
    block_size: int,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Single block-permutation diagnostic null.

    block_size=1 recovers the matched-step null. block_size>1 is a diagnostic
    dependence-preserving null and should not be promoted to formal evidence.
    """
    X = np.asarray(X)
    if len(X) < 2:
        return X.copy()
    rg = _as_rng(rng)
    steps = np.diff(X, axis=0)
    start = X[0:1]
    perm = block_permutation_indices(len(steps), block_size, rg)
    return np.vstack([start, start + np.cumsum(steps[perm], axis=0)])


def block_step_nulls(
    X: np.ndarray,
    n_nulls: int,
    block_size: int,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> list[np.ndarray]:
    """Block-permutation diagnostic nulls."""
    rg = _as_rng(rng)
    return [block_null(X, block_size, rg) for _ in range(int(n_nulls))]


def stride_subsample(X: np.ndarray, cap: int) -> np.ndarray:
    X = np.asarray(X)
    if len(X) <= int(cap):
        return X
    idx = np.linspace(0, len(X) - 1, int(cap), dtype=int)
    return X[idx]


def safe_lifetime(
    dgms,
    *,
    censoring_threshold: Optional[float] = None,
) -> float:
    """Maximum H1 lifetime, including right-censored lower bounds.

    When Ripser is run with a finite ``thresh``, a class still alive at that
    cutoff has death ``inf``. Such a bar is right-censored, not absent. If
    ``censoring_threshold`` is supplied, it contributes the rigorous lower
    bound ``censoring_threshold - birth`` instead of being discarded.
    """
    if dgms is None or len(dgms) < 2 or len(dgms[1]) == 0:
        return 0.0
    H1 = np.asarray(dgms[1], dtype=float)
    if H1.ndim != 2 or H1.shape[1] < 2:
        return 0.0

    finite_mask = np.isfinite(H1[:, 1])
    lifetimes = []
    if np.any(finite_mask):
        lifetimes.append(H1[finite_mask, 1] - H1[finite_mask, 0])

    censored_mask = ~finite_mask
    if np.any(censored_mask):
        if censoring_threshold is None or not np.isfinite(censoring_threshold):
            raise ValueError(
                "H1 diagram contains right-censored bars. Pass the finite "
                "Ripser threshold as censoring_threshold, or compute the "
                "complete filtration."
            )
        lifetimes.append(
            np.maximum(float(censoring_threshold) - H1[censored_mask, 0], 0.0)
        )

    if not lifetimes:
        return 0.0
    return float(np.max(np.concatenate(lifetimes)))


def sha256_file(path: Union[str, Path], block_size: int = 2**20, chunk_size: Optional[int] = None) -> str:
    """SHA-256 digest for release manifests."""
    p = Path(path)
    size = int(chunk_size if chunk_size is not None else block_size)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(size), b""):
            h.update(chunk)
    return h.hexdigest()


def symmetrized_knn_graph(X: np.ndarray, k: int):
    """Undirected union kNN graph for point clouds.

    Release convention:
        A = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False)
        G = A.maximum(A.T)
    """
    if kneighbors_graph is None:
        raise ImportError("scikit-learn is required for symmetrized_knn_graph")
    X = np.asarray(X)
    n = len(X)
    if n < 2:
        raise ValueError("Need at least two points for a kNN graph.")
    kk = min(max(1, int(k)), n - 1)
    A = kneighbors_graph(X, n_neighbors=kk, mode="distance", include_self=False)
    return A.maximum(A.T)


def symmetrized_knn_graph_from_distance_matrix(D_full: np.ndarray, k: int):
    """Undirected union kNN graph from a precomputed ambient distance matrix."""
    if csr_matrix is None:
        raise ImportError("scipy is required for precomputed-distance kNN graphs")
    D_full = np.asarray(D_full, dtype=float)
    if D_full.ndim != 2 or D_full.shape[0] != D_full.shape[1]:
        raise ValueError("D_full must be a square distance matrix.")
    n = D_full.shape[0]
    if n < 2:
        raise ValueError("Need at least two points for a kNN graph.")
    kk = min(max(1, int(k)), n - 1)
    rows, cols, data = [], [], []
    for i in range(n):
        # Exclude self explicitly, including in the rare case where diagonal is not exactly zero.
        order = np.argsort(D_full[i], kind="mergesort")
        nbrs = [int(j) for j in order if int(j) != i][:kk]
        rows.extend([i] * len(nbrs))
        cols.extend(nbrs)
        data.extend([float(D_full[i, j]) for j in nbrs])
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)


def geodesic_distance_matrix(
    X: np.ndarray,
    *,
    k0: int = 12,
    kmax: int = 50,
    sentinel_fill: bool = False,
    input_distance_matrix: bool = False,
    rescale: bool = False,
    eps: float = 1e-12,
    context_label: str = "",
) -> tuple[np.ndarray, int, bool]:
    """Build a connected symmetrized-kNN graph and return shortest-path distances.

    For point clouds this always uses:
        A = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False)
        G = A.maximum(A.T)
        D = shortest_path(G, directed=False)

    For memory-constrained neural scripts that already computed an ambient
    distance matrix, set ``input_distance_matrix=True``. The same undirected
    kNN-union convention is used after explicitly excluding self-neighbors.

    Default behavior is ``sentinel_fill=False``: a disconnected
    graph at ``kmax`` raises RuntimeError instead of silently replacing
    infinite distances. Set ``sentinel_fill=True`` only for explicitly labelled
    diagnostic/synthetic runs whose caveat is reported in the manuscript/logs.
    """
    if shortest_path is None:
        raise ImportError("scipy is required for geodesic_distance_matrix")
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 4:
        raise ValueError("Need at least four points for H1 geodesic analysis.")
    hard_max = min(int(kmax), n - 1)
    k = min(max(1, int(k0)), hard_max)
    while True:
        G = (symmetrized_knn_graph_from_distance_matrix(X, k)
             if input_distance_matrix else symmetrized_knn_graph(X, k))
        D = shortest_path(G, directed=False)
        if not np.isinf(D).any():
            used_sentinel = False
            break
        if k >= hard_max:
            finite = D[np.isfinite(D)]
            if not sentinel_fill or finite.size == 0:
                msg = f"kNN graph disconnected at kmax={hard_max}"
                if context_label:
                    msg += f" ({context_label})"
                raise RuntimeError(msg)
            fill = float(np.median(finite) * 10.0 + 1.0)
            warnings.warn(
                "Geodesic kNN graph disconnected "
                f"at k={hard_max}; inf entries filled with {fill:.3e}. "
                "Downstream H1 may be distorted."
                + (f" Context: {context_label}" if context_label else ""),
                RuntimeWarning,
                stacklevel=2,
            )
            D[~np.isfinite(D)] = fill
            used_sentinel = True
            break
        k = min(2 * k, hard_max)

    if rescale:
        positive = D[D > eps]
        if positive.size:
            med = float(np.median(positive))
            if med > 0:
                D = D / (med + eps)
    return D, k, used_sentinel


def h1_lifetime_from_distance_matrix(
    D: np.ndarray,
    *,
    pct: float = 95.0,
    eps: float = 1e-12,
    rescale: bool = False,
    complete_filtration: bool = False,
) -> float:
    """Compute the maximum H1 lifetime from a distance matrix.

    For formal analyses with small distance matrices, set
    ``complete_filtration=True``. The filtration then runs to the largest
    finite pairwise distance, so H1 classes are not right-censored by an
    arbitrary percentile cutoff. When ``complete_filtration=False``, bars
    surviving the percentile cutoff contribute rigorous lower bounds rather
    than being discarded.
    """
    if ripser is None:
        raise ImportError("ripser is required for persistent homology")
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix.")
    if len(D) < 4:
        return 0.0
    D_work = D.copy() if rescale else D
    if rescale:
        positive = D_work[(D_work > eps) & np.isfinite(D_work)]
        if not positive.size:
            return 0.0
        med = float(np.median(positive))
        if med > 0:
            D_work = D_work / (med + eps)
    positive = D_work[(D_work > eps) & np.isfinite(D_work)]
    if not positive.size:
        return 0.0
    if complete_filtration:
        thresh = float(np.max(positive))
    else:
        if not (0.0 < float(pct) <= 100.0):
            raise ValueError("pct must lie in (0, 100].")
        thresh = float(np.percentile(positive, pct))
    dgms = ripser(D_work, maxdim=1, distance_matrix=True, thresh=thresh)["dgms"]
    return safe_lifetime(dgms, censoring_threshold=thresh)


def geodesic_h1_lifetime(
    X: np.ndarray,
    *,
    k0: int = 12,
    kmax: int = 50,
    pct: float = 95.0,
    eps: float = 1e-12,
    sentinel_fill: bool = False,
    input_distance_matrix: bool = False,
    return_k: bool = False,
    return_sentinel: bool = False,
    complete_filtration: bool = False,
    context_label: str = "",
):
    """Convenience wrapper: shared geodesic construction + H1 lifetime."""
    D, k_used, sentinel_used = geodesic_distance_matrix(
        X,
        k0=k0,
        kmax=kmax,
        sentinel_fill=sentinel_fill,
        input_distance_matrix=input_distance_matrix,
        rescale=True,
        eps=eps,
        context_label=context_label,
    )
    L = h1_lifetime_from_distance_matrix(
        D,
        pct=pct,
        eps=eps,
        rescale=False,
        complete_filtration=complete_filtration,
    )
    if return_k and return_sentinel:
        return L, k_used, sentinel_used
    if return_k:
        return L, k_used
    if return_sentinel:
        return L, sentinel_used
    return L
