#!/usr/bin/env python3
"""Observed path-fidelity certificate for the graph-geodesic audit.

This is a targeted, non-formal reanalysis.  It does not retrain networks, it
does not generate permutation nulls, and it does not alter the matched-step
decision rule in the main audit.

--------------------------------------------------------------------------
WHAT IS BEING COMPUTED
--------------------------------------------------------------------------
Let X = (x_0, ..., x_{n-1}) be the ordered checkpoints and
a_r = ||x_{r+1} - x_r||_2 the consecutive chord lengths.  The open-path
reference metric induced by the observed checkpoint order is

    d_star(i, j) = sum_{r=i}^{j-1} a_r,     i < j.

Let G be the realised weighted OR-symmetrised kNN graph with Euclidean edge
weights w_uv = ||x_u - x_v||_2, and d_G its shortest-path metric.  Define

    mu_G  = min over graph edges {u,v} of  w_uv / d_star(u, v),   0 < mu_G <= 1
    eta_G = max over r of  d_G(r, r+1) / a_r,                     eta_G >= 1
    kappa_G = eta_G / mu_G

Then, for all i, j,

    mu_G * d_star(i, j) <= d_G(i, j) <= eta_G * d_star(i, j),           (M)

because (lower) every graph path has weighted length at least mu_G times its
d_star-length and d_star obeys the triangle inequality, and (upper) chaining
d_G along consecutive checkpoints gives
d_G(i,j) <= sum_r d_G(r, r+1) <= eta_G * d_star(i,j).  Note that (M) does NOT
require every consecutive pair to be a graph edge; c_adj is descriptive only,
and enters solely through eta_G.

Because d_star is an OPEN path metric, (X, d_star) is isometric to a finite
subset of the real line, so VR(X, d_star; s) is the clique complex of an
interval graph and H1(VR(X, d_star; s)) = 0 at EVERY scale s.  Combining (M)
with the resulting multiplicative interleaving

    VR(d_G, r) subset VR(d_star, r/mu_G) subset VR(d_G, eta_G*r/mu_G)

the map H1(VR(d_G, r)) -> H1(VR(d_G, kappa_G * r)) factors through zero, so
every H1 interval [b, d) of VR(X, d_G) satisfies

    d / b <= kappa_G.                                                   (C)

(C) is scale-invariant, so the repository's median rescaling does not affect
it.

--------------------------------------------------------------------------
WHAT THIS SCRIPT IS AND IS NOT EVIDENCE FOR
--------------------------------------------------------------------------
(M) and (C) are theorems.  Verifying them on data checks THIS IMPLEMENTATION,
not the science.  Columns prefixed `impl_` are implementation diagnostics and
must never be reported as findings.  A violation of `impl_metric_inequality`
or `impl_bound_violation` indicates a bug in this script, not a property of
the trajectory.

The scientific output is the joint comparison

    kappa_G   versus   max(d / b)

summarised by `kappa_slack = kappa_G / max_ratio_lower_bound`, read together
with `mu_argmin_lag`.  If kappa_G is orders of magnitude above every observed
d/b for BOTH the contractible controls and the neural trajectories, the
certificate does not discriminate and the theory gate should close honestly.

The certificate says only whether an observed feature is compatible with
graph-metric distortion of an open ordered path.  It is not a formal test, it
does not interact with p_perm, and it cannot establish recurrence.

--------------------------------------------------------------------------
SCOPE LIMITATION: OPEN-PATH REFERENCE ONLY
--------------------------------------------------------------------------
The H1-triviality step requires an open path reference.  A genuinely cyclic
trajectory has a cyclic reference metric whose own Rips filtration carries
H1, and (C) does not apply.  The clean-circle positive control is therefore
DELIBERATELY EXCLUDED: it is out of scope, not a failure.  The contractible
controls provided here (helix, overpass) are open curves and are in scope.

--------------------------------------------------------------------------
PROVENANCE
--------------------------------------------------------------------------
Graph construction matches the final audit: increment-Gram distances,
OR-symmetrisation via the shared routine, stable precomputed kNN selection,
geometric doubling schedule, no sentinel filling.

The frozen neural specification uses kmax = 199 for n = 200, i.e. n - 1.
The default --kmax 0 resolves to n - 1 and therefore reproduces the frozen
value at n = 200; pass --kmax 199 explicitly for the provenance record.  This
rerun package also aligns the Experiment 5 audit with that n - 1 limit.

--------------------------------------------------------------------------
CORRECTIONS IN THIS REVISION (audit of the released run)
--------------------------------------------------------------------------
FIX A  distance_from_step_gram.  The released reconstruction recovered
       D2[i,j] = sum_{a,b in [i,j)} G[a,b] by inclusion-exclusion on a double
       cumulative sum.  Each of the four prefix terms has magnitude up to the
       sum over ALL n^2 Gram entries, while the target for a consecutive pair
       is the single entry G[r,r].  On a converging trajectory the
       cancellation destroys the smallest distances.  Measured consequence in
       the released certificate: 34 of 36 neural rows report c_adj = 1.000000
       together with eta_G > 1, up to eta_G - 1 = 5.79e-03.  That combination
       is impossible in exact arithmetic (Fact 4).  Replaced by a
       diagonal-anchored recursion that is EXACT at lag 1.  Validated against
       exact float64 coordinate differences: worst-case all-pair relative
       error 2.9e-02 -> 8.7e-16, induced eta_G - 1 -> exactly 0.

       In this rerun package the production routine is centralised in
       audit_common.py and used by code_10, code_14, code_16, code_17,
       code_19, code_20, and this certificate.

FIX B  Reconstruction consistency anchor.  diag(D_ambient, 1) must reproduce
       sqrt(diag(G)) exactly.  Exported as impl_recon_rel_err and enforced.

FIX C  eta_G consistency.  Wherever {r, r+1} is a graph edge,
       d_G(r, r+1) = a_r exactly, so eta_G = 1 exactly when c_adj = 1.
       The exported eps_incon_raw is an internal invariant residual and must
       be zero to numerical tolerance.  It is not an external numerical-error
       bound and is not used to make a persistence-resolution claim.

FIX D  Control provenance.  Controls were generated directly at n points with
       a shared RNG stream and were therefore NOT the clouds the manuscript
       reports.  They are now generated at N = 1200 with the code_01 stream
       order and stride-subsampled to 600, matching code_01 exactly.

FIX E  Location-robust audit_common import (the old parents[1] assumed a
       scripts/ location while the file ships at the repository root).

NEW    Theorem E scale-localised ceiling kappa_G(b) = eta_G(b/mu_G(b))/mu_G(b)
       is now computed per bar, together with the localisation gain
       kappa_G / kappa_G(b) and the pre-registered necessary condition
       required_mu_for_bind = b/d.  Since eta_G(s) >= 1 and mu_G(b) <= 1, no
       ceiling of this family can bind unless mu_G(b) lies in [b/d, 1].

Usage (run from the repository root, where audit_common.py lives):

    python code_21_path_fidelity_certificate.py \
        --traj-dir exp4_results/trajectories \
        --kmax 199 \
        --with-controls \
        --out-dir exp4_results/appendix/path_fidelity
"""
from __future__ import annotations

# Pin BLAS/OpenMP before importing NumPy/SciPy.
import os as _os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_key, "1")

import argparse
import gc
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

# FIX A (location-robust import).  The previous `parents[1]` assumed this file
# lived in scripts/, but it ships at the repository root, where parents[1] is
# the repository's PARENT directory.  It worked only because Python puts the
# script's own directory on sys.path first.  Resolve audit_common explicitly.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE if (_HERE / "audit_common.py").exists() else _HERE.parent
if not (_REPO_ROOT / "audit_common.py").exists():
    raise SystemExit(
        f"audit_common.py not found next to {_HERE} or its parent; "
        "run this script from inside the release tree."
    )
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audit_common import (  # noqa: E402
    distance_from_step_gram as shared_distance_from_step_gram,
    gram_reconstruction_error,
    sha256_file,
    step_lengths_from_gram as shared_step_lengths_from_gram,
    symmetrized_knn_graph_from_distance_matrix,
)


ARCH_ORDER = {
    "mlp2_mnist": 0,
    "smallcnn_cifar10": 1,
    "resnet18_cifar10": 2,
}
OPT_ORDER = {
    "sgd": 0,
    "sgd_momentum": 1,
    "adam": 2,
    "adamw": 3,
}

EPS = 1e-12
BOUND_TOL = 1e-9

# Graph modes.  `or_knn` is the audited construction.  `complete_euclidean`
# is the ambient-Euclidean comparison in which every pair is an edge; it is
# the shortcut-pathology arm of the controls and is never applied to the
# neural cells.
MODE_KNN = "or_knn"
MODE_EUCLID = "complete_euclidean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=Path("exp4_results") / "trajectories",
        help="Directory containing full-space trajectory .npy files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exp4_results") / "appendix" / "path_fidelity",
        help="Directory for the CSV outputs and grouped summary.",
    )
    parser.add_argument(
        "--main-results",
        type=Path,
        default=None,
        help="Optional current main topology CSV to merge for annotation.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*__*__seed*.npy",
        help="Glob pattern for trajectory arrays.",
    )
    parser.add_argument("--k0", type=int, default=12, help="Initial k.")
    parser.add_argument(
        "--kmax",
        type=int,
        default=0,
        help=(
            "Maximum k; 0 resolves to n_points - 1.  The frozen neural spec "
            "is 199 at n = 200.  Do not use 50."
        ),
    )
    parser.add_argument("--chunk-cols", type=int, default=50_000)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=36,
        help="Expected number of main neural trajectories; 0 disables warning.",
    )
    parser.add_argument(
        "--shortcut-ratio",
        type=float,
        default=0.5,
        help=(
            "Ratio below which a graph edge is counted as a STRONG shortcut. "
            "Every non-consecutive chord is strictly below 1.0 by the "
            "triangle inequality, so 1.0 would be uninformative."
        ),
    )
    parser.add_argument(
        "--with-controls",
        action="store_true",
        help=(
            "Also run the open-path contractible controls (helix, overpass) "
            "under both the OR-kNN graph and the ambient-Euclidean complete "
            "graph.  These calibrate what kappa_G looks like when the answer "
            "is known."
        ),
    )
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="Run only the controls; skip the neural trajectory directory.",
    )
    parser.add_argument(
        "--control-n",
        type=int,
        default=600,
        help="Ordered stride subsample size for the controls.",
    )
    parser.add_argument("--control-seed", type=int, default=42)
    parser.add_argument(
        "--hash-trajectories",
        action="store_true",
        help="Also compute SHA-256 digests; this rereads every large .npy.",
    )
    return parser.parse_args()


# ===========================================================================
# TRAJECTORY DISCOVERY
# ===========================================================================
def parse_trajectory_name(path: Path) -> tuple[str, str, int] | None:
    if path.name.endswith("__metrics.npy"):
        return None
    match = re.match(
        r"^(?P<arch>.+?)__(?P<optimizer>.+?)__seed(?P<seed>\d+)\.npy$",
        path.name,
    )
    if match is None:
        return None
    return match.group("arch"), match.group("optimizer"), int(match.group("seed"))


def iter_trajectory_files(
    traj_dir: Path, pattern: str
) -> list[tuple[str, str, int, Path]]:
    rows: list[tuple[str, str, int, Path]] = []
    for path in sorted(traj_dir.glob(pattern)):
        parsed = parse_trajectory_name(path)
        if parsed is None:
            continue
        arch, optimizer, seed = parsed
        rows.append((arch, optimizer, seed, path))
    rows.sort(
        key=lambda row: (
            ARCH_ORDER.get(row[0], 99),
            OPT_ORDER.get(row[1], 99),
            row[2],
        )
    )
    return rows


# ===========================================================================
# DISTANCES
# ===========================================================================
def step_gram_chunked(path: Path, chunk_cols: int) -> np.ndarray:
    """Accumulate the float64 Gram matrix of checkpoint increments."""
    if chunk_cols <= 0:
        raise ValueError("chunk_cols must be positive")
    X = np.load(path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {X.shape} for {path}")
    n, d = map(int, X.shape)
    if n < 4:
        raise ValueError(f"Need at least four checkpoints, got n={n} for {path}")
    gram = np.zeros((n - 1, n - 1), dtype=np.float64)
    for start in range(0, d, chunk_cols):
        end = min(start + chunk_cols, d)
        X_chunk = np.asarray(X[:, start:end], dtype=np.float64)
        steps = np.diff(X_chunk, axis=0)
        gram += steps @ steps.T
        del X_chunk, steps
    gram = 0.5 * (gram + gram.T)
    del X
    return gram


def step_gram_from_array(X: np.ndarray) -> np.ndarray:
    steps = np.diff(np.asarray(X, dtype=np.float64), axis=0)
    gram = steps @ steps.T
    return 0.5 * (gram + gram.T)


def _legacy_distance_from_step_gram_prefix(gram: np.ndarray) -> np.ndarray:
    """DEPRECATED reconstruction retained only for provenance comparison.

    This private comparison helper reproduces the pre-rerun reconstruction.
    It recovers

        D2[i, j] = sum_{a=i}^{j-1} sum_{b=i}^{j-1} G[a, b]

    by inclusion-exclusion on a DOUBLE cumulative sum.  The four prefix terms
    each have magnitude up to sum over ALL n^2 Gram entries, while the target
    value for a consecutive pair is the single entry G[r, r].  On a converging
    trajectory whose late steps are orders of magnitude smaller than the net
    displacement, the cancellation destroys the small distances.

    Measured consequence in the released certificate: 34 of 36 neural rows
    report c_adj = 1.000000 (every consecutive pair is a graph edge) together
    with eta_G > 1, up to eta_G - 1 = 5.79e-03.  That combination is
    impossible in exact arithmetic: if {r, r+1} is an edge then
    d_G(r, r+1) = a_r exactly, so eta_G = 1 exactly.  It is not called by
    the rerun package.
    """
    G = np.asarray(gram, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("The step Gram matrix must be square.")
    prefix = np.pad(
        np.cumsum(np.cumsum(G, axis=0), axis=1),
        ((1, 0), (1, 0)),
    )
    diagonal = np.diag(prefix)
    D2 = diagonal[:, None] + diagonal[None, :] - prefix - prefix.T
    D2 = 0.5 * (D2 + D2.T)
    D = np.sqrt(np.maximum(D2, 0.0))
    np.fill_diagonal(D, 0.0)
    return D


def _legacy_anchored_distance_from_step_gram(gram: np.ndarray) -> np.ndarray:
    """Recover all pairwise checkpoint chord distances from step Gram data.

    FIX A (numerical).  Diagonal-anchored recursion replacing the legacy
    inclusion-exclusion form.  Writing s_a for the increments and
    G[a, b] = <s_a, s_b>, the identity

        ||x_j+1 - x_i||^2 = ||x_j - x_i||^2
                            + 2 * sum_{a=i}^{j-1} G[a, j]
                            + G[j, j]

    is accumulated outward from the diagonal.  Every partial sum is taken from
    the anchor i, so no large quantities are subtracted.  Consequences:

      * lag 1 is EXACT: D[r, r+1] = sqrt(G[r, r]) = a_r bit-for-bit, so
        eta_G = 1 exactly whenever every consecutive pair is a graph edge;
      * short lags, which set the kNN edge weights and therefore mu_G, are
        accurate to machine precision instead of ~1e-2 relative;
      * the np.maximum(D2, 0) clip can no longer manufacture a zero-length
        consecutive step.

    Validated against exact float64 coordinate differences on drifting
    trajectories with geometrically decaying step size: worst-case all-pair
    relative error falls from 2.9e-02 (legacy) to 8.7e-16, and the induced
    eta_G - 1 falls from 2.9e-02 to exactly 0.

    Cost is O(n^3) with n = 200 checkpoints, i.e. negligible next to the
    chunked Gram accumulation over the full parameter dimension.
    """
    G = np.asarray(gram, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("The step Gram matrix must be square.")
    m = int(G.shape[0])
    n = m + 1
    diag_G = np.diag(G)
    D2 = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        block = G[i:, i:]
        # cross[t] = sum_{a=i}^{i+t} G[a, i+t+1]
        csum = np.cumsum(block, axis=0)
        t = np.arange(block.shape[0] - 1)
        cross = csum[t, t + 1]
        incr = np.empty(block.shape[0], dtype=np.float64)
        incr[0] = diag_G[i]                       # lag 1: exactly ||s_i||^2
        incr[1:] = 2.0 * cross + diag_G[i + 1:]
        D2[i, i + 1:] = np.cumsum(incr)
    D2 = D2 + D2.T
    np.fill_diagonal(D2, 0.0)
    D = np.sqrt(np.maximum(D2, 0.0))
    np.fill_diagonal(D, 0.0)
    return D


def _legacy_step_lengths_from_gram(gram: np.ndarray) -> np.ndarray:
    """FIX 1: consecutive chord lengths taken directly from diag(G).

    a_r = ||x_{r+1} - x_r|| = sqrt(G_rr) exactly.

    NOTE ON FIX 1 VS FIX A.  Fix 1 alone made a_r accurate while the graph
    edge weights still came from the legacy prefix-sum reconstruction.  The
    two quantities then disagreed, which is precisely what produced
    eta_G > 1 with c_adj = 1 in the released certificate.  Fix A repairs the
    reconstruction, so a_r and D[r, r+1] now agree bit-for-bit and this
    function is a consistency anchor rather than a divergent second opinion.
    """
    return np.sqrt(np.maximum(np.diag(np.asarray(gram, dtype=np.float64)), 0.0))


def path_metric_from_step_lengths(step_lengths: np.ndarray) -> np.ndarray:
    """Finite OPEN path reference metric induced by checkpoint order."""
    a = np.asarray(step_lengths, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("step_lengths must be one-dimensional.")
    coordinate = np.concatenate(
        [np.array([0.0], dtype=np.float64), np.cumsum(a, dtype=np.float64)]
    )
    return np.abs(coordinate[:, None] - coordinate[None, :])


# ===========================================================================
# GRAPHS
# ===========================================================================
def edge_arrays(graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coo = graph.tocoo()
    mask = coo.row < coo.col
    return (
        np.asarray(coo.row[mask], dtype=np.int64),
        np.asarray(coo.col[mask], dtype=np.int64),
        np.asarray(coo.data[mask], dtype=np.float64),
    )


def complete_euclidean_graph(D_ambient: np.ndarray) -> csr_matrix:
    """Complete graph with Euclidean weights (ambient-Euclidean arm)."""
    D = np.asarray(D_ambient, dtype=np.float64).copy()
    np.fill_diagonal(D, 0.0)
    return csr_matrix(D)


def adaptive_graph(
    D_ambient: np.ndarray,
    *,
    k0: int,
    kmax: int,
    mode: str,
) -> tuple[object, np.ndarray, dict]:
    """Build the audited OR-kNN graph (doubling k) or the complete graph."""
    D = np.asarray(D_ambient, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D_ambient must be square.")
    n = int(D.shape[0])

    if mode == MODE_EUCLID:
        graph = complete_euclidean_graph(D)
        D_geo = shortest_path(graph, directed=False)
        return (
            graph,
            D_geo,
            {
                "graph_mode": MODE_EUCLID,
                "k_initial": n - 1,
                "k_used": n - 1,
                "kmax": n - 1,
                "initial_components": 1,
                "final_components": 1,
                "connected_at_initial_k": True,
                "connected_final": True,
                "adaptive_growth_used": False,
                "disconnected_at_kmax": False,
            },
        )

    hard_max = min(n - 1, int(kmax) if int(kmax) > 0 else n - 1)
    k_initial = min(max(1, int(k0)), hard_max)

    graph = symmetrized_knn_graph_from_distance_matrix(D, k_initial)
    n_components, _ = connected_components(graph, directed=False, return_labels=True)
    initial_components = int(n_components)
    k = k_initial
    while n_components != 1 and k < hard_max:
        k = min(2 * k, hard_max)
        graph = symmetrized_knn_graph_from_distance_matrix(D, k)
        n_components, _ = connected_components(
            graph, directed=False, return_labels=True
        )

    D_geo = shortest_path(graph, directed=False)
    finite = bool(np.isfinite(D_geo).all())
    connected = bool(int(n_components) == 1 and finite)
    metadata = {
        "graph_mode": MODE_KNN,
        "k_initial": int(k_initial),
        "k_used": int(k),
        "kmax": int(hard_max),
        "initial_components": initial_components,
        "final_components": int(n_components),
        "connected_at_initial_k": bool(initial_components == 1),
        "connected_final": connected,
        "adaptive_growth_used": bool(k > k_initial),
        # FIX 5: the canonical geodesic_distance_matrix raises RuntimeError
        # when the graph is still disconnected at kmax (sentinel_fill=False).
        # Here we continue and emit an explicitly non-assessable row instead
        # of aborting the whole sweep, but the condition is flagged.
        "disconnected_at_kmax": bool(not connected),
    }
    return graph, D_geo, metadata


# ===========================================================================
# CERTIFICATE
# ===========================================================================
def safe_ratio_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict.fromkeys(("min", "p05", "median", "p95", "max"), np.nan)
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compute_certificate(
    gram: np.ndarray,
    *,
    k0: int,
    kmax: int,
    mode: str,
    shortcut_ratio: float,
) -> tuple[dict, object, np.ndarray, np.ndarray]:
    """Compute mu_G, eta_G, kappa_G and the implementation checks."""
    step_lengths = shared_step_lengths_from_gram(gram)
    D_ambient = shared_distance_from_step_gram(gram)
    n = int(D_ambient.shape[0])
    D_star = path_metric_from_step_lengths(step_lengths)

    # FIX B (reconstruction consistency anchor).  The consecutive entries of
    # the reconstructed ambient matrix must reproduce diag(G) exactly.  Any
    # drift here propagates straight into the edge weights, mu_G and eta_G,
    # so it is measured, exported, and hard-failed rather than trusted.
    impl_recon_rel_err = gram_reconstruction_error(gram, D_ambient)
    impl_reconstruction_exact = bool(impl_recon_rel_err <= BOUND_TOL)

    graph, D_geo, graph_meta = adaptive_graph(
        D_ambient, k0=k0, kmax=kmax, mode=mode
    )
    rows, cols, weights = edge_arrays(graph)
    edge_ref = D_star[rows, cols] if len(rows) else np.array([], dtype=float)

    zero_step_count = int(np.count_nonzero(step_lengths <= EPS))
    all_steps_positive = bool(zero_step_count == 0)
    edge_ref_positive = bool(len(edge_ref) == 0 or np.all(edge_ref > EPS))
    finite_graph_metric = bool(np.isfinite(D_geo).all())

    # c_adj is descriptive.  (M) does not require c_adj = 1; consecutive-edge
    # absence is charged to eta_G through the graph detour.
    edge_set = set(zip(rows.tolist(), cols.tolist()))
    adjacent_present = np.array(
        [(r, r + 1) in edge_set for r in range(n - 1)], dtype=bool
    )
    c_adj = float(np.mean(adjacent_present)) if n > 1 else np.nan

    # ---- mu_G -------------------------------------------------------------
    edge_ratios = np.array([], dtype=np.float64)
    mu_raw = np.nan
    mu_argmin_i = -1
    mu_argmin_j = -1
    mu_argmin_lag = -1
    mu_argmin_weight = np.nan
    mu_argmin_ref = np.nan
    if len(weights) and edge_ref_positive:
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_ratios = weights / edge_ref
        finite_mask = np.isfinite(raw_ratios)
        edge_ratios = raw_ratios[finite_mask]
        if edge_ratios.size:
            local = int(np.argmin(raw_ratios[finite_mask]))
            idx = int(np.flatnonzero(finite_mask)[local])
            mu_raw = float(raw_ratios[idx])
            # FIX 3: identify the edge that sets mu_G.  This is the single
            # most diagnostic column: a large lag means one long-range
            # near-self-approach, a small lag means generic checkpoint-scale
            # meander, and the two have completely different remedies.
            mu_argmin_i = int(rows[idx])
            mu_argmin_j = int(cols[idx])
            mu_argmin_lag = int(abs(mu_argmin_j - mu_argmin_i))
            mu_argmin_weight = float(weights[idx])
            mu_argmin_ref = float(edge_ref[idx])

    mu_G = float(np.clip(mu_raw, 0.0, 1.0)) if np.isfinite(mu_raw) else np.nan
    # A chord cannot exceed the ordered path length between its endpoints, so
    # mu_raw > 1 by more than roundoff means the reference or the graph is
    # inconsistent.  Surface it instead of hiding it in the clip.
    impl_mu_in_range = bool(
        np.isfinite(mu_raw) and mu_raw <= 1.0 + BOUND_TOL and mu_raw > 0.0
    )

    theta_G = float(1.0 - mu_G) if np.isfinite(mu_G) else np.nan

    # ---- eta_G ------------------------------------------------------------
    eta_G = np.nan
    eta_argmax_index = -1
    eps_incon = np.nan
    impl_eta_consistency_ok = False
    if finite_graph_metric and all_steps_positive:
        adjacent_geodesic = np.diag(D_geo, k=1)
        eta_values = adjacent_geodesic / step_lengths
        if np.isfinite(eta_values).all() and eta_values.size == n - 1:
            eta_G = float(np.max(eta_values))
            eta_argmax_index = int(np.argmax(eta_values))

        # Where a consecutive edge is present, its graph distance must equal
        # the direct Gram-diagonal step length.  This is an internal invariant
        # check only; it is not converted into an external error bound.
        if np.any(adjacent_present):
            eps_incon = float(
                np.max(
                    np.abs(
                        adjacent_geodesic[adjacent_present]
                        - step_lengths[adjacent_present]
                    )
                )
            )
        else:
            eps_incon = 0.0
        # eta_G == 1 exactly is REQUIRED when every consecutive pair is an edge.
        impl_eta_consistency_ok = bool(
            (int(np.sum(~adjacent_present)) > 0)
            or (np.isfinite(eta_G) and abs(eta_G - 1.0) <= BOUND_TOL)
        )

    # ---- assessability ----------------------------------------------------
    # FIX 5: every route to a non-assessable certificate is named, rather
    # than collapsing to a silent NaN -> inf.
    reasons: list[str] = []
    if not finite_graph_metric:
        reasons.append("disconnected_graph_infinite_geodesic")
    if not all_steps_positive:
        reasons.append(f"zero_length_reference_step_x{zero_step_count}")
    if not edge_ref_positive:
        reasons.append("zero_length_reference_for_graph_edge")
    if not np.isfinite(mu_G) or mu_G <= EPS:
        reasons.append("mu_G_undefined_or_zero")
    if not np.isfinite(eta_G):
        reasons.append("eta_G_undefined")
    if not impl_mu_in_range:
        reasons.append("mu_raw_out_of_range")
    if not impl_reconstruction_exact:
        reasons.append("ambient_reconstruction_inconsistent_with_diag_gram")
    if not impl_eta_consistency_ok:
        reasons.append("eta_G_gt_1_with_all_consecutive_edges_present")
    assessable = bool(not reasons)

    kappa_G = float(eta_G / mu_G) if assessable else np.inf

    # ---- implementation check of (M) --------------------------------------
    upper = np.triu_indices(n, k=1)
    pair_ref = D_star[upper]
    pair_geo = D_geo[upper]
    pair_valid = (pair_ref > EPS) & np.isfinite(pair_geo)
    pair_ratios = pair_geo[pair_valid] / pair_ref[pair_valid]
    pair_summary = safe_ratio_summary(pair_ratios)

    if assessable and pair_ratios.size:
        lower_ok = bool(np.min(pair_ratios) + BOUND_TOL >= mu_G)
        upper_ok = bool(np.max(pair_ratios) <= eta_G + BOUND_TOL)
    else:
        lower_ok = False
        upper_ok = False

    strong_shortcut_count = (
        int(np.sum(edge_ratios < float(shortcut_ratio))) if edge_ratios.size else 0
    )

    row = {
        **graph_meta,
        "n_points": n,
        "edge_count": int(len(weights)),
        "possible_edges": int(n * (n - 1) // 2),
        "graph_density": float(len(weights) / max(n * (n - 1) / 2.0, 1.0)),
        "zero_step_count": zero_step_count,
        "all_steps_positive": all_steps_positive,
        "edge_ref_positive": edge_ref_positive,
        "finite_graph_metric": finite_graph_metric,
        "adjacent_edge_count": int(np.sum(adjacent_present)),
        "missing_adjacent_edge_count": int(np.sum(~adjacent_present)),
        "c_adj": c_adj,
        # ---- certificate quantities ----
        "mu_raw": mu_raw,
        "mu_G": mu_G,
        "theta_G": theta_G,
        "eta_G": eta_G,
        "kappa_G": kappa_G,
        "mu_argmin_i": mu_argmin_i,
        "mu_argmin_j": mu_argmin_j,
        "mu_argmin_lag": mu_argmin_lag,
        "mu_argmin_weight": mu_argmin_weight,
        "mu_argmin_ref_length": mu_argmin_ref,
        "eta_argmax_index": eta_argmax_index,
        # ---- edge-ratio distribution (FIX 2: no fallback to pair ratios) ---
        "edge_ratio_min": float(np.min(edge_ratios)) if edge_ratios.size else np.nan,
        "edge_ratio_p05": (
            float(np.percentile(edge_ratios, 5)) if edge_ratios.size else np.nan
        ),
        "edge_ratio_median": (
            float(np.median(edge_ratios)) if edge_ratios.size else np.nan
        ),
        # FIX 4: every non-consecutive chord is strictly shorter than the
        # ordered path between its endpoints, so a "< 1.0" count would flag
        # essentially the whole edge set.  This counts STRONG shortcuts only.
        "strong_shortcut_ratio_threshold": float(shortcut_ratio),
        "strong_shortcut_edge_count": strong_shortcut_count,
        # Heuristic robust variant.  NOT a bound: replacing the minimum by a
        # low quantile breaks the proof of (M).  Reported only to show how
        # much of kappa_G is driven by a single edge.
        "kappa_p05_heuristic_not_a_bound": (
            float(eta_G / np.percentile(edge_ratios, 5))
            if (edge_ratios.size and np.isfinite(eta_G)
                and np.percentile(edge_ratios, 5) > EPS)
            else np.nan
        ),
        # ---- pair-ratio distribution ----
        "pair_ratio_min": pair_summary["min"],
        "pair_ratio_p05": pair_summary["p05"],
        "pair_ratio_median": pair_summary["median"],
        "pair_ratio_p95": pair_summary["p95"],
        "pair_ratio_max": pair_summary["max"],
        # ---- status ----
        "assessable": assessable,
        "not_assessable_reason": ";".join(reasons),
        "open_path_reference": True,
        # FIX 8: implementation diagnostics, explicitly namespaced.  These
        # verify theorems and MUST hold; they are not findings.
        "impl_lower_inequality_ok": lower_ok,
        "impl_upper_inequality_ok": upper_ok,
        "impl_metric_inequality_ok": bool(lower_ok and upper_ok),
        "impl_mu_in_range": impl_mu_in_range,
        # FIX B / FIX C diagnostics.  These MUST hold; a False here invalidates
        # mu_G, eta_G and kappa_G for the row and is not a property of data.
        "impl_reconstruction_exact": impl_reconstruction_exact,
        "impl_recon_rel_err": impl_recon_rel_err,
        "impl_eta_consistency_ok": impl_eta_consistency_ok,
        "eps_incon_raw": eps_incon,
    }
    # Context for the Theorem E scale-localised ceiling.  Rescaling to the PH
    # median happens later, in observed_h1_bars, where the factor is known.
    locality_ctx = {
        "edge_w_raw": np.asarray(weights, dtype=np.float64),
        "edge_ratio": np.asarray(
            edge_ratios if edge_ratios.size else np.array([]), dtype=np.float64
        ),
        "edge_ratio_all": (
            np.asarray(weights, dtype=np.float64) / np.asarray(edge_ref, dtype=np.float64)
            if len(weights) and edge_ref_positive
            else np.array([], dtype=np.float64)
        ),
        "step_raw": np.asarray(step_lengths, dtype=np.float64),
        "eta_ratio": (
            np.diag(D_geo, k=1) / step_lengths
            if (finite_graph_metric and all_steps_positive)
            else np.full(n - 1, np.nan)
        ),
    }
    return row, graph, D_geo, D_star, locality_ctx


# ===========================================================================
# THEOREM E — SCALE-LOCALISED CEILING
# ===========================================================================
def mu_local(edge_w_rescaled: np.ndarray, edge_ratio: np.ndarray, r: float) -> float:
    """mu_G(r) = min over graph edges with w_e <= r of w_e / d_star(e)."""
    mask = np.isfinite(edge_w_rescaled) & np.isfinite(edge_ratio) & (edge_w_rescaled <= r)
    if not np.any(mask):
        return np.nan
    return float(np.min(edge_ratio[mask]))


def eta_local(step_rescaled: np.ndarray, eta_ratio: np.ndarray, s: float) -> float:
    """eta_G(s) = max over consecutive pairs with a_q <= s of d_G(q,q+1)/a_q."""
    mask = np.isfinite(step_rescaled) & np.isfinite(eta_ratio) & (step_rescaled <= s)
    if not np.any(mask):
        return 1.0
    return float(np.max(eta_ratio[mask]))


def kappa_local(ctx: dict, b: float) -> tuple[float, float]:
    """Theorem E ceiling kappa_G(b) = eta_G(b / mu_G(b)) / mu_G(b).

    Returns (kappa_G(b), mu_G(b)).  All inputs are in the same median-rescaled
    units as the birth scale b, because the thresholds in mu_G(r) and
    eta_G(s) are NOT scale invariant even though kappa_G itself is.
    """
    if not ctx or not np.isfinite(b) or b <= 0.0:
        return np.nan, np.nan
    mu_b = mu_local(ctx["edge_w_rescaled"], ctx["edge_ratio"], b)
    if not np.isfinite(mu_b) or mu_b <= EPS:
        return np.nan, mu_b
    eta_s = eta_local(ctx["step_rescaled"], ctx["eta_ratio"], b / mu_b)
    return float(eta_s / mu_b), float(mu_b)


# ===========================================================================
# PERSISTENCE
# ===========================================================================
def rescale_for_complete_ph(D_geo: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Repository median rescaling, then a complete-filtration threshold."""
    D = np.asarray(D_geo, dtype=np.float64)
    positive = D[(D > EPS) & np.isfinite(D)]
    if positive.size == 0:
        return D.copy(), np.nan, 0.0
    median = float(np.median(positive))
    scale = median + EPS
    D_scaled = D / scale
    positive_scaled = D_scaled[(D_scaled > EPS) & np.isfinite(D_scaled)]
    threshold = float(np.max(positive_scaled)) if positive_scaled.size else 0.0
    return D_scaled, scale, threshold


def observed_h1_bars(
    D_geo: np.ndarray,
    kappa_G: float,
    assessable: bool,
    locality_ctx: dict | None = None,
) -> tuple[list[dict], dict]:
    """Complete observed H1 barcode with d/b, separating censored bars.

    FIX 6.  A right-censored bar has true death >= threshold.  Therefore:

      * threshold / b > kappa_G  ->  the bound is CONFIRMED VIOLATED, because
        the true ratio is at least this large;
      * threshold / b <= kappa_G ->  INCONCLUSIVE.  The true death may exceed
        the threshold, so the bar is NOT proven to satisfy the bound.

    Only uncensored bars can confirm compliance.  Under a complete filtration
    the threshold equals the largest pairwise distance, at which VR is a full
    simplex, so censoring should be rare and confined to the boundary; if it
    is common, that itself is worth reporting.
    """
    empty = {
        "ph_scale_median": np.nan,
        "ph_threshold": np.nan,
        "kappa_local_at_max_ratio_bar": np.nan,
        "localisation_gain": np.nan,
        "h1_lifetime_max": np.nan,
        "h1_bar_count": 0,
        "h1_censored_count": 0,
        "h1_within_confirmed_count": 0,
        "h1_censored_inconclusive_count": 0,
        "h1_exceeds_confirmed_count": 0,
        "max_ratio_uncensored": np.nan,
        "max_ratio_lower_bound": np.nan,
        "kappa_slack": np.nan,
        "all_bars_confirmed_within": False,
        "impl_bound_violation": False,
        "h1_ph_status": "skipped",
    }

    if not np.isfinite(D_geo).all():
        out = dict(empty)
        out["h1_ph_status"] = "skipped_disconnected_graph"
        return [], out

    D_ph, ph_scale, threshold = rescale_for_complete_ph(D_geo)

    ctx_rescaled: dict = {}
    if locality_ctx and np.isfinite(ph_scale) and ph_scale > 0:
        ctx_rescaled = {
            "edge_w_rescaled": locality_ctx["edge_w_raw"] / ph_scale,
            "edge_ratio": locality_ctx["edge_ratio_all"],
            "step_rescaled": locality_ctx["step_raw"] / ph_scale,
            "eta_ratio": locality_ctx["eta_ratio"],
        }
        if ctx_rescaled["edge_ratio"].size != ctx_rescaled["edge_w_rescaled"].size:
            ctx_rescaled = {}

    if threshold <= 0.0:
        out = dict(empty)
        out.update(
            ph_scale_median=ph_scale,
            ph_threshold=threshold,
            h1_lifetime_max=0.0,
            all_bars_confirmed_within=True,
            h1_ph_status="empty_metric",
        )
        return [], out

    try:
        from ripser import ripser
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "code_21 requires ripser. Install the supplied environment before "
            "running the Phase 1 certificate."
        ) from exc
    diagrams = ripser(D_ph, maxdim=1, distance_matrix=True, thresh=threshold)["dgms"]
    H1 = (
        np.asarray(diagrams[1], dtype=np.float64)
        if len(diagrams) > 1
        else np.empty((0, 2))
    )

    bars: list[dict] = []
    for bar_id, (birth, death) in enumerate(H1):
        censored = bool(not np.isfinite(death))
        death_lower_bound = float(threshold if censored else death)
        lifetime = float(max(death_lower_bound - birth, 0.0))
        ratio_lb = (
            float(death_lower_bound / birth) if birth > EPS else np.inf
        )

        if not assessable or not np.isfinite(kappa_G):
            status = "not_assessable"
        elif ratio_lb > kappa_G + BOUND_TOL:
            status = "exceeds_bound_confirmed"
        elif censored:
            status = "censored_inconclusive"
        else:
            status = "within_bound_confirmed"

        # Theorem E, evaluated at this bar's own birth scale.
        kappa_b, mu_b = kappa_local(ctx_rescaled, float(birth))
        # Necessary condition for ANY Theorem D/E ratio ceiling to bind:
        # eta_G(s) >= 1 gives kappa_G(b) >= 1 / mu_G(b), and Fact 2 gives
        # mu_G(b) <= 1, so binding requires mu_G(b) in [b/d, 1].  Recording
        # the requirement makes the outcome falsifiable in advance.
        required_mu_for_bind = (
            float(1.0 / ratio_lb) if np.isfinite(ratio_lb) and ratio_lb > 0 else np.nan
        )

        bars.append(
            {
                "bar_id": int(bar_id),
                "birth": float(birth),
                "death_reported": float(death) if np.isfinite(death) else np.inf,
                "death_lower_bound": death_lower_bound,
                "lifetime_lower_bound": lifetime,
                "death_birth_ratio_lower_bound": ratio_lb,
                "right_censored": censored,
                "kappa_G": kappa_G,
                "bound_status": status,
                # ---- Theorem E ----
                "mu_local_at_birth": mu_b,
                "kappa_local_at_birth": kappa_b,
                "localisation_gain": (
                    float(kappa_G / kappa_b)
                    if (np.isfinite(kappa_G) and np.isfinite(kappa_b) and kappa_b > EPS)
                    else np.nan
                ),
                # A finite bar is genuinely binding only when its ratio reaches
                # the local ceiling to numerical tolerance.  A strict value
                # below the ceiling is a theorem violation, not stronger
                # evidence for the ceiling.
                "kappa_local_binds": bool(
                    (not censored)
                    and np.isfinite(kappa_b)
                    and np.isfinite(ratio_lb)
                    and abs(kappa_b - ratio_lb) <= BOUND_TOL
                ),
                "required_mu_for_bind": required_mu_for_bind,
            }
        )

    statuses = [bar["bound_status"] for bar in bars]
    ratios_lb = np.asarray(
        [bar["death_birth_ratio_lower_bound"] for bar in bars], dtype=np.float64
    )
    ratios_unc = np.asarray(
        [
            bar["death_birth_ratio_lower_bound"]
            for bar in bars
            if not bar["right_censored"]
        ],
        dtype=np.float64,
    )
    max_ratio_lb = float(np.max(ratios_lb)) if ratios_lb.size else np.nan
    max_ratio_unc = float(np.max(ratios_unc)) if ratios_unc.size else np.nan

    # The scientific number: how far above the observed ratios the ceiling
    # sits.  ~1 means the certificate binds; >>1 means it is vacuous.
    kappa_slack = (
        float(kappa_G / max_ratio_lb)
        if (np.isfinite(kappa_G) and np.isfinite(max_ratio_lb) and max_ratio_lb > EPS)
        else np.nan
    )

    max_ratio_bar = (
        max(bars, key=lambda bar: bar["death_birth_ratio_lower_bound"])
        if bars else None
    )
    summary = {
        "ph_scale_median": ph_scale,
        "ph_threshold": threshold,
        "kappa_local_at_max_ratio_bar": (
            max_ratio_bar["kappa_local_at_birth"] if max_ratio_bar else np.nan
        ),
        "localisation_gain": (
            max_ratio_bar["localisation_gain"] if max_ratio_bar else np.nan
        ),
        "h1_lifetime_max": float(
            max((bar["lifetime_lower_bound"] for bar in bars), default=0.0)
        ),
        "h1_bar_count": int(len(bars)),
        "h1_censored_count": int(sum(bar["right_censored"] for bar in bars)),
        "h1_within_confirmed_count": int(statuses.count("within_bound_confirmed")),
        "h1_censored_inconclusive_count": int(statuses.count("censored_inconclusive")),
        "h1_exceeds_confirmed_count": int(statuses.count("exceeds_bound_confirmed")),
        "max_ratio_uncensored": max_ratio_unc,
        "max_ratio_lower_bound": max_ratio_lb,
        "kappa_slack": kappa_slack,
        "all_bars_confirmed_within": bool(
            bars and all(s == "within_bound_confirmed" for s in statuses)
        )
        or (not bars),
        # An UNCENSORED bar above kappa_G contradicts a theorem, so it is an
        # implementation error, never a property of the trajectory.
        "impl_bound_violation": bool(
            any(
                bar["bound_status"] == "exceeds_bound_confirmed"
                and not bar["right_censored"]
                for bar in bars
            )
        ),
        "h1_ph_status": "computed",
    }
    return bars, summary


# ===========================================================================
# CONTROLS
# ===========================================================================
def generate_helix(rng: np.random.Generator, n: int) -> np.ndarray:
    """3D helix: contractible open curve, metric-shortcut pathology."""
    t = np.linspace(0.0, 1.0, n)
    theta = 6.0 * np.pi * t
    X = np.vstack([3.0 * np.cos(theta), 3.0 * np.sin(theta), 12.0 * t]).T
    return X + 0.03 * rng.standard_normal(X.shape)


def generate_overpass(rng: np.random.Generator, n: int, d: int = 50) -> np.ndarray:
    """50D alpha-curve overpass: contractible open curve, projection artifact."""
    t = np.linspace(-2.2, 2.2, n)
    X = np.zeros((n, d))
    X[:, 0] = 500.0 * (t**3 - 3.0 * t)
    X[:, 1] = 500.0 * (t**2 - 1.0)
    t_cross = np.sqrt(3.0)
    bump = 1000.0 * np.exp(-200.0 * (t + t_cross) ** 2) - 1000.0 * np.exp(
        -200.0 * (t - t_cross) ** 2
    )
    for i in range(2, d):
        X[:, i] = bump
    return X + 0.5 * rng.standard_normal(X.shape)


CONTROL_N_SAMPLES = 1200   # code_01 N_SAMPLES
CONTROL_SUBSAMPLE = 600    # code_01 SUBSAMPLE


def stride_subsample(X: np.ndarray, cap: int) -> np.ndarray:
    """Ordered stride subsample, identical to audit_common.stride_subsample."""
    X = np.asarray(X)
    if len(X) <= int(cap):
        return X
    idx = np.linspace(0, len(X) - 1, int(cap), dtype=int)
    return X[idx]


def control_specs(n: int, seed: int) -> list[tuple[str, str, int, np.ndarray]]:
    """Open-path contractible controls.

    The clean-circle control is deliberately absent: a cyclic reference
    metric carries its own H1 and the interleaving argument behind (C) does
    not apply.  Excluding it is a scope statement, not an omission.

    FIX D (control provenance).  The previous version generated each control
    DIRECTLY at n points with a shared RNG stream, so the resulting clouds
    were not the ones the manuscript reports.  The canonical synthetic
    validation (code_01) generates at N_SAMPLES = 1200 with
    default_rng(RNG_SEED) and then takes an ordered stride subsample to
    SUBSAMPLE = 600, drawing the helix noise first and the overpass noise
    afterwards from the same stream.  That order and those sizes are
    reproduced here so kappa_G is computed on the manuscript's controls.

    `n` is the SUBSAMPLE size; the generation size is fixed at
    CONTROL_N_SAMPLES to preserve the RNG stream.
    """
    rng = np.random.default_rng(seed)
    helix = generate_helix(rng, CONTROL_N_SAMPLES)
    overpass = generate_overpass(rng, CONTROL_N_SAMPLES)
    return [
        (
            "control_helix",
            "contractible_open_curve",
            seed,
            stride_subsample(helix, n),
        ),
        (
            "control_overpass",
            "contractible_open_curve",
            seed,
            stride_subsample(overpass, n),
        ),
    ]


# ===========================================================================
# I/O
# ===========================================================================
def load_main_results(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"arch", "optimizer", "seed"}
    if not required.issubset(df.columns):
        print(f"[warn] Cannot merge {path}: missing {sorted(required - set(df.columns))}")
        return None
    df = df.copy()
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    optional = [
        "geo_Lobs",
        "geo_pperm",
        "geo_formal_trigger",
        "geo_k_used_obs",
        "decision",
        "kinematic_R",
        "val_acc",
    ]
    keep = ["arch", "optimizer", "seed"] + [c for c in optional if c in df]
    return df[keep].drop_duplicates(["arch", "optimizer", "seed"])


def grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["group", "arch", "optimizer", "graph_mode"], dropna=False)
        .agg(
            n_paths=("seed", "count"),
            assessable_count=("assessable", "sum"),
            connected_count=("connected_final", "sum"),
            c_adj_median=("c_adj", "median"),
            mu_median=("mu_G", "median"),
            mu_argmin_lag_median=("mu_argmin_lag", "median"),
            eta_median=("eta_G", "median"),
            kappa_median=("kappa_G", "median"),
            kappa_min=("kappa_G", "min"),
            kappa_max=("kappa_G", "max"),
            max_ratio_median=("max_ratio_lower_bound", "median"),
            kappa_slack_median=("kappa_slack", "median"),
            kappa_slack_min=("kappa_slack", "min"),
            kappa_local_median=("kappa_local_at_max_ratio_bar", "median"),
            localisation_gain_median=("localisation_gain", "median"),
            recon_rel_err_max=("impl_recon_rel_err", "max"),
            eta_consistency_ok_count=("impl_eta_consistency_ok", "sum"),
            h1_bar_count_total=("h1_bar_count", "sum"),
            exceeds_confirmed_total=("h1_exceeds_confirmed_count", "sum"),
            inconclusive_total=("h1_censored_inconclusive_count", "sum"),
        )
        .reset_index()
    )


CERT_COLS = [
    "group", "arch", "optimizer", "seed", "graph_mode",
    "n_points", "d_full",
    "k_initial", "k_used", "kmax",
    "initial_components", "final_components",
    "connected_at_initial_k", "connected_final", "adaptive_growth_used",
    "disconnected_at_kmax",
    "edge_count", "graph_density",
    "zero_step_count", "all_steps_positive", "edge_ref_positive",
    "finite_graph_metric",
    "adjacent_edge_count", "missing_adjacent_edge_count", "c_adj",
    # certificate
    "mu_raw", "mu_G", "theta_G", "eta_G", "kappa_G",
    "mu_argmin_i", "mu_argmin_j", "mu_argmin_lag",
    "mu_argmin_weight", "mu_argmin_ref_length", "eta_argmax_index",
    "edge_ratio_min", "edge_ratio_p05", "edge_ratio_median",
    "strong_shortcut_ratio_threshold", "strong_shortcut_edge_count",
    "kappa_p05_heuristic_not_a_bound",
    "pair_ratio_min", "pair_ratio_p05", "pair_ratio_median",
    "pair_ratio_p95", "pair_ratio_max",
    # persistence and the headline comparison
    "h1_bar_count", "h1_lifetime_max",
    "max_ratio_uncensored", "max_ratio_lower_bound", "kappa_slack",
    # Theorem E scale-localised ceiling
    "kappa_local_at_max_ratio_bar", "localisation_gain",
    "h1_within_confirmed_count", "h1_censored_inconclusive_count",
    "h1_exceeds_confirmed_count", "h1_censored_count",
    "all_bars_confirmed_within",
    "ph_scale_median", "ph_threshold", "h1_ph_status",
    # status
    "assessable", "not_assessable_reason", "open_path_reference",
    # implementation diagnostics only
    "impl_lower_inequality_ok", "impl_upper_inequality_ok",
    "impl_metric_inequality_ok", "impl_mu_in_range", "impl_bound_violation",
    "impl_reconstruction_exact", "impl_recon_rel_err",
    "impl_eta_consistency_ok", "eps_incon_raw",
    # provenance
    "geo_Lobs", "geo_pperm", "geo_formal_trigger", "decision",
    "trajectory_file", "trajectory_sha256", "elapsed_sec",
]


def main() -> None:
    args = parse_args()
    start_time = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []

    if not args.controls_only:
        trajectories = iter_trajectory_files(args.traj_dir, args.pattern)
        if not trajectories:
            raise SystemExit(
                f"No trajectory .npy files found in {args.traj_dir} "
                f"with pattern {args.pattern}."
            )
        if args.expected_count and len(trajectories) != args.expected_count:
            print(
                f"[warn] Expected {args.expected_count} trajectories, "
                f"found {len(trajectories)}. Continuing."
            )
        for arch, optimizer, seed, path in trajectories:
            jobs.append(
                {
                    "group": "neural",
                    "arch": arch,
                    "optimizer": optimizer,
                    "seed": seed,
                    "path": path,
                    "array": None,
                    "modes": [MODE_KNN],
                }
            )

    if args.with_controls or args.controls_only:
        for name, family, seed, X in control_specs(args.control_n, args.control_seed):
            jobs.append(
                {
                    "group": "control",
                    "arch": name,
                    "optimizer": family,
                    "seed": seed,
                    "path": None,
                    "array": X,
                    # Both arms: the audited OR-kNN graph, and the ambient
                    # Euclidean complete graph in which the shortcut is not
                    # suppressed.  The contrast is the calibration.
                    "modes": [MODE_KNN, MODE_EUCLID],
                }
            )

    if not jobs:
        raise SystemExit("Nothing to do: no trajectories and no controls selected.")

    print("=" * 78)
    print("Observed path-fidelity certificate")
    print("=" * 78)
    print(f"Trajectory directory : {args.traj_dir if not args.controls_only else '(skipped)'}")
    print(f"Cells                : {sum(len(j['modes']) for j in jobs)}")
    print(f"k0 / kmax            : {args.k0} / {args.kmax or 'n_points-1'}")
    print("Distance reconstruction: diagonal-anchored (FIX A); lag 1 is exact")
    print(f"Strong-shortcut cut  : edge ratio < {args.shortcut_ratio}")
    print("Reference            : OPEN path metric (cyclic controls out of scope)")
    print("No retraining, null permutations, or formal decisions are performed.")
    print("kappa_G >= max(d/b) is a THEOREM; confirming it checks this code,")
    print("not the trajectories.  Read kappa_slack and mu_argmin_lag instead.")
    print("=" * 78)

    certificate_rows: list[dict] = []
    bar_rows: list[dict] = []
    total_cells = sum(len(job["modes"]) for job in jobs)
    cell_index = 0

    for job in jobs:
        if job["array"] is not None:
            X = np.asarray(job["array"], dtype=np.float64)
            gram = step_gram_from_array(X)
            n_points, d_full = map(int, X.shape)
            del X
        else:
            shape = np.load(job["path"], mmap_mode="r").shape
            n_points, d_full = map(int, shape)
            gram = step_gram_chunked(job["path"], args.chunk_cols)

        for mode in job["modes"]:
            cell_start = time.time()
            cell_index += 1
            label = f"{job['arch']}/{job['optimizer']}/seed{job['seed']}"
            print(
                f"[{cell_index:02d}/{total_cells:02d}] {label} "
                f"mode={mode} shape=({n_points}, {d_full})"
            )

            row, _graph, D_geo, D_star, locality_ctx = compute_certificate(
                gram,
                k0=args.k0,
                kmax=(args.kmax if args.kmax > 0 else n_points - 1),
                mode=mode,
                shortcut_ratio=args.shortcut_ratio,
            )
            bars, h1_summary = observed_h1_bars(
                D_geo,
                row["kappa_G"],
                row["assessable"],
                locality_ctx=locality_ctx,
            )

            row.update(
                {
                    "group": job["group"],
                    "arch": job["arch"],
                    "optimizer": job["optimizer"],
                    "seed": job["seed"],
                    "d_full": d_full,
                    "trajectory_file": str(job["path"]) if job["path"] else "",
                    "trajectory_sha256": (
                        sha256_file(job["path"])
                        if (args.hash_trajectories and job["path"])
                        else ""
                    ),
                    **h1_summary,
                    "elapsed_sec": time.time() - cell_start,
                }
            )
            certificate_rows.append(row)

            for bar in bars:
                bar_rows.append(
                    {
                        "group": job["group"],
                        "arch": job["arch"],
                        "optimizer": job["optimizer"],
                        "seed": job["seed"],
                        "graph_mode": mode,
                        **bar,
                    }
                )

            if row["assessable"]:
                print(
                    f"    k={row['k_used']} c_adj={row['c_adj']:.3f} "
                    f"mu={row['mu_G']:.6g} (lag {row['mu_argmin_lag']}) "
                    f"eta={row['eta_G']:.6g} kappa={row['kappa_G']:.6g} "
                    f"H1={row['h1_bar_count']} "
                    f"max(d/b)={row['max_ratio_lower_bound']:.6g} "
                    f"slack={row['kappa_slack']:.4g} "
                    f"({row['elapsed_sec']:.1f}s)"
                )
            else:
                print(
                    f"    k={row['k_used']} NOT ASSESSABLE "
                    f"[{row['not_assessable_reason']}] "
                    f"({row['elapsed_sec']:.1f}s)"
                )
            if row["impl_bound_violation"] or not row["impl_metric_inequality_ok"]:
                if row["assessable"]:
                    print(
                        "    [IMPLEMENTATION ERROR] a proved inequality failed; "
                        "this is a bug in this script, not a finding."
                    )

            del D_geo, D_star
            gc.collect()

        del gram
        gc.collect()

    certificate = pd.DataFrame(certificate_rows)
    bars_df = pd.DataFrame(bar_rows)

    main_results = load_main_results(args.main_results)
    if main_results is not None:
        certificate = certificate.merge(
            main_results,
            on=["arch", "optimizer", "seed"],
            how="left",
            suffixes=("", "_main"),
        )

    cert_cols = [c for c in CERT_COLS if c in certificate.columns]
    certificate = certificate[
        cert_cols + [c for c in certificate.columns if c not in cert_cols]
    ]

    out_certificate = args.out_dir / "path_fidelity_certificate_main.csv"
    out_bars = args.out_dir / "path_fidelity_h1_bars_main.csv"
    out_summary = args.out_dir / "path_fidelity_summary_by_group.csv"
    certificate.to_csv(out_certificate, index=False)
    bars_df.to_csv(out_bars, index=False)
    grouped_summary(certificate).to_csv(out_summary, index=False)

    assessable = certificate[certificate["assessable"]]

    print("=" * 78)
    print(f"Wrote certificate rows : {out_certificate}")
    print(f"Wrote H1 bar rows      : {out_bars} ({len(bars_df)} bars)")
    print(f"Wrote grouped summary  : {out_summary}")
    print("-" * 78)
    print("IMPLEMENTATION DIAGNOSTICS (must pass; not scientific findings)")
    print(
        f"  assessable cells        : {int(certificate['assessable'].sum())}"
        f"/{len(certificate)}"
    )
    if len(assessable):
        print(
            f"  metric inequality ok    : "
            f"{int(assessable['impl_metric_inequality_ok'].sum())}/{len(assessable)}"
        )
        print(
            f"  bound violations (bugs) : "
            f"{int(assessable['impl_bound_violation'].sum())}"
        )
    print("-" * 78)
    print("SCIENTIFIC OUTPUT: kappa_G versus max(d/b)")
    for group in sorted(assessable["group"].unique()) if len(assessable) else []:
        for mode in sorted(assessable[assessable["group"] == group]["graph_mode"].unique()):
            sub = assessable[
                (assessable["group"] == group) & (assessable["graph_mode"] == mode)
            ]
            slack = sub["kappa_slack"].replace([np.inf, -np.inf], np.nan).dropna()
            print(
                f"  {group:8s} {mode:19s} n={len(sub):3d}  "
                f"kappa_G med={sub['kappa_G'].median():10.4g}  "
                f"max(d/b) med={sub['max_ratio_lower_bound'].median():8.4g}  "
                f"slack med={slack.median() if len(slack) else float('nan'):8.4g}  "
                f"mu lag med={sub['mu_argmin_lag'].median():6.1f}"
            )
    print("-" * 78)
    print(
        "Read: slack near 1 means the certificate binds.  Slack orders of\n"
        "magnitude above 1 for BOTH controls and neural cells means it is\n"
        "vacuous and the theory gate should close honestly."
    )
    print(f"Elapsed                : {(time.time() - start_time) / 60.0:.1f} min")
    print("=" * 78)


if __name__ == "__main__":
    main()
