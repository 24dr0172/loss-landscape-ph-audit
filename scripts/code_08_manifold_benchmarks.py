#!/usr/bin/env python3
"""Static manifold benchmark stress test.

Purpose
-------
Evaluate graph-geodesic persistent homology on static point-cloud benchmarks:
circle, Swiss roll, torus, and figure-eight.

Null model
----------
These are unordered point clouds, so the matched-step trajectory null is not
applicable. The null ensemble samples per-coordinate Gaussians matched to the
empirical mean and standard deviation of each dataset.

Decision convention
-------------------
The reported tail fraction is descriptive for this fitted Gaussian static-cloud null. The
separation flag, TSR, zrob, delta, and null-collapse diagnostics are descriptive
only.

Outputs
-------
Writes manifold_results.csv under results/manifold_benchmarks/.
"""

from __future__ import annotations

# Shared audit primitives.
from pathlib import Path as _AuditPath
import sys as _audit_sys

_audit_sys.path.insert(0, str(_AuditPath(__file__).resolve().parents[1]))

from audit_common import (
    compute_stats,
    geodesic_distance_matrix,
    safe_lifetime,
)

from pathlib import Path
import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ripser import ripser


# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = Path("results") / "manifold_benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42

N_SAMPLES = 1200
SUBSAMPLE = 600
N_NULLS = 300

GEODESIC_K = 12
GEODESIC_MAX_K = 50
GEODESIC_PCT = 95.0

ALPHA = 0.05
DELTA_MIN = 1e-3
LOBS_MULT = 5.0

EPS = 1e-12
EPS_SIG = 1e-6
MAD_SCALE = 1.4826

N_JOBS = -1

# Significant-bar threshold:
# Count bars with lifetime > BAR_THRESH * max_lifetime.
BAR_THRESH = 0.10

# This suite is intentionally a stress test, not a release gate. The known
# Swiss-roll false positive is retained as a reported limitation. Set
# STRICT_MANIFOLD_BENCHMARK=1 only when you explicitly want any mismatch to
# terminate the script with a non-zero exit status.
STRICT_MANIFOLD_BENCHMARK = os.environ.get(
    "STRICT_MANIFOLD_BENCHMARK", "0"
).strip().lower() in {"1", "true", "yes", "y", "on"}


print("=" * 70)
print("Static manifold benchmark stress test")
print("=" * 70)
print(f"N_SAMPLES={N_SAMPLES}")
print(f"SUBSAMPLE={SUBSAMPLE}")
print(f"N_NULLS={N_NULLS}")
print(f"GEODESIC_K={GEODESIC_K}")
print(f"GEODESIC_MAX_K={GEODESIC_MAX_K}")
print(f"GEODESIC_PCT={GEODESIC_PCT}")
print(f"ALPHA={ALPHA}")
print("Null model: Gaussian matched to empirical mean + per-dim std")
print(f"Bar threshold: > {BAR_THRESH * 100:.0f}% of max H1 lifetime")
print("Static Gaussian tail score: descriptive stress-test diagnostic only")
print(f"STRICT_MANIFOLD_BENCHMARK={int(STRICT_MANIFOLD_BENCHMARK)}")
print("zrob and TSR are descriptive only.")
print("=" * 70)


# =============================================================================
# PH CORE
# =============================================================================

def _random_subsample(
    X: np.ndarray,
    subsample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Random subsampling for static point clouds.

    This is correct here because the manifold samples are unordered point clouds.
    """
    if len(X) <= subsample:
        return X

    idx = rng.choice(len(X), subsample, replace=False)
    return X[idx]


def _build_geodesic(
    X: np.ndarray,
    k0: int = GEODESIC_K,
) -> tuple[np.ndarray | None, int]:
    """
    Build a median-rescaled graph-geodesic distance matrix.

    Uses the centralized audit_common implementation, which applies:

        A = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False)
        G = A.maximum(A.T)
        D = shortest_path(G, directed=False)

    Returns:
        D, k_used

    Graph construction failures are hard failures.
    Disconnected-geodesic failures are not converted into Lmax=0 rows.
    """
    if len(X) < 4:
        return None, k0

    D, k_used, sentinel_used = geodesic_distance_matrix(
        X,
        k0=k0,
        kmax=GEODESIC_MAX_K,
        sentinel_fill=False,
        rescale=True,
        eps=EPS,
        context_label="manifold_benchmark",
    )
    if sentinel_used:
        raise RuntimeError(
            "Internal error: sentinel_used=True even though sentinel_fill=False."
        )
    return D, k_used


def _ph_from_dist(D: np.ndarray) -> tuple[float, int]:
    """Run VR PH and retain lower bounds for bars censored by the cutoff."""
    if D is None or len(D) < 4:
        return 0.0, 0
    positive = D[(D > EPS) & np.isfinite(D)]
    thresh = float(np.percentile(positive, GEODESIC_PCT)) if positive.size else 1.0
    dgms = ripser(D, maxdim=1, distance_matrix=True, thresh=thresh)["dgms"]
    max_life = safe_lifetime(dgms, censoring_threshold=thresh)
    if len(dgms) < 2 or len(dgms[1]) == 0 or max_life <= 0:
        return float(max_life), 0
    H1 = np.asarray(dgms[1], dtype=float)
    deaths = np.where(np.isfinite(H1[:, 1]), H1[:, 1], thresh)
    lifetimes = np.maximum(deaths - H1[:, 0], 0.0)
    n_sig = int(np.sum(lifetimes > BAR_THRESH * max_life))
    return float(max_life), n_sig


def _geodesic_Lmax(
    X: np.ndarray,
    subsample: int,
    rng: np.random.Generator,
) -> float:
    """
    Scalar geodesic Lmax for observed/null static point clouds.

    This function is used in the null ensemble.
    """
    Xs = _random_subsample(X, subsample, rng)

    if len(Xs) < 4:
        return 0.0

    D, _ = _build_geodesic(Xs)

    if D is None:
        return 0.0

    Lmax, _ = _ph_from_dist(D)

    return Lmax


def _geodesic_full(
    X: np.ndarray,
    subsample: int,
    rng: np.random.Generator,
    k0: int = GEODESIC_K,
) -> tuple[float, int, int]:
    """
    Full observed geodesic PH for static point cloud.

    Returns:
        Lmax, n_significant_H1_bars, k_used
    """
    Xs = _random_subsample(X, subsample, rng)

    if len(Xs) < 4:
        return 0.0, 0, k0

    D, k_used = _build_geodesic(Xs, k0=k0)
    Lmax, n_sig = _ph_from_dist(D)
    return Lmax, n_sig, k_used


# =============================================================================
# NULL MODEL
# =============================================================================

def gaussian_null(
    X: np.ndarray,
    n_nulls: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Gaussian null matched to empirical mean and per-dimension standard deviation.

    Appropriate for static point clouds only.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + EPS

    nulls = []

    for _ in range(n_nulls):
        Xn = rng.standard_normal(X.shape) * std + mean
        nulls.append(Xn.astype(np.float32))

    return nulls


# =============================================================================
# MANIFOLD GENERATORS
# =============================================================================

def sample_circle(
    n: int,
    r: float = 3.0,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Circle in R^3.

    Ground truth:
        beta1 = 1
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    t = rng.uniform(0, 2.0 * np.pi, n)

    X = np.column_stack([
        r * np.cos(t),
        r * np.sin(t),
        np.zeros(n),
    ])

    X += sigma * rng.standard_normal(X.shape)

    return X.astype(np.float32)


def sample_swiss_roll(
    n: int,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Swiss roll in R^3.

    Ground truth:
        beta1 = 0
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(0, 1, n))
    height = rng.uniform(0, 10, n)

    X = np.column_stack([
        t * np.cos(t),
        height,
        t * np.sin(t),
    ])

    X = (X - X.mean(axis=0)) / (X.std() + EPS)
    X += sigma * rng.standard_normal(X.shape)

    return X.astype(np.float32)


def sample_torus(
    n: int,
    R: float = 3.0,
    r: float = 1.0,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Torus in R^3.

    Ground truth:
        beta1 = 2
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    u = rng.uniform(0, 2.0 * np.pi, n)
    v = rng.uniform(0, 2.0 * np.pi, n)

    X = np.column_stack([
        (R + r * np.cos(v)) * np.cos(u),
        (R + r * np.cos(v)) * np.sin(u),
        r * np.sin(v),
    ])

    X += sigma * rng.standard_normal(X.shape)

    return X.astype(np.float32)


def sample_figure8(
    n: int,
    r: float = 1.5,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Figure-8, modelled as a wedge of two circles in R^3.

    Ground truth:
        beta1 = 2
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    n1 = n // 2
    n2 = n - n1

    t1 = rng.uniform(0, 2.0 * np.pi, n1)
    t2 = rng.uniform(0, 2.0 * np.pi, n2)

    C1 = np.column_stack([
        -r + r * np.cos(t1),
        r * np.sin(t1),
        np.zeros(n1),
    ])

    C2 = np.column_stack([
        r + r * np.cos(t2),
        r * np.sin(t2),
        np.zeros(n2),
    ])

    X = np.vstack([C1, C2])
    X += sigma * rng.standard_normal(X.shape)

    return X.astype(np.float32)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(
    name: str,
    X: np.ndarray,
    seed: int,
    expected_beta1: int,
) -> dict:
    """
    Run full geodesic PH benchmark on one manifold dataset.
    """
    print()
    print("=" * 70)
    print(
        f"{name} | N={len(X)} | d={X.shape[1]} "
        f"| expected beta1={expected_beta1}"
    )
    print("=" * 70)

    t0 = time.time()

    # Observed statistic
    rng_obs = np.random.default_rng(seed)
    Lobs, n_sig, k_used_obs = _geodesic_full(
        X,
        SUBSAMPLE,
        rng_obs,
    )

    # Null ensemble
    rng_null = np.random.default_rng(seed + 10)
    nulls = gaussian_null(
        X,
        N_NULLS,
        rng_null,
    )

    Lnull = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(_geodesic_Lmax)(
                nulls[i],
                SUBSAMPLE,
                np.random.default_rng(seed + 100 + i),
            )
            for i in range(N_NULLS)
        ),
        dtype=float,
    )

    stats = compute_stats(
        Lobs,
        Lnull,
        alpha=ALPHA,
        delta_min=DELTA_MIN,
        lobs_mult=LOBS_MULT,
        eps=EPS,
        eps_sig=EPS_SIG,
        mad_scale=MAD_SCALE,
    )

    elapsed = time.time() - t0

    empirical_tail_trigger = bool(stats["pperm"] < ALPHA)
    decision = (
        "static_gaussian_tail_trigger"
        if empirical_tail_trigger
        else "static_gaussian_tail_non_trigger"
    )

    expected_decision = (
        "expected_h1_trigger"
        if expected_beta1 > 0
        else "expected_no_h1_trigger"
    )

    correct = (
        (expected_beta1 > 0 and empirical_tail_trigger)
        or (expected_beta1 == 0 and not empirical_tail_trigger)
    )

    bar_check = "N/A"
    if expected_beta1 == 2:
        bar_check = f"{n_sig}/2 {'PASS' if n_sig >= 2 else 'PARTIAL'}"

    print(
        f"Geodesic:"
        f" k_used={k_used_obs}"
        f" Lobs={Lobs:.4f}"
        f" null_med={stats['null_med']:.4f}"
        f" null_max={stats['null_max']:.4f}"
        f" TSR={stats['TSR']:.3f}"
        f" zrob={stats['zrob']:+.2f}"
        f" p={stats['pperm']:.4f}"
        f" delta={stats['delta']:.4f}"
        f" fallback={stats['fallback']}"
        f" empirical_tail_trigger={empirical_tail_trigger}"
    )

    print(
        f"Significant H1 bars: {n_sig}"
        f" | expected beta1={expected_beta1}"
        f" | bar_check={bar_check}"
    )

    print(
        f"Decision: {decision}"
        f" | expected={expected_decision}"
        f" | {'PASS' if correct else 'FAIL'}"
        f" | elapsed={elapsed:.1f}s"
    )

    return dict(
        dataset=name,
        expected_beta1=expected_beta1,
        n_sig_bars=n_sig,
        k_used=k_used_obs,
        bar_check=bar_check,
        decision=decision,
        expected_decision=expected_decision,
        correct=correct,
        geo_Lobs=stats["Lobs"],
        geo_null_med=stats["null_med"],
        geo_null_max=stats["null_max"],
        geo_TSR=stats["TSR"],
        # Compatibility field retained because compute_stats uses the same
        # finite-sample p-value formula. Manuscript tables should use geo_pnull
        # for this static Gaussian-null benchmark, not call it a permutation p.
        geo_pperm=stats["pperm"],
        geo_pnull=stats["pperm"],
        geo_null_type="gaussian_empirical_static",
        geo_pct=stats["pct"],
        geo_zrob=stats["zrob"],
        geo_delta=stats["delta"],
        geo_zero_frac=stats["zero_frac"],
        geo_null_collapsed=stats["null_collapsed"],
        geo_zrob_collapsed=stats["zrob_collapsed"],
        geo_fallback=stats["fallback"],
        geo_separation_flag=stats["separation_flag"],
        geo_formal_trigger=False,
        geo_empirical_tail_trigger=empirical_tail_trigger,
        geo_nominal_trigger=False,
        analysis_role="control_or_benchmark",
        is_formal_inference=False,
        N_SAMPLES=N_SAMPLES,
        SUBSAMPLE=SUBSAMPLE,
        N_NULLS=N_NULLS,
        GEODESIC_K=GEODESIC_K,
        GEODESIC_MAX_K=GEODESIC_MAX_K,
        GEODESIC_PCT=GEODESIC_PCT,
        ALPHA=ALPHA,
        DELTA_MIN=DELTA_MIN,
        LOBS_MULT=LOBS_MULT,
        BAR_THRESH=BAR_THRESH,
        elapsed_s=round(elapsed, 1),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> pd.DataFrame:
    master_rng = np.random.default_rng(RNG_SEED)

    benchmarks = [
        (
            "Circle",
            sample_circle(N_SAMPLES, rng=master_rng),
            1000,
            1,
        ),
        (
            "Swiss Roll",
            sample_swiss_roll(N_SAMPLES, rng=master_rng),
            2000,
            0,
        ),
        (
            "Torus",
            sample_torus(N_SAMPLES, rng=master_rng),
            3000,
            2,
        ),
        (
            "Figure-8",
            sample_figure8(N_SAMPLES, rng=master_rng),
            4000,
            2,
        ),
    ]

    results = []

    for name, X, seed, beta1 in benchmarks:
        row = run_benchmark(
            name=name,
            X=X,
            seed=seed,
            expected_beta1=beta1,
        )
        results.append(row)

    df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for _, row in df.iterrows():
        status = "pass" if row["correct"] else "FAIL"
        print(
            f"[{status}]"
            f" {row['dataset']:12s}"
            f" beta1={row['expected_beta1']}"
            f" bars={row['n_sig_bars']}"
            f" geo_p={row['geo_pperm']:.4f}"
            f" TSR={row['geo_TSR']:.3f}"
            f" => {row['decision']}"
        )

    all_pass = bool(df["correct"].all())
    n_pass = int(df["correct"].sum())
    n_total = len(df)

    print()

    if all_pass:
        print(f"All {n_total}/{n_total} manifold benchmarks passed.")
    else:
        failed = df[~df["correct"]]["dataset"].tolist()
        print(f"{n_pass}/{n_total} passed. FAILED: {failed}")

    out = OUT_DIR / "manifold_results.csv"

    df.to_csv(
        out,
        index=False,
    )

    print()
    print(f"Saved: {out}")

    if not all_pass:
        failed = df[~df["correct"]][[
            "dataset", "decision", "expected_decision",
            "geo_pnull", "geo_Lobs", "geo_TSR",
        ]]
        message = (
            "Manifold stress-test mismatches were observed.\n"
            + failed.to_string(index=False)
        )
        if STRICT_MANIFOLD_BENCHMARK:
            raise RuntimeError(message)

        print("\n[WARN] " + message)
        print(
            "\nThese mismatches are retained as diagnostic results and do not "
            "stop the release reproduction. The Swiss-roll false positive "
            "is a reported limitation of this static Gaussian-null stress test."
        )

    return df


if __name__ == "__main__":
    main()
