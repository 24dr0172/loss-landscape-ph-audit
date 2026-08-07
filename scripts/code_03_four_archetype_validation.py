#!/usr/bin/env python3
"""Four-archetype trajectory validation.

Purpose
-------
Run the synthetic sanity validation on pure drift, curved drift, true circle,
and helix trajectories.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Implementation convention
-------------------------
Geodesic persistent homology uses stride subsampling for ordered synthetic
trajectories, positive-distance median rescaling, percentile filtration, and
sentinel_fill=False for formal geodesic evaluations.

Outputs
-------
Writes results_corrected.csv under results/four_archetype_validation/.
"""
# Shared audit primitives.
from pathlib import Path as _AuditPath
import sys as _audit_sys
_audit_sys.path.insert(0, str(_AuditPath(__file__).resolve().parents[1]))
from audit_common import (
    compute_stats,
    safe_lifetime,
    matched_step_nulls as null_matched_steps,
    geodesic_h1_lifetime,
)
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from ripser import ripser
# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
OUT_DIR = Path("results") / "four_archetype_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 1234
N_SAMPLES = 1200
SUBSAMPLE = 600
N_NULLS = 300
TOTAL_PROJECTIONS = 100
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
print("Four-archetype trajectory validation")
print(f"  N_SAMPLES={N_SAMPLES}")
print(f"  SUBSAMPLE={SUBSAMPLE}")
print(f"  N_NULLS={N_NULLS}")
print(f"  PROJECTIONS={TOTAL_PROJECTIONS}")
print(f"  Geodesic k={GEODESIC_K}, pct={GEODESIC_PCT}")
print(f"  Formal decision: pperm < {ALPHA}; separation_flag is diagnostic only")
print("  zrob and TSR are descriptive only")
# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────
def _mad(arr: np.ndarray) -> float:
    """Median absolute deviation."""
    arr = np.asarray(arr, dtype=float)
    return float(np.median(np.abs(arr - np.median(arr))))
# ─────────────────────────────────────────────────────────────
# PH CORE
# ─────────────────────────────────────────────────────────────
def _adaptive_thresh(X, pct=95.0):
    """Euclidean percentile filtration threshold."""
    d = pdist(X)
    return float(np.percentile(d, pct)) if len(d) > 0 else None
def _random_subsample(X, cap, rng):
    """Random subsample, used for Euclidean/projection diagnostics."""
    if len(X) <= cap:
        return X
    idx = rng.choice(len(X), cap, replace=False)
    return X[idx]
def _stride_subsample(X, cap):
    """
    Stride subsample, used for geodesic PH on ordered synthetic trajectories.
    This preserves the temporal/arc ordering of 1D trajectory archetypes.
    Random subsampling can create large gaps along a helix and allow kNN
    shortcuts across the coil.
    """
    if len(X) <= cap:
        return X
    idx = np.linspace(0, len(X) - 1, cap, dtype=int)
    return X[idx]
def max_H1_euclidean(X, subsample, rng):
    """
    Euclidean persistent homology.
    Used for:
        - ambient Euclidean diagnostic
        - projection diagnostic
    This is not the geodesic formal test.
    """
    X = _random_subsample(X, subsample, rng)
    thresh = _adaptive_thresh(X, pct=95.0)
    kwargs = {"maxdim": 1}
    if thresh is not None:
        kwargs["thresh"] = thresh
    dgms = ripser(X, **kwargs)["dgms"]
    return safe_lifetime(dgms, censoring_threshold=thresh)
def max_H1_geodesic(X, subsample, rng=None):
    """Shared graph-geodesic PH via audit_common.geodesic_distance_matrix."""
    X = _stride_subsample(X, subsample)
    if len(X) < 4:
        return 0.0
    return geodesic_h1_lifetime(
        X,
        k0=GEODESIC_K,
        kmax=GEODESIC_MAX_K,
        pct=GEODESIC_PCT,
        eps=EPS,
        sentinel_fill=False,
        context_label="code_03_four_archetype_validation",
    )
# ─────────────────────────────────────────────────────────────
# NULL MODEL
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# AMBIENT TESTS
# ─────────────────────────────────────────────────────────────
def ambient_tests(X, seed):
    """
    Runs ambient Euclidean PH and ambient graph-geodesic PH.
    Both tests use the same matched-step null trajectories, but PH
    randomness uses independent deterministic seeds.
    """
    nulls = null_matched_steps(
        X,
        N_NULLS,
        np.random.default_rng(seed + 100),
    )
    # Ambient Euclidean diagnostic
    Lobs_euc = max_H1_euclidean(
        X,
        SUBSAMPLE,
        np.random.default_rng(seed + 10),
    )
    H_null_euc = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(max_H1_euclidean)(
                nulls[i],
                SUBSAMPLE,
                np.random.default_rng(seed + 200 + i),
            )
            for i in range(N_NULLS)
        )
    )
    stats_euc = compute_stats(Lobs_euc, H_null_euc)
    # Ambient geodesic formal test
    Lobs_geo = max_H1_geodesic(
        X,
        SUBSAMPLE,
        np.random.default_rng(seed + 300),
    )
    H_null_geo = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(max_H1_geodesic)(
                nulls[i],
                SUBSAMPLE,
                np.random.default_rng(seed + 400 + i),
            )
            for i in range(N_NULLS)
        )
    )
    stats_geo = compute_stats(Lobs_geo, H_null_geo)
    return stats_euc, stats_geo
# ─────────────────────────────────────────────────────────────
# PROJECTION SELECTION TEST
# ─────────────────────────────────────────────────────────────
def projection_test(X, seed):
    """
    Selection-corrected projection test.
    Observed statistic:
        max over TOTAL_PROJECTIONS random 2D projections.
    Null statistic:
        for each matched-step null trajectory, take the same max over
        the same projection family.
    This controls the projection look-elsewhere effect by using the
    max-statistic under the null.
    """
    proj_rng = np.random.default_rng(seed + 1000)
    null_rng = np.random.default_rng(seed + 2000)
    proj_mats = []
    for _ in range(TOTAL_PROJECTIONS):
        Q, _ = np.linalg.qr(
            proj_rng.standard_normal((X.shape[1], 2))
        )
        proj_mats.append(Q[:, :2])
    obs_vals = np.array([
        max_H1_euclidean(
            X @ P,
            SUBSAMPLE,
            np.random.default_rng(seed + 3000 + j),
        )
        for j, P in enumerate(proj_mats)
    ])
    Lobs = float(np.max(obs_vals))
    nulls = null_matched_steps(
        X,
        N_NULLS,
        null_rng,
    )
    def null_max_worker(Ntraj, i, pmats, seed_base):
        vals = []
        for j, P in enumerate(pmats):
            val = max_H1_euclidean(
                Ntraj @ P,
                SUBSAMPLE,
                np.random.default_rng(seed_base + i * 1000 + j),
            )
            vals.append(val)
        return float(np.max(vals))
    H_null = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(null_max_worker)(
                nulls[i],
                i,
                proj_mats,
                seed + 4000,
            )
            for i in range(N_NULLS)
        )
    )
    return compute_stats(Lobs, H_null)
# ─────────────────────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────────────────────
def generate_pure_drift(rng):
    """
    Linear drift.
    Ground truth:
        contractible negative control.
    """
    t = np.linspace(0, 1, N_SAMPLES)
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    X = np.outer(t * 5.0, direction)
    X += 0.03 * rng.standard_normal(X.shape)
    return X
def generate_curved_drift(rng):
    """
    Quadratically curved drift.
    Ground truth:
        contractible negative control.
    """
    t = np.linspace(0, 1, N_SAMPLES)
    X = np.zeros((N_SAMPLES, 3))
    X[:, 0] = 3.0 * t**2
    X[:, 1] = 2.0 * t
    X += 0.03 * rng.standard_normal(X.shape)
    return X
def generate_circle(rng):
    """
    True circle.
    Ground truth:
        genuine H1 positive control.
    """
    t = np.linspace(0, 2 * np.pi, N_SAMPLES, endpoint=False)
    X = np.vstack([
        3.0 * np.cos(t),
        3.0 * np.sin(t),
        np.zeros_like(t),
    ]).T
    X += 0.03 * rng.standard_normal(X.shape)
    return X
def generate_helix(rng):
    """
    Helix with three full rotations.
    Ground truth:
        contractible trajectory, not a genuine loop.
    This is the metric/projection hallucination target.
    The pitch uses 12*t to match the final main synthetic experiment.
    """
    t = np.linspace(0, 1, N_SAMPLES)
    theta = 6.0 * np.pi * t
    X = np.vstack([
        3.0 * np.cos(theta),
        3.0 * np.sin(theta),
        12.0 * t,
    ]).T
    X += 0.03 * rng.standard_normal(X.shape)
    return X
# ─────────────────────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────────────────────
def classify(stats_geo, stats_proj):
    """
    Complete synthetic classification rule.
    Priority:
        1. If geodesic formal test fires:
               robust_recurrence
        2. Else, if projection statistic fires:
               projection_hallucination
        3. Else:
               no_recurrence
    Projection evidence cannot override a geodesic rejection.
    """
    if stats_geo["formal_trigger"]:
        return "robust_recurrence"
    if stats_proj["formal_trigger"]:
        return "projection_hallucination"
    return "no_recurrence"
# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────
def run_all():
    scenarios = [
        ("Pure Drift", generate_pure_drift, 0),
        ("Curved Drift", generate_curved_drift, 100),
        ("True Circle", generate_circle, 200),
        ("Helix", generate_helix, 300),
    ]
    results = []
    t0 = time.time()
    for name, generator, seed_offset in scenarios:
        print()
        print("=" * 70)
        print(f"Running: {name}  seed_offset={seed_offset}")
        print("=" * 70)
        rng = np.random.default_rng(RNG_SEED + seed_offset)
        X = generator(rng)
        stats_euc, stats_geo = ambient_tests(
            X,
            RNG_SEED + seed_offset + 1000,
        )
        stats_proj = projection_test(
            X,
            RNG_SEED + seed_offset + 2000,
        )
        decision = classify(stats_geo, stats_proj)
        print(
            f"  Ambient Euclidean:"
            f" Lobs={stats_euc['Lobs']:.5f}"
            f" null_med={stats_euc['null_med']:.5f}"
            f" null_max={stats_euc['null_max']:.5f}"
            f" TSR={stats_euc['TSR']:.3f}"
            f" zrob={stats_euc['zrob']:+.2f}"
            f" p={stats_euc['pperm']:.4f}"
            f" formal={stats_euc['formal_trigger']}"
        )
        print(
            f"  Ambient Geodesic:"
            f" Lobs={stats_geo['Lobs']:.5f}"
            f" null_med={stats_geo['null_med']:.5f}"
            f" null_max={stats_geo['null_max']:.5f}"
            f" TSR={stats_geo['TSR']:.3f}"
            f" zrob={stats_geo['zrob']:+.2f}"
            f" p={stats_geo['pperm']:.4f}"
            f" formal={stats_geo['formal_trigger']}"
        )
        print(
            f"  Projection Max:"
            f" Lobs={stats_proj['Lobs']:.5f}"
            f" null_med={stats_proj['null_med']:.5f}"
            f" null_max={stats_proj['null_max']:.5f}"
            f" TSR={stats_proj['TSR']:.3f}"
            f" zrob={stats_proj['zrob']:+.2f}"
            f" p={stats_proj['pperm']:.4f}"
            f" formal={stats_proj['formal_trigger']}"
        )
        print(f"  => decision: {decision}")
        def flatten(prefix, stats):
            return {f"{prefix}_{key}": value for key, value in stats.items()}
        row = dict(
            scenario=name,
            decision=decision,
            RNG_SEED=RNG_SEED,
            N_SAMPLES=N_SAMPLES,
            SUBSAMPLE=SUBSAMPLE,
            N_NULLS=N_NULLS,
            TOTAL_PROJECTIONS=TOTAL_PROJECTIONS,
            GEODESIC_K=GEODESIC_K,
            GEODESIC_MAX_K=GEODESIC_MAX_K,
            GEODESIC_PCT=GEODESIC_PCT,
            ALPHA=ALPHA,
            DELTA_MIN=DELTA_MIN,
            LOBS_MULT=LOBS_MULT,
        )
        row.update(flatten("amb_euc", stats_euc))
        row.update(flatten("amb_geo", stats_geo))
        row.update(flatten("proj", stats_proj))
        results.append(row)
    df = pd.DataFrame(results)
    out_path = OUT_DIR / "results_corrected.csv"
    df.to_csv(out_path, index=False)

    expected = {
        "Pure Drift": "no_recurrence",
        "Curved Drift": "no_recurrence",
        "True Circle": "robust_recurrence",
        "Helix": "projection_hallucination",
    }
    bad = df[df.apply(lambda r: r["decision"] != expected[r["scenario"]], axis=1)]
    if len(bad):
        raise RuntimeError(
            "Four-archetype validation failed expected decisions. "
            f"CSV saved at {out_path}.\n"
            + bad[["scenario", "decision", "amb_geo_pperm", "proj_pperm"]].to_string(index=False)
        )

    elapsed = (time.time() - t0) / 60.0
    print()
    print("=" * 70)
    print(f"Finished in {elapsed:.1f} minutes")
    print(f"Results saved to: {out_path}")
    print("=" * 70)
    summary_cols = [
        "scenario",
        "decision",
        "amb_geo_Lobs",
        "amb_geo_TSR",
        "amb_geo_zrob",
        "amb_geo_pperm",
        "amb_geo_formal_trigger",
        "amb_geo_separation_flag",
        "amb_geo_nominal_trigger",
        "proj_TSR",
        "proj_pperm",
        "proj_formal_trigger",
        "proj_separation_flag",
        "proj_nominal_trigger",
    ]
    print(df[summary_cols])
    return df
if __name__ == "__main__":
    run_all()
