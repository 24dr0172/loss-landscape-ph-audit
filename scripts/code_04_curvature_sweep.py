#!/usr/bin/env python3
"""Curvature/projection sensitivity sweep.

Purpose
-------
Test whether increasing curvature in a contractible trajectory increases
projection-level apparent H1 while the graph-geodesic formal test remains
negative.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Outputs
-------
Writes curvature_sweep_results.csv and curvature_sweep_summary.csv under
results/curvature_sweep/.
"""
# Shared audit primitives.
from pathlib import Path as _AuditPath
import sys as _audit_sys
_audit_sys.path.insert(0, str(_AuditPath(__file__).resolve().parents[1]))
from audit_common import (
    compute_stats,
    matched_step_null,
    matched_step_nulls,
    matched_step_nulls as null_matched_steps,
    block_null,
    block_step_nulls,
    block_permutation_indices,
    geodesic_distance_matrix,
    geodesic_h1_lifetime,
    h1_lifetime_from_distance_matrix,
    safe_lifetime,
    sha256_file,
    stride_subsample as audit_stride_subsample,
)
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from ripser import ripser
# ============================================================
# CONFIG
# ============================================================
OUT_DIR = Path("results") / "curvature_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 42
N_SAMPLES = 1200
NOISE = 0.03
N_NULLS = 300
TOTAL_PROJECTIONS = 100
SUBSAMPLE = 600
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
LAMBDA_GRID = np.linspace(0, 1, 8)
# ============================================================
# STATISTICS
# ============================================================
def _mad(arr):
    """Median absolute deviation."""
    arr = np.asarray(arr, dtype=float)
    return float(np.median(np.abs(arr - np.median(arr))))
# ============================================================
# PH CORE
# ============================================================
def adaptive_thresh(X, pct=95.0):
    """Euclidean percentile filtration threshold."""
    d = pdist(X)
    return float(np.percentile(d, pct)) if len(d) > 0 else None
def random_subsample(X, cap, rng):
    """Random subsampling for Euclidean/projection diagnostics."""
    if len(X) <= cap:
        return X
    idx = rng.choice(len(X), cap, replace=False)
    return X[idx]
def stride_subsample(X, cap):
    """
    Stride subsampling for ordered synthetic trajectories.
    This preserves temporal/arc ordering and avoids artificial gaps along
    helix-like curves.
    """
    if len(X) <= cap:
        return X
    idx = np.linspace(0, len(X) - 1, cap, dtype=int)
    return X[idx]
def max_H1_euclidean(X, subsample, rng):
    """
    Maximum finite H1 lifetime using Euclidean Vietoris-Rips PH.
    Used for projection diagnostics.
    """
    X = random_subsample(X, subsample, rng)
    thresh = adaptive_thresh(X, pct=95.0)
    kwargs = {"maxdim": 1}
    if thresh is not None:
        kwargs["thresh"] = thresh
    dgms = ripser(X, **kwargs)["dgms"]
    if len(dgms) < 2 or len(dgms[1]) == 0:
        return 0.0
    H1 = dgms[1]
    finite = H1[np.isfinite(H1[:, 1])]
    if len(finite) == 0:
        return 0.0
    return float(np.max(finite[:, 1] - finite[:, 0]))
def max_H1_geodesic(X, subsample, rng=None):
    """Shared graph-geodesic PH via audit_common.geodesic_distance_matrix."""
    X = stride_subsample(X, subsample)
    if len(X) < 4:
        return 0.0
    return geodesic_h1_lifetime(
        X,
        k0=GEODESIC_K,
        kmax=GEODESIC_MAX_K,
        pct=GEODESIC_PCT,
        eps=EPS,
        sentinel_fill=False,
    )
# ============================================================
# NULL MODEL
# ============================================================
# ============================================================
# DATA GENERATOR
# ============================================================
def generate_family(lam, seed):
    """
    Curvature/helix family.
    lam = 0:
        nearly straight vertical drift.
    lam = 1:
        one full circular turn while drifting upward.
    Ground truth:
        contractible for all lam in this sweep.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N_SAMPLES)
    X = np.vstack([
        3.0 * np.cos(2.0 * np.pi * lam * t),
        3.0 * np.sin(2.0 * np.pi * lam * t),
        5.0 * t,
    ]).T
    X += NOISE * rng.standard_normal(X.shape)
    return X
# ============================================================
# SWEEP
# ============================================================
def run_sweep():
    print("=" * 70)
    print("Curvature/projection sensitivity sweep")
    print("=" * 70)
    print(f"RNG_SEED={RNG_SEED}")
    print(f"N_SAMPLES={N_SAMPLES}")
    print(f"SUBSAMPLE={SUBSAMPLE}")
    print(f"N_NULLS={N_NULLS}")
    print(f"TOTAL_PROJECTIONS={TOTAL_PROJECTIONS}")
    print(f"GEODESIC_K={GEODESIC_K}")
    print(f"GEODESIC_PCT={GEODESIC_PCT}")
    print(f"LAMBDA_GRID={LAMBDA_GRID}")
    print(f"Formal decision: pperm < {ALPHA}; separation_flag is diagnostic only")
    print("zrob and TSR are descriptive only.")
    print("=" * 70)
    t0 = time.time()
    results = []
    for i, lam in enumerate(LAMBDA_GRID):
        print()
        print("-" * 70)
        print(f"Running lambda={lam:.4f}  ({i+1}/{len(LAMBDA_GRID)})")
        print("-" * 70)
        base_seed = RNG_SEED + i * 500000
        X = generate_family(lam, base_seed)
        # Separate null ensembles for geodesic and projection diagnostics,
        # used for this appendix diagnostic.
        nulls_geo = null_matched_steps(
            X,
            N_NULLS,
            np.random.default_rng(base_seed + 100),
        )
        nulls_proj = null_matched_steps(
            X,
            N_NULLS,
            np.random.default_rng(base_seed + 200),
        )
        # --------------------------------------------------------
        # Geodesic formal test
        # --------------------------------------------------------
        L_geo = max_H1_geodesic(
            X,
            SUBSAMPLE,
            np.random.default_rng(base_seed + 10),
        )
        H_geo = np.array(
            Parallel(n_jobs=N_JOBS)(
                delayed(max_H1_geodesic)(
                    nulls_geo[j],
                    SUBSAMPLE,
                    np.random.default_rng(base_seed + 1000 + j),
                )
                for j in range(N_NULLS)
            )
        )
        stats_geo = compute_stats(L_geo, H_geo)
        # --------------------------------------------------------
        # Projection max-statistic diagnostic
        # --------------------------------------------------------
        proj_rng = np.random.default_rng(base_seed + 3000)
        proj_mats = []
        for _ in range(TOTAL_PROJECTIONS):
            Q, _ = np.linalg.qr(
                proj_rng.standard_normal((3, 2))
            )
            proj_mats.append(Q[:, :2])
        obs_vals = np.array([
            max_H1_euclidean(
                X @ P,
                SUBSAMPLE,
                np.random.default_rng(base_seed + 4000 + k),
            )
            for k, P in enumerate(proj_mats)
        ])
        L_proj = float(np.max(obs_vals))
        def null_proj_worker(null_traj, null_index, pmats, seed_base):
            vals = []
            for k, P in enumerate(pmats):
                val = max_H1_euclidean(
                    null_traj @ P,
                    SUBSAMPLE,
                    np.random.default_rng(seed_base + null_index * 2000 + k),
                )
                vals.append(val)
            return float(np.max(vals))
        H_proj = np.array(
            Parallel(n_jobs=N_JOBS)(
                delayed(null_proj_worker)(
                    nulls_proj[j],
                    j,
                    proj_mats,
                    base_seed + 5000,
                )
                for j in range(N_NULLS)
            )
        )
        stats_proj = compute_stats(L_proj, H_proj)
        print(
            f"Geodesic:"
            f" Lobs={stats_geo['Lobs']:.5f}"
            f" null_med={stats_geo['null_med']:.5f}"
            f" null_max={stats_geo['null_max']:.5f}"
            f" TSR={stats_geo['TSR']:.3f}"
            f" zrob={stats_geo['zrob']:+.2f}"
            f" p={stats_geo['pperm']:.4f}"
            f" formal={stats_geo['formal_trigger']}"
        )
        print(
            f"Projection:"
            f" Lobs={stats_proj['Lobs']:.5f}"
            f" null_med={stats_proj['null_med']:.5f}"
            f" null_max={stats_proj['null_max']:.5f}"
            f" TSR={stats_proj['TSR']:.3f}"
            f" zrob={stats_proj['zrob']:+.2f}"
            f" p={stats_proj['pperm']:.4f}"
            f" formal={stats_proj['formal_trigger']}"
        )
        row = dict(
            lambda_val=float(lam),
            geo_Lobs=stats_geo["Lobs"],
            geo_null_med=stats_geo["null_med"],
            geo_null_max=stats_geo["null_max"],
            geo_TSR=stats_geo["TSR"],
            geo_pperm=stats_geo["pperm"],
            geo_zrob=stats_geo["zrob"],
            geo_delta=stats_geo["delta"],
            geo_zero_frac=stats_geo["zero_frac"],
            geo_fallback=stats_geo["fallback"],
            geo_separation_flag=stats_geo["separation_flag"],
            geo_formal_trigger=stats_geo["formal_trigger"],
            geo_nominal_trigger=stats_geo["nominal_trigger"],
            proj_Lobs=stats_proj["Lobs"],
            proj_null_med=stats_proj["null_med"],
            proj_null_max=stats_proj["null_max"],
            proj_TSR=stats_proj["TSR"],
            proj_pperm=stats_proj["pperm"],
            proj_zrob=stats_proj["zrob"],
            proj_delta=stats_proj["delta"],
            proj_zero_frac=stats_proj["zero_frac"],
            proj_fallback=stats_proj["fallback"],
            proj_separation_flag=stats_proj["separation_flag"],
            proj_formal_trigger=stats_proj["formal_trigger"],
            proj_nominal_trigger=stats_proj["nominal_trigger"],
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
        results.append(row)
    df = pd.DataFrame(results)
    rho, p = spearmanr(df["lambda_val"], df["proj_TSR"])
    print()
    print("=" * 70)
    print("Curvature Sweep Summary")
    print("=" * 70)
    print(f"Spearman proj TSR vs lambda: rho={rho:.6f}, p={p:.6g}")
    print(f"Any geodesic formal trigger? {bool(df['geo_formal_trigger'].any())}")
    print(f"Max geo TSR: {df['geo_TSR'].max():.6f}")
    any_geo_formal = bool(df["geo_formal_trigger"].any())
    if any_geo_formal:
        print("WARNING: Geodesic formal trigger fired for at least one lambda.")
    else:
        print("Formal trigger abstained for all lambda values, as expected.")

    out_csv = OUT_DIR / "curvature_sweep_results.csv"
    df.to_csv(out_csv, index=False)

    summary_df = pd.DataFrame([{
        "spearman_proj_TSR_vs_lambda": rho,
        "spearman_p": p,
        "any_geo_formal_trigger": any_geo_formal,
        "max_geo_TSR": float(df["geo_TSR"].max()),
        "N_SAMPLES": N_SAMPLES,
        "SUBSAMPLE": SUBSAMPLE,
        "N_NULLS": N_NULLS,
        "TOTAL_PROJECTIONS": TOTAL_PROJECTIONS,
        "GEODESIC_K": GEODESIC_K,
        "GEODESIC_MAX_K": GEODESIC_MAX_K,
        "GEODESIC_PCT": GEODESIC_PCT,
        "ALPHA": ALPHA,
    }])
    summary_csv = OUT_DIR / "curvature_sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved summary CSV: {summary_csv}")

    if any_geo_formal:
        bad = df[df["geo_formal_trigger"]]
        raise RuntimeError(
            "Curvature sweep failed: geodesic formal trigger fired for a "
            "contractible family.\n"
            + bad[["lambda_val", "geo_Lobs", "geo_pperm", "geo_TSR"]].to_string(index=False)
        )

    elapsed = (time.time() - t0) / 60.0
    print(f"Elapsed time: {elapsed:.1f} minutes")
    print()
    print(df[[
        "lambda_val",
        "geo_TSR",
        "geo_pperm",
        "geo_formal_trigger",
        "geo_separation_flag",
        "geo_nominal_trigger",
        "proj_TSR",
        "proj_pperm",
        "proj_formal_trigger",
        "proj_separation_flag",
        "proj_nominal_trigger",
    ]])
    return df
if __name__ == "__main__":
    run_sweep()
