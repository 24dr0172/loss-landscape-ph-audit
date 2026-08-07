#!/usr/bin/env python3
"""Synthetic validation pipeline.

Purpose
-------
Run two controlled trajectory diagnostics: a 3D helix metric-shortcut pathology
and a 50D overpass projection-amplification pathology.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Implementation convention
-------------------------
Graph-geodesic calls use the shared audit_common.py construction, median
rescaling over positive geodesic distances, percentile filtration thresholds,
and sentinel_fill=False for formal geodesic evaluations.

Outputs
-------
Writes synthetic_validation_results.csv under results/synthetic_validation/.
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
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from ripser import ripser
from joblib import Parallel, delayed
# ============================================================
# CONFIG
# ============================================================
OUT_DIR = Path("results") / "synthetic_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 42
N_SAMPLES = 1200
SUBSAMPLE = 600
N_NULLS = 300
GEODESIC_K = 12
# Synthetic overpass matched-step nulls can be disconnected at kmax=50.
# For this synthetic validation only, allow adaptive k to reach full
# connectivity rather than using sentinel fill or assigning Lmax=0.
GEODESIC_MAX_K = SUBSAMPLE - 1
GEODESIC_PCT = 95.0
ALPHA = 0.05
DELTA_MIN = 1e-3
LOBS_MULT = 5.0
EPS = 1e-12
EPS_SIG = 1e-6
MAD_SCALE = 1.4826
N_JOBS = -1
# ============================================================
# DATA GENERATORS
# ============================================================
def generate_helix(rng):
    """
    3D helix: metric shortcut pathology.
    Ground truth:
        Contractible trajectory.
    Expected:
        Ambient Euclidean PH may fire due to shortcuts across coils.
        Ambient geodesic PH should reject.
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
def generate_pure_projection_hallucination(rng, d=50):
    """
    50D alpha-curve overpass.
    An open curve crosses itself in the first two coordinates. A large but
    localised displacement in the remaining 48 coordinates separates the
    apparent crossing in ambient space.
    Expected:
        Ambient Euclidean may show an apparent H1 shortcut.
        Ambient geodesic should reject intrinsic recurrence.
        PCA 2D Euclidean should show projection-amplified apparent H1.
    """
    t = np.linspace(-2.2, 2.2, N_SAMPLES)
    X = np.zeros((N_SAMPLES, d))
    # High-variance 2D alpha curve
    X[:, 0] = 500.0 * (t**3 - 3.0 * t)
    X[:, 1] = 500.0 * (t**2 - 1.0)
    # Localised high-dimensional overpass bump at t = ±sqrt(3)
    t_cross = np.sqrt(3.0)
    bump = (
        1000.0 * np.exp(-200.0 * (t + t_cross) ** 2)
        - 1000.0 * np.exp(-200.0 * (t - t_cross) ** 2)
    )
    for i in range(2, d):
        X[:, i] = bump
    X += 0.5 * rng.standard_normal(X.shape)
    return X
# ============================================================
# PH CORE
# ============================================================
def _mad(a: np.ndarray) -> float:
    """Median absolute deviation."""
    a = np.asarray(a, dtype=float)
    return float(np.median(np.abs(a - np.median(a))))
def _adaptive_thresh(X, pct=95.0):
    """Euclidean percentile filtration threshold."""
    if len(X) < 2:
        return None
    d = pdist(X)
    if len(d) == 0:
        return None
    return float(np.percentile(d, pct))
def _subsample(X, cap, rng=None, stride=False):
    """
    Subsampling helper.
    For Euclidean ambient diagnostics:
        random subsampling.
    For geodesic PH and PCA-overpass diagnostic:
        stride subsampling to preserve ordered synthetic curve structure.
    """
    if len(X) <= cap:
        return X
    if stride:
        idx = np.linspace(0, len(X) - 1, cap, dtype=int)
        return X[idx]
    if rng is None:
        raise ValueError("rng must be provided for random subsampling.")
    idx = rng.choice(len(X), cap, replace=False)
    return X[idx]
def max_H1_euclidean(X, subsample, rng):
    """
    Maximum finite H1 lifetime using Euclidean Vietoris-Rips PH.
    Uses random subsampling.
    """
    X = _subsample(X, subsample, rng=rng, stride=False)
    thresh = _adaptive_thresh(X, pct=95.0)
    kwargs = {"maxdim": 1}
    if thresh is not None:
        kwargs["thresh"] = thresh
    dgms = ripser(X, **kwargs)["dgms"]
    return safe_lifetime(dgms, censoring_threshold=thresh)
def max_H1_euclidean_stride(X, subsample):
    """
    Maximum finite H1 lifetime using Euclidean Vietoris-Rips PH.
    Uses deterministic stride subsampling.
    This is used for the PCA-overpass diagnostic because the projected
    overpass is an ordered synthetic curve. Random subsampling can remove
    crucial crossing/loop support and weaken the artifact.
    """
    X = _subsample(X, subsample, rng=None, stride=True)
    thresh = _adaptive_thresh(X, pct=95.0)
    kwargs = {"maxdim": 1}
    if thresh is not None:
        kwargs["thresh"] = thresh
    dgms = ripser(X, **kwargs)["dgms"]
    return safe_lifetime(dgms, censoring_threshold=thresh)
def max_H1_geodesic(X, subsample, rng=None):
    """Shared graph-geodesic PH via audit_common.geodesic_distance_matrix."""
    X = _subsample(X, subsample, rng=rng, stride=True)
    if len(X) < 4:
        return 0.0
    return geodesic_h1_lifetime(
        X,
        k0=GEODESIC_K,
        kmax=GEODESIC_MAX_K,
        pct=GEODESIC_PCT,
        eps=EPS,
        sentinel_fill=False,
        context_label="code_01_synthetic_validation",
    )
def max_H1_euclidean_seeded(X, subsample, seed):
    """Seeded wrapper for parallel Euclidean PH."""
    return max_H1_euclidean(
        X,
        subsample,
        np.random.default_rng(int(seed)),
    )
def max_H1_geodesic_seeded(X, subsample, seed):
    """Seeded wrapper for parallel geodesic PH."""
    return max_H1_geodesic(
        X,
        subsample,
        np.random.default_rng(int(seed)),
    )
# ============================================================
# NULL MODEL
# ============================================================
# ============================================================
# STATISTICS
# ============================================================
# ============================================================
# REPORTING
# ============================================================
def print_result(name, stats, expected_label):
    """Pretty-print one PH result."""
    formal = stats["formal_trigger"]
    print(f"[{name}]")
    print(
        f"  Lobs={stats['Lobs']:.5f}"
        f"  null_med={stats['null_med']:.5f}"
        f"  null_max={stats['null_max']:.5f}"
    )
    print(
        f"  TSR={stats['TSR']:.3f}"
        f"  zrob={stats['zrob']:+.2f}"
        f"  pperm={stats['pperm']:.4f}"
        f"  delta={stats['delta']:.5f}"
        f"  fallback={stats['fallback']}"
        f"  formal={formal}"
    )
    print(
        f"  Decision: {'SIGNIFICANT H1' if formal else 'NO H1'}"
        f" — expected: {expected_label}"
    )
    print()
def flatten(prefix, stats):
    """Flatten a stats dictionary for CSV output."""
    return {f"{prefix}_{key}": value for key, value in stats.items()}
# ============================================================
# EXPERIMENT 1: METRIC SHORTCUT PATHOLOGY
# ============================================================
def run_experiment_1(rng):
    print()
    print("=" * 70)
    print("EXPERIMENT 1: METRIC SHORTCUT PATHOLOGY (3D Helix)")
    print("=" * 70)
    print("Expected: Euclidean → SIGNIFICANT (false positive)")
    print("          Geodesic  → NO H1 (correct negative)")
    print()
    X = generate_helix(rng)
    nulls = null_matched_steps(
        X,
        N_NULLS,
        rng,
    )
    # Ambient Euclidean
    Lobs_euc = max_H1_euclidean(
        X,
        SUBSAMPLE,
        np.random.default_rng(RNG_SEED + 10),
    )
    euc_seeds = RNG_SEED + 10000 + np.arange(N_NULLS) * 1009
    Lnull_euc = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(max_H1_euclidean_seeded)(
                nulls[i],
                SUBSAMPLE,
                int(euc_seeds[i]),
            )
            for i in range(N_NULLS)
        )
    )
    stats_euc = compute_stats(Lobs_euc, Lnull_euc)
    print_result(
        "Ambient Euclidean",
        stats_euc,
        "SIGNIFICANT — metric shortcut false positive",
    )
    # Ambient Geodesic formal test
    Lobs_geo = max_H1_geodesic(
        X,
        SUBSAMPLE,
        np.random.default_rng(RNG_SEED + 20),
    )
    geo_seeds = RNG_SEED + 20000 + np.arange(N_NULLS) * 1009
    Lnull_geo = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(max_H1_geodesic_seeded)(
                nulls[i],
                SUBSAMPLE,
                int(geo_seeds[i]),
            )
            for i in range(N_NULLS)
        )
    )
    stats_geo = compute_stats(Lobs_geo, Lnull_geo)
    print_result(
        "Ambient Geodesic (formal test)",
        stats_geo,
        "NO H1 — correct rejection",
    )
    result = dict(
        experiment="Exp 1 — Metric Shortcut Pathology",
        euc_formal_trigger=stats_euc["formal_trigger"],
        geo_formal_trigger=stats_geo["formal_trigger"],
        expected_euc_formal_trigger=True,
        expected_geo_formal_trigger=False,
    )
    result.update(flatten("euc", stats_euc))
    result.update(flatten("geo", stats_geo))
    return result
# ============================================================
# EXPERIMENT 2: PROJECTION AMPLIFICATION
# ============================================================
def run_experiment_2(rng):
    print()
    print("=" * 75)
    print("EXPERIMENT 2: PROJECTION AMPLIFICATION (50D Alpha Curve Overpass)")
    print("=" * 75)
    print("Expected: Ambient Euclidean  → SIGNIFICANT (ambient shortcut)")
    print("          Ambient Geodesic   → NO H1 (correctly sees no loop)")
    print("          PCA (2D) Euclidean → SIGNIFICANT (projection artifact)")
    print()
    X = generate_pure_projection_hallucination(rng, d=50)
    nulls = null_matched_steps(
        X,
        N_NULLS,
        rng,
    )
    # 1. Ambient Euclidean
    Lobs_euc = max_H1_euclidean(
        X,
        SUBSAMPLE,
        np.random.default_rng(RNG_SEED + 30),
    )
    euc_seeds = RNG_SEED + 30000 + np.arange(N_NULLS) * 1009
    Lnull_euc = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(max_H1_euclidean_seeded)(
                nulls[i],
                SUBSAMPLE,
                int(euc_seeds[i]),
            )
            for i in range(N_NULLS)
        )
    )
    stats_euc = compute_stats(Lobs_euc, Lnull_euc)
    print_result(
        "Ambient Euclidean",
        stats_euc,
        "SIGNIFICANT — metric shortcut across the 48D gap",
    )
    # 2. Ambient geodesic formal test
    Lobs_geo = max_H1_geodesic(
        X,
        SUBSAMPLE,
        np.random.default_rng(RNG_SEED + 40),
    )
    geo_seeds = RNG_SEED + 40000 + np.arange(N_NULLS) * 1009
    Lnull_geo = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(max_H1_geodesic_seeded)(
                nulls[i],
                SUBSAMPLE,
                int(geo_seeds[i]),
            )
            for i in range(N_NULLS)
        )
    )
    stats_geo = compute_stats(Lobs_geo, Lnull_geo)
    print_result(
        "Ambient Geodesic (formal test)",
        stats_geo,
        "NO H1 — correctly sees no intrinsic loop",
    )
    # 3. PCA 2D Euclidean diagnostic
    #
    # Important correction:
    # Use deterministic stride subsampling for PCA diagnostic because the
    # projected overpass is an ordered synthetic curve. Random subsampling
    # can weaken the projected-loop artifact.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    nulls_pca = [pca.transform(n) for n in nulls]
    print(
        f"  [PCA] Explained variance (PC1, PC2) = "
        f"({pca.explained_variance_ratio_[0]:.3f}, "
        f"{pca.explained_variance_ratio_[1]:.3f})"
    )
    print()
    Lobs_pca = max_H1_euclidean_stride(
        X_pca,
        SUBSAMPLE,
    )
    Lnull_pca = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(max_H1_euclidean_stride)(
                nulls_pca[i],
                SUBSAMPLE,
            )
            for i in range(N_NULLS)
        )
    )
    stats_pca = compute_stats(Lobs_pca, Lnull_pca)
    print_result(
        "PCA (2D) Euclidean",
        stats_pca,
        "SIGNIFICANT — amplified projection hallucination",
    )
    result = dict(
        experiment="Exp 2 — Projection Amplification",
        euc_formal_trigger=stats_euc["formal_trigger"],
        geo_formal_trigger=stats_geo["formal_trigger"],
        pca_formal_trigger=stats_pca["formal_trigger"],
        expected_euc_formal_trigger=True,
        expected_geo_formal_trigger=False,
        expected_pca_formal_trigger=True,
        pca_var_pc1=float(pca.explained_variance_ratio_[0]),
        pca_var_pc2=float(pca.explained_variance_ratio_[1]),
    )
    result.update(flatten("euc", stats_euc))
    result.update(flatten("geo", stats_geo))
    result.update(flatten("pca", stats_pca))
    return result
# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Synthetic validation pipeline")
    print("=" * 70)
    print(f"RNG_SEED={RNG_SEED}")
    print(f"N_SAMPLES={N_SAMPLES}")
    print(f"SUBSAMPLE={SUBSAMPLE}")
    print(f"N_NULLS={N_NULLS}")
    print(f"GEODESIC_K={GEODESIC_K}")
    print(f"GEODESIC_PCT={GEODESIC_PCT}")
    print(f"Formal decision: pperm < {ALPHA}; separation_flag is diagnostic only")
    print("zrob and TSR are descriptive only.")
    print("=" * 70)
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    r1 = run_experiment_1(rng)
    r2 = run_experiment_2(rng)
    rows = [r1, r2]
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "synthetic_validation_results.csv"
    df.to_csv(out_csv, index=False)
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"Exp 1 — Euclidean formal={r1['euc_formal_trigger']} "
        f"(expect True) | Geodesic formal={r1['geo_formal_trigger']} "
        f"(expect False)"
    )
    print(
        f"Exp 2 — Euclidean formal={r2['euc_formal_trigger']} "
        f"(expect True) | Geodesic formal={r2['geo_formal_trigger']} "
        f"(expect False) | PCA formal={r2['pca_formal_trigger']} "
        f"(expect True)"
    )
    all_pass = (
        bool(r1["euc_formal_trigger"]) is True
        and bool(r1["geo_formal_trigger"]) is False
        and bool(r2["euc_formal_trigger"]) is True
        and bool(r2["geo_formal_trigger"]) is False
        and bool(r2["pca_formal_trigger"]) is True
    )
    print()
    if not all_pass:
        raise RuntimeError(
            "Synthetic validation failed expected control pattern. "
            "Expected: Exp1 Euclidean=True, Exp1 Geodesic=False, "
            "Exp2 Euclidean=True, Exp2 Geodesic=False, Exp2 PCA=True. "
            f"Observed: Exp1 Euclidean={r1['euc_formal_trigger']}, "
            f"Exp1 Geodesic={r1['geo_formal_trigger']}, "
            f"Exp2 Euclidean={r2['euc_formal_trigger']}, "
            f"Exp2 Geodesic={r2['geo_formal_trigger']}, "
            f"Exp2 PCA={r2['pca_formal_trigger']}. "
            f"CSV saved at {out_csv}"
        )
    print("All checks passed.")
    elapsed = (time.time() - t0) / 60.0
    print()
    print(f"Saved CSV: {out_csv}")
    print(f"Elapsed time: {elapsed:.1f} minutes")
    return df
if __name__ == "__main__":
    main()
