#!/usr/bin/env python3
"""Transition-focused robustness audit.

Purpose
-------
Evaluate graph-geodesic persistent homology across k values on ordered
synthetic helix and circle controls. This complements the clean and noisy
k-sweep outputs by checking decision-transition stability.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Implementation convention
-------------------------
The graph-geodesic construction is imported from audit_common.py, excludes
self-neighbors, uses the symmetrized kNN convention A.maximum(A.T), and fails
loudly if the formal kNN graph remains disconnected at the configured kmax.
"""

from __future__ import annotations
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

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = Path("results") / "transition_focused_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 1200
SUBSAMPLE = 600
N_NULLS = 200

SEEDS = [101, 202, 303, 404]
K_VALUES = [6, 8, 10, 12, 14, 16, 20]

GEODESIC_MAX_K = 50
GEODESIC_PCT = 95.0

ALPHA = 0.05
DELTA_MIN = 1e-3
LOBS_MULT = 5.0

MAD_SCALE = 1.4826
EPS = 1e-12
EPS_SIG = 1e-6

N_JOBS = -1

HELIX_PITCHES = [12, 15, 18, 22]
CIRCLE_RADII = [2.0, 3.0]


# =============================================================================
# STATISTICS
# =============================================================================

def _mad(arr: np.ndarray) -> float:
    """Median absolute deviation."""
    arr = np.asarray(arr, dtype=float)
    return float(np.median(np.abs(arr - np.median(arr))))




# =============================================================================
# SUBSAMPLING
# =============================================================================

def stride_subsample(X: np.ndarray, cap: int) -> np.ndarray:
    """
    Stride subsampling for ordered synthetic trajectories.

    This preserves the temporal/arc ordering of helix and circle controls.
    Random subsampling can create artificial gaps and introduce kNN shortcuts.
    """
    if len(X) <= cap:
        return X

    idx = np.linspace(0, len(X) - 1, cap, dtype=int)
    return X[idx]


# =============================================================================
# GEODESIC PH
# =============================================================================

def max_H1_geodesic(
    X: np.ndarray,
    subsample: int,
    seed: int,
    k_init: int,
) -> tuple[float, int]:
    """Shared graph-geodesic PH via audit_common; returns (Lmax, k_used)."""
    _ = np.random.default_rng(seed)  # kept for signature/reproducibility
    X = stride_subsample(X, subsample)
    if len(X) < 4:
        return 0.0, min(max(1, int(k_init)), max(1, len(X) - 1))
    L, k_used = geodesic_h1_lifetime(
        X,
        k0=k_init,
        kmax=GEODESIC_MAX_K,
        pct=GEODESIC_PCT,
        eps=EPS,
        sentinel_fill=False,
        return_k=True,
    )
    return float(L), int(k_used)


# =============================================================================
# NULL MODEL
# =============================================================================



# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_helix(
    seed: int,
    pitch: float,
    radius: float = 3.0,
) -> np.ndarray:
    """
    Helix negative control.

    Ground truth:
        contractible ordered trajectory.
    """
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 1, N_SAMPLES)
    theta = 6.0 * np.pi * t

    X = np.vstack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        pitch * t,
    ]).T

    X += 0.05 * rng.standard_normal(X.shape)

    return X


def generate_circle(
    seed: int,
    radius: float = 2.5,
) -> np.ndarray:
    """
    Circle positive control.

    Ground truth:
        genuine H1.
    """
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 2.0 * np.pi, N_SAMPLES, endpoint=False)

    X = np.vstack([
        radius * np.cos(t),
        radius * np.sin(t),
        np.zeros_like(t),
    ]).T

    X += 0.05 * rng.standard_normal(X.shape)

    return X


# =============================================================================
# WORKERS
# =============================================================================

def null_worker(
    null_traj: np.ndarray,
    null_index: int,
    base_seed: int,
    k_init: int,
) -> float:
    """
    One null PH evaluation.
    """
    L, _ = max_H1_geodesic(
        null_traj,
        SUBSAMPLE,
        base_seed + 30 + null_index,
        k_init,
    )
    return L


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 72)
    print("Transition-focused robustness audit")
    print("Transition-focused robustness audit")
    print("=" * 72)
    print(f"N_SAMPLES={N_SAMPLES}")
    print(f"SUBSAMPLE={SUBSAMPLE}")
    print(f"N_NULLS={N_NULLS}")
    print(f"SEEDS={SEEDS}")
    print(f"K_VALUES={K_VALUES}")
    print(f"GEODESIC_MAX_K={GEODESIC_MAX_K}")
    print(f"GEODESIC_PCT={GEODESIC_PCT}")
    print(f"ALPHA={ALPHA}")
    print("Formal decision: pperm < ALPHA only")
    print("separation_flag / nominal_trigger are diagnostic only")
    print("=" * 72)

    t0 = time.time()

    configs = (
        [
            ("helix", pitch, k, seed)
            for pitch in HELIX_PITCHES
            for k in K_VALUES
            for seed in SEEDS
        ]
        +
        [
            ("circle", radius, k, seed)
            for radius in CIRCLE_RADII
            for k in K_VALUES
            for seed in SEEDS
        ]
    )

    records = []

    for dtype, param, k, seed in tqdm(configs, desc="Running cells"):
        base_seed = seed * 100000 + k * 1000

        if dtype == "helix":
            X = generate_helix(
                seed=base_seed + 1,
                pitch=param,
            )
            dataset = f"helix_p{param}"
        else:
            X = generate_circle(
                seed=base_seed + 1,
                radius=param,
            )
            dataset = f"circle_r{param}"

        Lobs, k_used = max_H1_geodesic(
            X,
            SUBSAMPLE,
            base_seed + 10,
            k,
        )

        nulls = null_matched_steps(
            X,
            N_NULLS,
            base_seed + 20,
        )

        Lnull = np.array(
            Parallel(n_jobs=N_JOBS)(
                delayed(null_worker)(
                    nulls[i],
                    i,
                    base_seed,
                    k,
                )
                for i in range(N_NULLS)
            ),
            dtype=float,
        )

        stats = compute_stats(Lobs, Lnull)

        records.append({
            "dataset": dataset,
            "type": dtype,
            "param": float(param),
            "k": int(k),
            "k_used": int(k_used),
            "seed": int(seed),
            "N_SAMPLES": N_SAMPLES,
            "SUBSAMPLE": SUBSAMPLE,
            "N_NULLS": N_NULLS,
            "GEODESIC_MAX_K": GEODESIC_MAX_K,
            "GEODESIC_PCT": GEODESIC_PCT,
            "ALPHA": ALPHA,
            "DELTA_MIN": DELTA_MIN,
            "LOBS_MULT": LOBS_MULT,
            **stats,
        })

    df = pd.DataFrame(records)

    raw_path = OUT_DIR / "raw_results.csv"
    df.to_csv(raw_path, index=False)

    print()
    print("=" * 72)
    print(f"Raw results saved: {raw_path}")
    print("=" * 72)

    helix_bad = df[(df["type"] == "helix") & (df["formal_trigger"].astype(bool))]
    if len(helix_bad):
        raise RuntimeError(
            "Transition-focused robustness failed: helix negative control "
            "produced formal geodesic positives.\n"
            + helix_bad[["dataset", "param", "seed", "k", "k_used", "Lobs", "pperm", "TSR"]].to_string(index=False)
        )

    # =========================================================================
    # ANALYSIS SUMMARY
    # =========================================================================

    analysis_rows = []

    for dataset in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == dataset].copy()

        freq = (
            sub.groupby("k")["formal_trigger"]
            .mean()
            .reindex(K_VALUES)
        )

        tsr_by_k = (
            sub.groupby("k")["TSR"]
            .mean()
            .reindex(K_VALUES)
        )

        zrob_by_k = (
            sub.groupby("k")["zrob"]
            .mean()
            .reindex(K_VALUES)
        )

        tau_freq, tau_freq_p = kendalltau(
            K_VALUES,
            freq.values,
        )

        tau_tsr, tau_tsr_p = kendalltau(
            K_VALUES,
            tsr_by_k.values,
        )

        flip_count = 0
        total_pairs = 0

        for seed in SEEDS:
            sdata = sub[sub["seed"] == seed].sort_values("k")
            vals = sdata["formal_trigger"].astype(bool).to_numpy()

            for i in range(len(vals) - 1):
                total_pairs += 1
                if vals[i + 1] != vals[i]:
                    flip_count += 1

        flip_rate = (
            flip_count / total_pairs
            if total_pairs > 0
            else np.nan
        )

        analysis_rows.append({
            "dataset": dataset,
            "type": str(sub["type"].iloc[0]),
            "param": float(sub["param"].iloc[0]),
            "n_total": int(len(sub)),
            "n_formal_positive": int(sub["formal_trigger"].sum()),
            "formal_fire_rate": float(sub["formal_trigger"].mean()),
            "min_pperm": float(sub["pperm"].min()),
            "mean_Lobs": float(sub["Lobs"].mean()),
            "mean_TSR": float(sub["TSR"].mean()),
            "mean_zrob": float(sub["zrob"].mean()),
            "max_TSR": float(sub["TSR"].max()),
            "kendall_tau_formal_freq": tau_freq,
            "tau_formal_freq_p": tau_freq_p,
            "kendall_tau_TSR": tau_tsr,
            "tau_TSR_p": tau_tsr_p,
            "flip_rate": float(flip_rate),
        })

    analysis_df = pd.DataFrame(analysis_rows)

    analysis_path = OUT_DIR / "analysis_summary.csv"
    analysis_df.to_csv(analysis_path, index=False)

    print()
    print("=" * 72)
    print("Analysis summary")
    print("=" * 72)
    print(analysis_df.to_string(index=False))
    print(f"Analysis summary saved: {analysis_path}")

    # =========================================================================
    # PER-K SUMMARY
    # =========================================================================

    per_k_summary = (
        df.groupby(["dataset", "type", "param", "k"])
        .agg(
            fire_rate=("formal_trigger", "mean"),
            n_positive=("formal_trigger", "sum"),
            n_total=("formal_trigger", "count"),
            mean_Lobs=("Lobs", "mean"),
            mean_TSR=("TSR", "mean"),
            mean_zrob=("zrob", "mean"),
            min_pperm=("pperm", "min"),
            max_pperm=("pperm", "max"),
        )
        .reset_index()
    )

    per_k_path = OUT_DIR / "per_k_summary.csv"
    per_k_summary.to_csv(per_k_path, index=False)

    print(f"Per-k summary saved: {per_k_path}")

    elapsed = (time.time() - t0) / 60.0

    print()
    print("=" * 72)
    print(f"Finished in {elapsed:.1f} minutes")
    print("=" * 72)

    return df, analysis_df


if __name__ == "__main__":
    main()
