#!/usr/bin/env python3
"""Noisy-circle graph-geodesic k-sweep.

Purpose
-------
Test whether noisy circle positive controls remain detectable under
graph-geodesic persistent homology across k values.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Outputs
-------
Writes raw_results.csv, analysis_summary.csv, and per_k_summary.csv under
results/circle_noise_robustness/.
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
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from tqdm import tqdm
# ============================================================
# CONFIG
# ============================================================
OUT_DIR = Path("results") / "circle_noise_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N = 1200
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
NOISE = 0.1
N_JOBS = -1
print("=" * 70)
print("Noisy-circle graph-geodesic k-sweep")
print("=" * 70)
print(f"N={N}")
print(f"SUBSAMPLE={SUBSAMPLE}")
print(f"N_NULLS={N_NULLS}")
print(f"SEEDS={SEEDS}")
print(f"K_VALUES={K_VALUES}")
print(f"NOISE={NOISE}")
print(f"Formal decision: pperm < {ALPHA}; separation_flag is diagnostic only")
print("zrob and TSR are descriptive only.")
print("=" * 70)
# ============================================================
# ROBUST STATS
# ============================================================
def _mad(arr):
    """Median absolute deviation."""
    arr = np.asarray(arr, dtype=float)
    return float(np.median(np.abs(arr - np.median(arr))))
# ============================================================
# DATA GENERATOR
# ============================================================
def generate_circle(seed, radius):
    """
    Noisy circle positive control.
    Ground truth:
        genuine H1.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2.0 * np.pi, N, endpoint=False)
    X = np.vstack([
        radius * np.cos(t),
        radius * np.sin(t),
        np.zeros_like(t),
    ]).T
    X += NOISE * rng.standard_normal(X.shape)
    return X
CIRCLE_VARIANTS = [
    ("circle_r2_noise01", 2.0),
    ("circle_r3_noise01", 3.0),
]
# ============================================================
# GEODESIC PH
# ============================================================
def stride_subsample(X, cap):
    """
    Stride subsampling for ordered synthetic trajectories.
    For circle controls this preserves the circular ordering and avoids
    random gaps that can destabilize kNN geodesic construction.
    """
    if len(X) <= cap:
        return X
    idx = np.linspace(0, len(X) - 1, cap, dtype=int)
    return X[idx]
def max_H1_geodesic(X, subsample, seed, k_init):
    """Shared graph-geodesic PH via audit_common.geodesic_distance_matrix."""
    _ = np.random.default_rng(seed)  # kept for signature/reproducibility
    X = stride_subsample(X, subsample)
    if len(X) < 4:
        return 0.0, min(k_init, max(1, len(X) - 1))
    L, k_used = geodesic_h1_lifetime(
        X,
        k0=k_init,
        kmax=GEODESIC_MAX_K,
        pct=GEODESIC_PCT,
        eps=EPS,
        sentinel_fill=False,
        return_k=True,
    )
    return L, k_used
# ============================================================
# NULL MODEL
# ============================================================
# ============================================================
# WORKER
# ============================================================
def null_worker(null_traj, null_index, base_seed, k_init):
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
# ============================================================
# RUN
# ============================================================
def main():
    records = []
    t0 = time.time()
    configs = [
        (name, radius, k, seed)
        for name, radius in CIRCLE_VARIANTS
        for k in K_VALUES
        for seed in SEEDS
    ]
    for name, radius, k, seed in tqdm(configs):
        base_seed = seed * 100000 + k * 1000
        X = generate_circle(
            base_seed + 1,
            radius,
        )
        Lobs, k_used = max_H1_geodesic(
            X,
            SUBSAMPLE,
            base_seed + 10,
            k,
        )
        nulls = matched_step_nulls(
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
            )
        )
        stats = compute_stats(
            Lobs,
            Lnull,
        )
        records.append({
            "dataset": name,
            "radius": radius,
            "k": k,
            "k_used": k_used,
            "seed": seed,
            "N": N,
            "SUBSAMPLE": SUBSAMPLE,
            "N_NULLS": N_NULLS,
            "NOISE": NOISE,
            "GEODESIC_MAX_K": GEODESIC_MAX_K,
            "GEODESIC_PCT": GEODESIC_PCT,
            "ALPHA": ALPHA,
            "DELTA_MIN": DELTA_MIN,
            "LOBS_MULT": LOBS_MULT,
            **stats,
        })
    df = pd.DataFrame(records)
    raw_path = OUT_DIR / "raw_results.csv"
    df.to_csv(
        raw_path,
        index=False,
    )
    print()
    print("=" * 70)
    print(f"Raw results saved: {raw_path}")
    print("=" * 70)
    # ============================================================
    # ANALYSIS
    # ============================================================
    analysis_rows = []
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset].copy()
        freq_by_k = subset.groupby("k")["formal_trigger"].mean()
        zrob_by_k = subset.groupby("k")["zrob"].mean()
        tsr_by_k = subset.groupby("k")["TSR"].mean()
        k_vals = sorted(freq_by_k.index)
        freq_vals = [freq_by_k[k] for k in k_vals]
        tsr_vals = [tsr_by_k[k] for k in k_vals]
        tau_freq, tau_freq_p = kendalltau(
            k_vals,
            freq_vals,
        )
        tau_tsr, tau_tsr_p = kendalltau(
            k_vals,
            tsr_vals,
        )
        flip_count = 0
        total_pairs = 0
        for seed in SEEDS:
            sdata = subset[
                subset["seed"] == seed
            ].sort_values("k")
            vals = sdata[
                "formal_trigger"
            ].astype(bool).values
            for j in range(len(vals) - 1):
                total_pairs += 1
                if vals[j + 1] != vals[j]:
                    flip_count += 1
        flip_rate = (
            flip_count / total_pairs
            if total_pairs > 0
            else np.nan
        )
        analysis_rows.append({
            "dataset": dataset,
            "kendall_tau_formal_freq": tau_freq,
            "tau_formal_freq_p": tau_freq_p,
            "kendall_tau_TSR": tau_tsr,
            "tau_TSR_p": tau_tsr_p,
            "flip_rate": flip_rate,
            "mean_zrob": float(subset["zrob"].mean()),
            "mean_TSR": float(subset["TSR"].mean()),
            "max_TSR": float(subset["TSR"].max()),
            "min_pperm": float(subset["pperm"].min()),
            "n_formal_positive": int(subset["formal_trigger"].sum()),
            "n_total": int(len(subset)),
            "formal_fire_rate": float(subset["formal_trigger"].mean()),
        })
    analysis_df = pd.DataFrame(analysis_rows)
    analysis_path = OUT_DIR / "analysis_summary.csv"
    analysis_df.to_csv(
        analysis_path,
        index=False,
    )
    print()
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(analysis_df.to_string(index=False))
    print(f"Analysis summary saved: {analysis_path}")
    # Optional per-k summary for easy table inspection
    per_k_summary = (
        df.groupby(["dataset", "k"])
        .agg(
            fire_rate=("formal_trigger", "mean"),
            mean_Lobs=("Lobs", "mean"),
            mean_TSR=("TSR", "mean"),
            mean_zrob=("zrob", "mean"),
            min_pperm=("pperm", "min"),
            max_pperm=("pperm", "max"),
        )
        .reset_index()
    )
    per_k_path = OUT_DIR / "per_k_summary.csv"
    per_k_summary.to_csv(
        per_k_path,
        index=False,
    )
    print(f"Per-k summary saved: {per_k_path}")
    print()
    print(f"Finished in {(time.time() - t0) / 60:.1f} minutes.")
    return df, analysis_df, per_k_summary
if __name__ == "__main__":
    main()
