#!/usr/bin/env python3
"""Clean synthetic graph-geodesic k-sweep.

Purpose
-------
Test geodesic k-sensitivity on ordered synthetic helix and circle controls.
Helix controls are expected to remain negative; circle controls are expected
to remain positive except for possible finite-sample boundary misses.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Outputs
-------
Writes raw_results.csv, analysis_summary.csv, and transition_summary.csv under
results/k_sensitivity_transition/.
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
from scipy.stats import kendalltau
# ───────────────── CONFIG ─────────────────
OUT_DIR = Path("results") / "k_sensitivity_transition"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_SAMPLES = 1200
SUBSAMPLE = 600
N_NULLS = 200
SEEDS = [100, 200, 300, 400, 500, 600]
K_VALUES = [6, 8, 10, 12, 14, 16, 20]
GEODESIC_MAX_K = 50
GEODESIC_PCT = 95.0
ALPHA = 0.05
DELTA_MIN = 1e-3
LOBS_MULT = 5.0
EPS = 1e-12
EPS_SIG = 1e-6
MAD_SCALE = 1.4826
N_JOBS = -1
print("=" * 70)
print("Clean synthetic graph-geodesic k-sweep")
print("=" * 70)
print(f"N_SAMPLES={N_SAMPLES}")
print(f"SUBSAMPLE={SUBSAMPLE}")
print(f"N_NULLS={N_NULLS}")
print(f"SEEDS={SEEDS}")
print(f"K_VALUES={K_VALUES}")
print(f"GEODESIC_PCT={GEODESIC_PCT}")
print(f"Formal decision: pperm < {ALPHA}; separation_flag is diagnostic only")
print("zrob and TSR are descriptive only.")
print("=" * 70)
# ───────────────── ROBUST STATS ─────────────────
def _mad(arr):
    """Median absolute deviation."""
    arr = np.asarray(arr, dtype=float)
    return float(np.median(np.abs(arr - np.median(arr))))
# ───────────────── SUBSAMPLING ─────────────────
def stride_subsample(X, cap):
    """
    Stride subsampling for ordered synthetic trajectories.
    This preserves temporal/arc ordering and avoids artificial gaps that can
    allow kNN shortcuts in helix/circle controls.
    """
    if len(X) <= cap:
        return X
    idx = np.linspace(0, len(X) - 1, cap, dtype=int)
    return X[idx]
# ───────────────── GEODESIC PH ─────────────────
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
# ───────────────── NULL MODEL ─────────────────
# ───────────────── DATA GENERATORS ─────────────────
def generate_helix(seed, pitch, radius):
    """
    Helix negative control.
    Ground truth:
        contractible trajectory.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N_SAMPLES)
    theta = 6.0 * np.pi * t
    X = np.vstack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        pitch * t,
    ]).T
    X += 0.05 * rng.standard_normal((N_SAMPLES, 3))
    return X
def generate_circle(seed, radius, noise):
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
    X += noise * rng.standard_normal((N_SAMPLES, 3))
    return X
# ───────────────── NULL WORKER ─────────────────
def null_worker(null_traj, null_index, seed, k_init):
    """
    Worker for one null trajectory.
    Passing the trajectory directly avoids relying on list closure behavior
    inside joblib workers.
    """
    L, _ = max_H1_geodesic(
        null_traj,
        SUBSAMPLE,
        seed * 10000 + 100 + null_index,
        k_init,
    )
    return L
# ───────────────── CELL EVALUATION ─────────────────
def evaluate_cell(X, k_init, seed):
    """
    Evaluates one dataset/seed/k cell.
    """
    L_obs, k_used = max_H1_geodesic(
        X,
        SUBSAMPLE,
        seed + 1,
        k_init,
    )
    nulls = null_matched_steps(
        X,
        N_NULLS,
        seed + 2,
    )
    H_null = np.array(
        Parallel(n_jobs=N_JOBS)(
            delayed(null_worker)(
                nulls[i],
                i,
                seed,
                k_init,
            )
            for i in range(N_NULLS)
        )
    )
    stats = compute_stats(
        L_obs,
        H_null,
    )
    stats["k_init"] = k_init
    stats["k_used"] = k_used
    stats["seed"] = seed
    return stats
# ───────────────── RUN EXPERIMENT ─────────────────
def main():
    t0 = time.time()
    HELIX_VARIANTS = [
        ("helix_p10_r2", 10.0, 2.0),
        ("helix_p25_r3", 25.0, 3.0),
    ]
    CIRCLE_VARIANTS = [
        ("circle_r1_n005", 1.0, 0.05),
        ("circle_r2_n01", 2.0, 0.10),
    ]
    records = []
    # ---------------- Helix negative controls ----------------
    for name, pitch, radius in HELIX_VARIANTS:
        print()
        print("=" * 70)
        print(f"Helix: {name}")
        print("=" * 70)
        for k in K_VALUES:
            for seed in SEEDS:
                X = generate_helix(
                    seed=seed,
                    pitch=pitch,
                    radius=radius,
                )
                cell = evaluate_cell(
                    X,
                    k_init=k,
                    seed=seed,
                )
                record = {
                    "dataset": name,
                    "type": "helix",
                    "k": k,
                    "N_SAMPLES": N_SAMPLES,
                    "SUBSAMPLE": SUBSAMPLE,
                    "N_NULLS": N_NULLS,
                    "GEODESIC_MAX_K": GEODESIC_MAX_K,
                    "GEODESIC_PCT": GEODESIC_PCT,
                    "ALPHA": ALPHA,
                    "DELTA_MIN": DELTA_MIN,
                    "LOBS_MULT": LOBS_MULT,
                    **cell,
                }
                records.append(record)
                print(
                    f"k={k:>2} seed={seed:>3} "
                    f"formal={cell['formal_trigger']} "
                    f"TSR={cell['TSR']:.3f} "
                    f"p={cell['pperm']:.4f} "
                    f"Lobs={cell['Lobs']:.5f} "
                    f"k_used={cell['k_used']}"
                )
    # ---------------- Circle positive controls ----------------
    for name, radius, noise in CIRCLE_VARIANTS:
        print()
        print("=" * 70)
        print(f"Circle: {name}")
        print("=" * 70)
        for k in K_VALUES:
            for seed in SEEDS:
                X = generate_circle(
                    seed=seed,
                    radius=radius,
                    noise=noise,
                )
                cell = evaluate_cell(
                    X,
                    k_init=k,
                    seed=seed,
                )
                record = {
                    "dataset": name,
                    "type": "circle",
                    "k": k,
                    "N_SAMPLES": N_SAMPLES,
                    "SUBSAMPLE": SUBSAMPLE,
                    "N_NULLS": N_NULLS,
                    "GEODESIC_MAX_K": GEODESIC_MAX_K,
                    "GEODESIC_PCT": GEODESIC_PCT,
                    "ALPHA": ALPHA,
                    "DELTA_MIN": DELTA_MIN,
                    "LOBS_MULT": LOBS_MULT,
                    **cell,
                }
                records.append(record)
                print(
                    f"k={k:>2} seed={seed:>3} "
                    f"formal={cell['formal_trigger']} "
                    f"TSR={cell['TSR']:.3f} "
                    f"p={cell['pperm']:.4f} "
                    f"Lobs={cell['Lobs']:.5f} "
                    f"k_used={cell['k_used']}"
                )
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

    helix_bad = df[(df["type"] == "helix") & (df["formal_trigger"].astype(bool))]
    if len(helix_bad):
        raise RuntimeError(
            "k-sensitivity failed: helix negative control produced formal "
            "geodesic positives.\n"
            + helix_bad[["dataset", "seed", "k", "k_used", "Lobs", "pperm", "TSR"]].to_string(index=False)
        )

    # ───────────────── ANALYSIS ─────────────────
    analysis_rows = []
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset].copy()
        per_k_freq = sub.groupby("k")["formal_trigger"].mean()
        per_k_tsr = sub.groupby("k")["TSR"].mean()
        k_vals = sorted(per_k_freq.index)
        tsr_vals = [per_k_tsr[k] for k in k_vals]
        tau, p_tau = kendalltau(k_vals, tsr_vals)
        analysis_rows.append(dict(
            dataset=dataset,
            type=str(sub["type"].iloc[0]),
            kendall_tau_TSR=tau,
            tau_TSR_p=p_tau,
            n_formal_positive=int(sub["formal_trigger"].sum()),
            n_total=int(len(sub)),
            formal_fire_rate=float(sub["formal_trigger"].mean()),
            mean_TSR=float(sub["TSR"].mean()),
            max_TSR=float(sub["TSR"].max()),
            min_pperm=float(sub["pperm"].min()),
        ))
    analysis_df = pd.DataFrame(analysis_rows)
    analysis_path = OUT_DIR / "analysis_summary.csv"
    analysis_df.to_csv(
        analysis_path,
        index=False,
    )
    print()
    print("=" * 70)
    print(f"Analysis summary saved: {analysis_path}")
    print("=" * 70)
    print(analysis_df)
    # Optional transition summary across adjacent k values
    transition_rows = []
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset].copy()
        for seed in SEEDS:
            ss = sub[sub["seed"] == seed].sort_values("k")
            decisions = ss["formal_trigger"].astype(bool).to_numpy()
            ks = ss["k"].to_numpy()
            if len(decisions) < 2:
                continue
            flips = decisions[1:] != decisions[:-1]
            transition_rows.append(dict(
                dataset=dataset,
                type=str(ss["type"].iloc[0]),
                seed=seed,
                n_adjacent_transitions=int(len(flips)),
                n_flips=int(np.sum(flips)),
                flip_rate=float(np.mean(flips)),
                k_sequence=",".join(map(str, ks)),
                decision_sequence=",".join(map(str, decisions)),
            ))
    transition_df = pd.DataFrame(transition_rows)
    transition_path = OUT_DIR / "transition_summary.csv"
    transition_df.to_csv(
        transition_path,
        index=False,
    )
    print()
    print("=" * 70)
    print(f"Transition summary saved: {transition_path}")
    print("=" * 70)
    if len(transition_df) > 0:
        print(
            transition_df.groupby(["dataset", "type"])[
                ["n_flips", "flip_rate"]
            ].mean()
        )
    elapsed = (time.time() - t0) / 60.0
    print()
    print("=" * 70)
    print(f"Finished in {elapsed:.1f} minutes")
    print("=" * 70)
    return df, analysis_df, transition_df
if __name__ == "__main__":
    main()
