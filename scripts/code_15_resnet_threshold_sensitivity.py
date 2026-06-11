#!/usr/bin/env python3
"""Independent ResNet-18/SGD threshold-sensitivity audit.

Purpose
-------
Test filtration-threshold sensitivity for saved ResNet-18/SGD neural
trajectories using a RAM-safe standalone implementation.

Memory strategy
---------------
The trajectory is loaded with mmap_mode='r'. Observed and null distance matrices
are 200 x 200, nulls are processed sequentially, and chunked column-wise
accumulation is used for the full parameter vectors.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Outputs
-------
Writes threshold_sensitivity_results.csv and threshold_sensitivity_summary.csv
under exp4_results/threshold_sensitivity_resnet18_sgd/ by default.
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
import os
# Keep BLAS/OpenMP from over-threading on 16 GB RAM systems.
for _env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env, "1")
from pathlib import Path
import gc
import argparse
import numpy as np
import pandas as pd
from ripser import ripser
# ============================================================
# CONFIG
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="ResNet-18/SGD filtration-threshold sensitivity audit")
    p.add_argument("--base-dir", type=Path, default=Path("exp4_results"),
                   help="Base experiment directory containing trajectories/ and output folders.")
    p.add_argument("--traj-dir", type=Path, default=None,
                   help="Trajectory directory. Defaults to <base-dir>/trajectories.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory. Defaults to <base-dir>/threshold_sensitivity_resnet18_sgd.")
    p.add_argument("--arch", default="resnet18_cifar10")
    p.add_argument("--optimizer", default="sgd")
    p.add_argument("--seeds", default="20042,21042,22042")
    p.add_argument("--thresholds", default="90,95,99")
    p.add_argument("--n-null", type=int, default=50)
    p.add_argument("--subsample", type=int, default=200)
    p.add_argument("--geodesic-k", type=int, default=12)
    p.add_argument("--geodesic-max-k", type=int, default=50)
    p.add_argument("--chunk-cols", type=int, default=20_000)
    return p.parse_known_args()[0]

_args = parse_args()
BASE_DIR = _args.base_dir
TRAJ_DIR = _args.traj_dir if _args.traj_dir is not None else BASE_DIR / "trajectories"
OUT_DIR = _args.out_dir if _args.out_dir is not None else BASE_DIR / "threshold_sensitivity_resnet18_sgd"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARCH = _args.arch
OPT = _args.optimizer
SEEDS = [int(s) for s in str(_args.seeds).split(",") if s]
THRESHOLDS = [float(s) for s in str(_args.thresholds).split(",") if s]
SUBSAMPLE = int(_args.subsample)
N_NULL = int(_args.n_null)
GEODESIC_K = int(_args.geodesic_k)
GEODESIC_MAX_K = int(_args.geodesic_max_k)
ALPHA = 0.05
EPS = 1e-12
EPS_SIG = 1e-6
MAD_SCALE = 1.4826
CHUNK_COLS = int(_args.chunk_cols)
DELTA_MIN = 1e-3
LOBS_MULT = 5.0
# ============================================================
# BASIC STATS
# ============================================================
def _mad(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    return float(np.median(np.abs(arr - np.median(arr))))
# ============================================================
# DATA LOADING
# ============================================================
def load_trajectory(seed: int) -> np.memmap:
    path = TRAJ_DIR / f"{ARCH}__{OPT}__seed{seed}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")
    return np.load(path, mmap_mode="r")
# ============================================================
# CHUNKED DISTANCE COMPUTATION
# ============================================================
def gram_dist_chunked(X_mmap: np.ndarray, chunk_cols: int = CHUNK_COLS) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix for X_mmap by streaming columns.
    X_mmap has shape (n, d), typically (200, 11_173_962).
    """
    n, d = X_mmap.shape
    sq = np.zeros(n, dtype=np.float64)
    gram = np.zeros((n, n), dtype=np.float64)
    for start in range(0, d, chunk_cols):
        end = min(start + chunk_cols, d)
        Xc = X_mmap[:, start:end].astype(np.float32, copy=False)
        Xc64 = Xc.astype(np.float64, copy=False)
        sq += np.einsum("ij,ij->i", Xc64, Xc64)
        gram += Xc64 @ Xc64.T
        del Xc, Xc64
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0))
    np.fill_diagonal(D, 0.0)
    return D
def null_dist_chunked(X_mmap: np.ndarray, idx: int, master_seed: int, chunk_cols: int = CHUNK_COLS) -> np.ndarray:
    """
    Build a matched-step-permutation null trajectory distance matrix without
    materialising the full null trajectory in RAM.
    """
    n, d = X_mmap.shape
    rng = np.random.default_rng([master_seed, idx])
    perm = rng.permutation(n - 1)
    sq = np.zeros(n, dtype=np.float64)
    gram = np.zeros((n, n), dtype=np.float64)
    # Reusable buffers for each chunk
    steps_buf = np.empty((n - 1, chunk_cols), dtype=np.float32)
    null_buf = np.empty((n, chunk_cols), dtype=np.float32)
    for start in range(0, d, chunk_cols):
        end = min(start + chunk_cols, d)
        cols = end - start
        Xc = X_mmap[:, start:end].astype(np.float32, copy=False)
        sc = steps_buf[:, :cols]
        nc = null_buf[:, :cols]
        # steps = X[t+1] - X[t]
        np.subtract(Xc[1:], Xc[:-1], out=sc)
        # permute steps
        sc[:] = sc[perm]
        # reconstruct null trajectory
        nc[0] = Xc[0]
        np.cumsum(sc, axis=0, out=nc[1:])
        nc[1:] += Xc[0]
        nc64 = nc.astype(np.float64, copy=False)
        sq += np.einsum("ij,ij->i", nc64, nc64)
        gram += nc64 @ nc64.T
        del Xc, nc64
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0))
    np.fill_diagonal(D, 0.0)
    return D
# ============================================================
# H1 FROM A DISTANCE MATRIX
# ============================================================
def geodesic_h1_from_dist(
    D_full: np.ndarray,
    k_init: int,
    percentile: float,
) -> float:
    """Shared graph-geodesic max H1 lifetime from an ambient distance matrix."""
    if len(D_full) < 4:
        return 0.0
    return geodesic_h1_lifetime(
        D_full,
        input_distance_matrix=True,
        k0=k_init,
        kmax=GEODESIC_MAX_K,
        pct=percentile,
        eps=EPS,
        sentinel_fill=False,
    )
# ============================================================
# AUDIT
# ============================================================
def run_threshold_audit() -> pd.DataFrame:
    rows = []
    print("=" * 80)
    print(f"Threshold sensitivity audit: {ARCH} / {OPT}")
    print(f"Seeds: {SEEDS}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"SUBSAMPLE = {SUBSAMPLE} (for this neural audit, T = 200 so all points are used)")
    print(f"N_NULL = {N_NULL}  (default 50 for chunked ResNet sensitivity; override with --n-null)")
    print("=" * 80)
    for seed in SEEDS:
        print(f"\nLoading trajectory for seed {seed} ...")
        X = load_trajectory(seed)  # memmap, not loaded into RAM
        n, d = X.shape
        print(f"  trajectory shape = {n} x {d}")
        if n != SUBSAMPLE:
            raise ValueError(
                f"This audit expects n == SUBSAMPLE because the chunked path "
                f"does not currently subsample rows. Got n={n}, SUBSAMPLE={SUBSAMPLE}."
            )
        # Observed distance matrix once per seed
        print("  computing observed distance matrix ...")
        D_obs = gram_dist_chunked(X)
        obs_by_pct = {}
        for pct in THRESHOLDS:
            obs_by_pct[pct] = geodesic_h1_from_dist(D_obs, GEODESIC_K, pct)
            print(f"    observed H1 at {pct}% = {obs_by_pct[pct]:.6e}")
        # Null lifetimes for each threshold
        null_by_pct = {pct: [] for pct in THRESHOLDS}
        for i in range(N_NULL):
            print(f"  null {i+1:3d}/{N_NULL} ...", end="\r")
            D_null = null_dist_chunked(X, idx=i, master_seed=seed + 100)
            for pct in THRESHOLDS:
                l_null = geodesic_h1_from_dist(D_null, GEODESIC_K, pct)
                null_by_pct[pct].append(l_null)
            del D_null
            gc.collect()
        print(" " * 80, end="\r")
        # Compute stats and store rows. These are sensitivity diagnostics:
        # off-default threshold triggers must not be promoted to manuscript-level
        # recurrence claims.
        for pct in THRESHOLDS:
            stats = compute_stats(obs_by_pct[pct], np.array(null_by_pct[pct], dtype=float))
            decision = (
                "sensitivity_nominal_trigger"
                if stats["formal_trigger"]
                else "sensitivity_non_trigger"
            )
            rows.append({
                "architecture": ARCH,
                "optimizer": OPT,
                "seed": seed,
                "threshold_pct": pct,
                "GEODESIC_K": GEODESIC_K,
                "GEODESIC_MAX_K": GEODESIC_MAX_K,
                "N_NULL": N_NULL,
                "SUBSAMPLE": SUBSAMPLE,
                "CHUNK_COLS": CHUNK_COLS,
                "ALPHA": ALPHA,
                "decision": decision,
                "is_default_threshold": bool(abs(float(pct) - 95.0) < 1e-12),
                **stats,
            })
            print(
                f"  seed={seed} pct={pct} "
                f"Lobs={stats['Lobs']:.3e} "
                f"pperm={stats['pperm']:.4f} "
                f"zrob={stats['zrob']:+.2f} "
                f"zero_frac={stats['zero_frac']:.2f} "
                f"=> {decision}"
            )
        # free observed matrix
        del D_obs, X
        gc.collect()
    df = pd.DataFrame(rows)
    return df
def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pct in sorted(df["threshold_pct"].unique()):
        sub = df[df["threshold_pct"] == pct]
        out.append({
            "threshold_pct": pct,
            "n_rows": int(len(sub)),
            "n_positive": int(sub["formal_trigger"].sum()),
            "positive_rate": float(sub["formal_trigger"].mean()),
            "mean_pperm": float(sub["pperm"].mean()),
            "mean_zrob": float(sub["zrob"].mean()),
            "mean_Lobs": float(sub["Lobs"].mean()),
        })
    return pd.DataFrame(out)
def main() -> None:
    df = run_threshold_audit()
    csv_path = OUT_DIR / "threshold_sensitivity_results.csv"
    tex_path = OUT_DIR / "threshold_sensitivity_results.tex"
    summary_path = OUT_DIR / "threshold_sensitivity_summary.csv"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, escape=False, float_format="%.6g")
    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    print("\n=== Summary by threshold ===")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {tex_path}")
    print(f"  {summary_path}")
if __name__ == "__main__":
    main()
