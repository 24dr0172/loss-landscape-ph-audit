#!/usr/bin/env python3
"""Empirical block Type-I calibration for MLP2-scale random walks.

Purpose
-------
Estimate how often the block-permutation graph-geodesic diagnostic triggers on
non-recurrent empirical random walks built from real MLP2/MNIST step vectors.

Method
------
The script reads existing MLP2 trajectory arrays, extracts step vectors,
precomputes their step-step Gram matrix, and generates calibration random walks
by sampling empirical step vectors with replacement. Pairwise checkpoint
distances are then computed from the Gram matrix without constructing every
high-dimensional null trajectory explicitly.

Interpretation
--------------
The calibration is empirical and model-specific. It does not provide a theorem
for b>1 block surrogates.

Outputs
-------
Writes block_type1_full.csv, block_type1_summary.csv, run_metadata.json, and
optional null-lifetime arrays under the configured output directory.
"""
from __future__ import annotations

# Shared audit primitives.
from pathlib import Path as _AuditPath
import sys as _audit_sys

_audit_sys.path.insert(0, str(_AuditPath(__file__).resolve().parents[1]))

from audit_common import (
    compute_stats,
    geodesic_h1_lifetime,
    sha256_file,
    block_permutation_indices,
)
# Keep BLAS from oversubscribing when users run multiple scripts.
import os
for _env in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_env, "1")
import argparse
import csv
import glob
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import numpy as np
import pandas as pd
import warnings
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x
# =============================================================================
# Configuration dataclass
# =============================================================================
@dataclass
class Config:
    trajectory_glob: str
    out_dir: str
    n_calib: int
    n_nulls: int
    blocks: List[int]
    n_steps: int | None
    geodesic_k: int
    geodesic_max_k: int
    geodesic_pct: float
    alpha: float
    delta_min: float
    lobs_mult: float
    eps: float
    eps_sig: float
    mad_scale: float
    seed: int
    chunk_cols: int
    sample_with_replacement: bool
    save_nulls: bool
    resume: bool
    max_source_files: int | None
    dtype: str
# =============================================================================
# Utility functions
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
def parse_blocks(values: Sequence[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        if "," in v:
            out.extend(int(x) for x in v.split(",") if x.strip())
        else:
            out.append(int(v))
    return sorted(set(out))
def mad(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float(np.median(np.abs(a - np.median(a))))
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)
# =============================================================================
# Loading empirical MLP2 step pool
# =============================================================================
def load_step_pool(
    trajectory_paths: Sequence[Path],
    max_source_files: int | None = None,
    dtype: str = "float32",
) -> tuple[np.ndarray, list[dict]]:
    """Load trajectories and concatenate their step vectors.
    Each .npy file is expected to have shape (n_checkpoints, d). Steps are
    np.diff(X, axis=0). The returned step pool has shape (n_pool_steps, d).
    """
    if max_source_files is not None:
        trajectory_paths = list(trajectory_paths)[:max_source_files]
    if not trajectory_paths:
        raise FileNotFoundError(
            "No trajectory files matched the supplied --trajectory_glob. "
            "Example: --trajectory_glob 'exp4_results/trajectories/*mlp2_mnist*sgd*.npy'"
        )
    steps_list: list[np.ndarray] = []
    manifest: list[dict] = []
    expected_d: int | None = None
    print("Loading source trajectories and extracting steps:")
    for p in trajectory_paths:
        print(f"  - {p}")
        X = np.load(p, mmap_mode="r")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array in {p}, got shape {X.shape}")
        n, d = X.shape
        if expected_d is None:
            expected_d = d
        elif d != expected_d:
            raise ValueError(f"Dimension mismatch: {p} has d={d}, expected d={expected_d}")
        # Load only the diff into memory. For MLP2 this is manageable.
        S = np.diff(np.asarray(X, dtype=dtype), axis=0)
        steps_list.append(S)
        manifest.append(
            dict(
                file=str(p),
                shape=list(X.shape),
                dtype=str(X.dtype),
                n_steps=int(S.shape[0]),
                d=int(d),
                size_bytes=int(p.stat().st_size),
                sha256=sha256_file(p),
            )
        )
    S_pool = np.concatenate(steps_list, axis=0)
    print(f"Step pool shape: {S_pool.shape}  dtype={S_pool.dtype}")
    return S_pool, manifest
def compute_step_pool_gram_chunked(S_pool: np.ndarray, chunk_cols: int = 50_000) -> np.ndarray:
    """Compute G = S_pool @ S_pool.T with float64 accumulation.
    S_pool is kept in float32 to save memory; each column chunk is converted to
    float64 during multiplication.
    """
    n_pool, d = S_pool.shape
    G = np.zeros((n_pool, n_pool), dtype=np.float64)
    print(f"Computing empirical step-pool Gram matrix: n_pool={n_pool}, d={d}")
    for start in tqdm(range(0, d, chunk_cols), desc="Gram chunks"):
        end = min(start + chunk_cols, d)
        Xc = S_pool[:, start:end].astype(np.float64, copy=False)
        G += Xc @ Xc.T
        del Xc
    # Numerical symmetrization.
    G = 0.5 * (G + G.T)
    return G
# =============================================================================
# Distance construction from small step Gram matrices
# =============================================================================
def cumulative_distance_from_step_gram(G_steps: np.ndarray, order: np.ndarray | None = None) -> np.ndarray:
    """Return pairwise Euclidean distances between cumulative checkpoints.
    Parameters
    ----------
    G_steps:
        Step-step Gram matrix for the selected steps, shape (m, m).
    order:
        Optional permutation/order of step indices. If None, identity order is used.
    Returns
    -------
    D:
        Distance matrix between checkpoints X_0, ..., X_m, shape (m+1, m+1).
    """
    if order is not None:
        H = G_steps[np.ix_(order, order)]
    else:
        H = G_steps
    m = H.shape[0]
    # 2D prefix sum C where C[i,j] = sum_{a<i,b<j} H[a,b].
    C = np.zeros((m + 1, m + 1), dtype=np.float64)
    C[1:, 1:] = np.cumsum(np.cumsum(H, axis=0), axis=1)
    D2 = np.zeros((m + 1, m + 1), dtype=np.float64)
    # For i<j, ||X_j-X_i||^2 = sum_{a,b=i}^{j-1} H[a,b]
    for i in range(m + 1):
        # vectorized over j>i
        js = np.arange(i + 1, m + 1)
        if len(js) == 0:
            continue
        vals = C[js, js] - C[i, js] - C[js, i] + C[i, i]
        vals = np.maximum(vals, 0.0)
        D2[i, js] = vals
        D2[js, i] = vals
    return np.sqrt(D2)
# =============================================================================
# Graph-geodesic PH core
# =============================================================================
def max_h1_geodesic_from_euclidean_D(
    D_full: np.ndarray,
    geodesic_k: int = 12,
    geodesic_max_k: int = 50,
    geodesic_pct: float = 95.0,
    eps: float = 1e-12,
) -> float:
    """Graph-geodesic max H1 lifetime via audit_common.

    The final release policy is fail-loud: disconnected kNN-geodesic
    graphs raise RuntimeError instead of using sentinel-filled distances.
    """
    if D_full.shape[0] < 4:
        return 0.0

    return geodesic_h1_lifetime(
        D_full,
        input_distance_matrix=True,
        k0=geodesic_k,
        kmax=geodesic_max_k,
        pct=geodesic_pct,
        eps=eps,
        sentinel_fill=False,
    )
# =============================================================================
# Null generation and statistics
# =============================================================================
# =============================================================================
# Calibration experiment
# =============================================================================
def existing_completed_keys(full_csv: Path) -> set[tuple[int, int]]:
    if not full_csv.exists():
        return set()
    try:
        df = pd.read_csv(full_csv)
    except Exception:
        return set()
    if "calib_id" not in df.columns or "block_size" not in df.columns:
        return set()
    return set((int(r.calib_id), int(r.block_size)) for r in df.itertuples())
def append_row_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
def run_calibration(config: Config) -> None:
    out_dir = Path(config.out_dir)
    ensure_dir(out_dir)
    null_dir = out_dir / "null_lifetimes"
    if config.save_nulls:
        ensure_dir(null_dir)
    full_csv = out_dir / "block_type1_full.csv"
    summary_csv = out_dir / "block_type1_summary.csv"
    metadata_json = out_dir / "run_metadata.json"
    trajectory_paths_all = sorted(Path(p) for p in glob.glob(config.trajectory_glob))
    # Ignore auxiliary metric arrays saved next to trajectories, e.g.
    # mlp2_mnist__adam__seed1242__metrics.npy.  These are not parameter
    # trajectories and typically have shape (T, 2), causing dimension mismatch.
    trajectory_paths = [p for p in trajectory_paths_all if "__metrics" not in p.stem]
    skipped_aux = [p for p in trajectory_paths_all if p not in trajectory_paths]
    if skipped_aux:
        print(f"Skipping {len(skipped_aux)} auxiliary/non-trajectory files:")
        for p in skipped_aux[:20]:
            print(f"  - {p}")
        if len(skipped_aux) > 20:
            print(f"  ... and {len(skipped_aux) - 20} more")
    if not trajectory_paths:
        raise FileNotFoundError(
            f"No trajectory files matched trajectory_glob={config.trajectory_glob!r} "
            "after excluding auxiliary __metrics files."
        )
    S_pool, source_manifest = load_step_pool(
        trajectory_paths,
        max_source_files=config.max_source_files,
        dtype=config.dtype,
    )
    G_pool = compute_step_pool_gram_chunked(S_pool, chunk_cols=config.chunk_cols)
    n_pool, d = S_pool.shape
    # Default: match the first source trajectory's number of steps.
    if config.n_steps is None:
        # Most source files have same n_steps; use minimum to be conservative.
        config.n_steps = min(int(m["n_steps"]) for m in source_manifest)
    if config.n_steps <= 1:
        raise ValueError("n_steps must be > 1")

    # Formal no-sentinel policy:
    # calibration walks have n_steps increments and n_steps+1 checkpoints.
    # If kmax=50 disconnects, allow adaptive k to reach full connectivity.
    config.geodesic_max_k = max(int(config.geodesic_max_k), int(config.n_steps))
    if (not config.sample_with_replacement) and config.n_steps > n_pool:
        raise ValueError("Cannot sample without replacement: n_steps > n_pool")
    print("=" * 80)
    print("BLOCK TYPE-I CALIBRATION")
    print(f"Source files       : {len(source_manifest)}")
    print(f"Step pool          : n_pool={n_pool}, d={d}")
    print(f"Calibration walks  : {config.n_calib}")
    print(f"Steps per walk     : {config.n_steps}")
    print(f"Nulls per block    : {config.n_nulls}")
    print(f"Blocks             : {config.blocks}")
    print(f"Geodesic           : k={config.geodesic_k}, max_k={config.geodesic_max_k}, pct={config.geodesic_pct}")
    print(f"Formal trigger     : pperm < {config.alpha}")
    print(f"Separation flag    : delta >= {config.delta_min}, Lobs >= {config.lobs_mult} * null_max")
    print("=" * 80)
    metadata = dict(
        config=asdict(config),
        source_manifest=source_manifest,
        step_pool_shape=[int(n_pool), int(d)],
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        python=sys.version,
        platform=platform.platform(),
        package_versions=dict(
            numpy=np.__version__,
            pandas=pd.__version__,
        ),
    )
    metadata_json.write_text(json.dumps(metadata, indent=2))
    completed = existing_completed_keys(full_csv) if config.resume else set()
    rng_master = np.random.default_rng(config.seed)
    # Pre-generate calibration step indices deterministically so resume is stable.
    calib_indices = []
    for calib_id in range(config.n_calib):
        rng = np.random.default_rng([config.seed, calib_id, 12345])
        if config.sample_with_replacement:
            idx = rng.integers(0, n_pool, size=config.n_steps, endpoint=False, dtype=np.int64)
        else:
            idx = rng.choice(n_pool, size=config.n_steps, replace=False).astype(np.int64)
        calib_indices.append(idx)
    t_all = time.time()
    for calib_id in tqdm(range(config.n_calib), desc="Calibration walks"):
        step_pool_indices = calib_indices[calib_id]
        # Small Gram matrix for this calibration walk's selected steps.
        G_obs = G_pool[np.ix_(step_pool_indices, step_pool_indices)]
        # Observed calibration walk in identity order.
        D_obs = cumulative_distance_from_step_gram(G_obs, order=None)
        Lobs = max_h1_geodesic_from_euclidean_D(
            D_obs,
            geodesic_k=config.geodesic_k,
            geodesic_max_k=config.geodesic_max_k,
            geodesic_pct=config.geodesic_pct,
            eps=config.eps,
        )
        for b in config.blocks:
            key = (calib_id, b)
            if key in completed:
                continue
            t0 = time.time()
            Lnull = np.zeros(config.n_nulls, dtype=np.float64)
            for j in range(config.n_nulls):
                rng_null = np.random.default_rng([config.seed, calib_id, b, j])
                perm = block_permutation_indices(config.n_steps, b, rng_null)
                D_null = cumulative_distance_from_step_gram(G_obs, order=perm)
                Lnull[j] = max_h1_geodesic_from_euclidean_D(
                    D_null,
                    geodesic_k=config.geodesic_k,
                    geodesic_max_k=config.geodesic_max_k,
                    geodesic_pct=config.geodesic_pct,
                    eps=config.eps,
                )
            stats = compute_stats(
                Lobs=Lobs,
                Lnull=Lnull,
                alpha=config.alpha,
                delta_min=config.delta_min,
                lobs_mult=config.lobs_mult,
                eps=config.eps,
                eps_sig=config.eps_sig,
                mad_scale=config.mad_scale,
            )
            if config.save_nulls:
                np.save(null_dir / f"calib_{calib_id:05d}_b{b}.npy", Lnull)
            row = dict(
                experiment="block_type1_calibration_mlp2_empirical_random_walk",
                calib_id=calib_id,
                block_size=b,
                n_steps=config.n_steps,
                n_checkpoints=config.n_steps + 1,
                d=d,
                n_pool_steps=n_pool,
                n_source_files=len(source_manifest),
                n_nulls=config.n_nulls,
                geodesic_k=config.geodesic_k,
                geodesic_max_k=config.geodesic_max_k,
                geodesic_pct=config.geodesic_pct,
                alpha=config.alpha,
                sample_with_replacement=config.sample_with_replacement,
                seed=config.seed,
                runtime_sec=round(time.time() - t0, 3),
            )
            row.update(stats)
            append_row_csv(full_csv, row)
    print(f"Finished calibration in {(time.time() - t_all) / 60:.2f} minutes")
    make_summary(full_csv, summary_csv, alpha=config.alpha)
    print(f"Full results   : {full_csv}")
    print(f"Summary results: {summary_csv}")
def make_summary(full_csv: Path, summary_csv: Path, alpha: float) -> None:
    df = pd.read_csv(full_csv)
    rows = []
    for b, g in df.groupby("block_size", sort=True):
        n = len(g)
        k_formal = int(g["formal_trigger"].astype(bool).sum())
        k_sep = int(g["separation_flag"].astype(bool).sum())
        k_nominal = int(g["nominal_trigger"].astype(bool).sum())
        lo, hi = wilson_ci(k_formal, n)
        rows.append(
            dict(
                block_size=int(b),
                n_calib_cells=int(n),
                pperm_trigger_count=k_formal,
                empirical_fpr=k_formal / n if n else float("nan"),
                empirical_fpr_wilson95_lo=lo,
                empirical_fpr_wilson95_hi=hi,
                separation_flag_count=k_sep,
                nominal_trigger_count=k_nominal,
                mean_Lobs=float(g["Lobs"].mean()),
                median_Lobs=float(g["Lobs"].median()),
                # audit_common returns the key as "null_med".
                mean_null_median=float(g["null_med"].mean()),
                median_null_median=float(g["null_med"].median()),
                mean_pperm=float(g["pperm"].mean()),
                min_pperm=float(g["pperm"].min()),
                alpha=alpha,
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(summary_csv, index=False)
    print("\nEmpirical Type-I / FPR summary")
    print(out.to_string(index=False))
# =============================================================================
# CLI
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Block Type-I calibration for MLP2-scale empirical random walks."
    )
    p.add_argument(
        "--trajectory_glob",
        type=str,
        required=True,
        help="Glob for source MLP2 .npy trajectories, e.g. 'exp4_results/trajectories/*mlp2_mnist*sgd*.npy'.",
    )
    p.add_argument("--out_dir", type=str, default="exp4_results/block_type1_calibration")
    p.add_argument("--n_calib", type=int, default=200, help="Number of calibration random-walk trajectories.")
    p.add_argument("--n_nulls", type=int, default=200, help="Null permutations per calibration trajectory/block size.")
    p.add_argument("--blocks", nargs="+", default=["1", "5", "10", "20"], help="Block sizes, e.g. --blocks 1 5 10 20 or --blocks 1,5,10,20")
    p.add_argument("--n_steps", type=int, default=None, help="Steps per calibration walk. Default: min steps among source trajectories.")
    p.add_argument("--geodesic_k", type=int, default=12)
    p.add_argument("--geodesic_max_k", type=int, default=199)
    p.add_argument("--geodesic_pct", type=float, default=95.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--delta_min", type=float, default=1e-3)
    p.add_argument("--lobs_mult", type=float, default=5.0)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--eps_sig", type=float, default=1e-6)
    p.add_argument("--mad_scale", type=float, default=1.4826)
    p.add_argument("--seed", type=int, default=20260506)
    p.add_argument("--chunk_cols", type=int, default=50_000, help="Column chunk size for step-pool Gram computation.")
    p.add_argument("--without_replacement", action="store_true", help="Sample calibration steps without replacement from the source step pool.")
    p.add_argument("--save_nulls", action="store_true", help="Save null lifetime arrays for each calibration/block cell.")
    p.add_argument("--no_resume", action="store_true", help="Do not skip rows already present in block_type1_full.csv.")
    p.add_argument("--max_source_files", type=int, default=None, help="Optional cap on number of source trajectories loaded.")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Dtype for loaded step pool.")
    return p
def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    config = Config(
        trajectory_glob=args.trajectory_glob,
        out_dir=args.out_dir,
        n_calib=args.n_calib,
        n_nulls=args.n_nulls,
        blocks=parse_blocks(args.blocks),
        n_steps=args.n_steps,
        geodesic_k=args.geodesic_k,
        geodesic_max_k=args.geodesic_max_k,
        geodesic_pct=args.geodesic_pct,
        alpha=args.alpha,
        delta_min=args.delta_min,
        lobs_mult=args.lobs_mult,
        eps=args.eps,
        eps_sig=args.eps_sig,
        mad_scale=args.mad_scale,
        seed=args.seed,
        chunk_cols=args.chunk_cols,
        sample_with_replacement=not args.without_replacement,
        save_nulls=args.save_nulls,
        resume=not args.no_resume,
        max_source_files=args.max_source_files,
        dtype=args.dtype,
    )
    run_calibration(config)
if __name__ == "__main__":
    main()
