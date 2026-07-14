#!/usr/bin/env python3
"""Neural trajectory audit pipeline.

Purpose
-------
Train or load the manuscript-facing neural optimization trajectories and run
full-space graph-geodesic persistent-homology audits, PCA diagnostics,
block-surrogate robustness diagnostics, k/filtration sensitivity sweeps,
training-history summaries, and appendix tables.

Execution modes
---------------
Standard run:
    python scripts/code_10_neural_trajectory_pipeline.py

Block-surrogate diagnostics only:
    SKIP_MAIN=1 RUN_BLOCK_ROBUSTNESS=1 python scripts/code_10_neural_trajectory_pipeline.py

Neural k/filtration sensitivity only:
    SKIP_MAIN=1 RUN_NEURAL_SENSITIVITY=1 python scripts/code_10_neural_trajectory_pipeline.py

Decision convention
-------------------
Only b=1 matched-step permutation rows are formal tests. Block sizes b>1 are
dependence-preserving diagnostics. Formal triggers are defined by pperm < ALPHA;
separation flags, TSR, zrob, and null-collapse fields are diagnostics only.

Runtime notes
-------------
ResNet-18 full-space analysis uses chunked memory-mapped distance computation.
Set ALLOW_STUB=1 only for explicit smoke tests; stub outputs are redirected to
exp4_results_smoke/ and are not manuscript-facing.
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
# ---------------------------------------------------------------------------
# OMP thread pinning — must happen before any numpy/scipy import.
# Prevents N_JOBS_PCA workers × N_BLAS_THREADS explosion under joblib.
# ---------------------------------------------------------------------------
import gc
import os as _os
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_env_var, "1")
import threading
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from ripser import ripser
from sklearn.decomposition import PCA
from tqdm import tqdm
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {_dev}"
          + (f"  ({torch.cuda.get_device_name(0)})" if _dev.type == "cuda" else ""))
    if _dev.type == "cuda":
        cap  = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA capability: {cap[0]}.{cap[1]}  VRAM: {vram:.1f} GB")
        torch.backends.cudnn.benchmark = True
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    TORCH_AVAILABLE = False
    _dev = None
    print("PyTorch not found — training will fail loudly unless ALLOW_STUB=1.")
# =============================================================================
# CONFIG
# =============================================================================
OUT_DIR  = Path("exp4_results")
TRAJ_DIR = OUT_DIR / "trajectories"
OUT_MAIN = OUT_DIR / "main_paper"
OUT_APPX = OUT_DIR / "appendix"
OUT_BLOCK = OUT_DIR / "block_robustness"
OUT_SENS  = OUT_DIR / "neural_sensitivity"
for _d in [OUT_DIR, TRAJ_DIR, OUT_MAIN, OUT_APPX, OUT_BLOCK, OUT_SENS]:
    _d.mkdir(parents=True, exist_ok=True)
EPOCHS             = 200
BATCH_SIZE         = 512
LR_SGD             = 0.01
LR_ADAM            = 1e-3
DATALOADER_WORKERS = 4
import platform as _platform
if _platform.system() == "Windows":
    DATALOADER_WORKERS = 0
    print("Windows detected — DataLoader workers set to 0")
MASTER_SEED   = 42
N_SEEDS       = 3
SEED_OFFSETS  = [0, 1000, 2000]
ARCHITECTURES = ["mlp2_mnist", "smallcnn_cifar10", "resnet18_cifar10"]
OPTIMIZERS    = ["sgd", "sgd_momentum", "adam", "adamw"]
# Null counts
N_NULLS_FULL     = 200
N_NULLS_PCA      = 200
N_NULLS_PCA_FAST = 100
# Block permutation robustness diagnostics
BLOCK_SIZES         = [1, 5, 10, 20]
N_NULLS_BLOCK       = 200
RUN_BLOCK_ROBUSTNESS = bool(int(_os.environ.get("RUN_BLOCK_ROBUSTNESS", "0")))
SKIP_MAIN            = bool(int(_os.environ.get("SKIP_MAIN", "0")))
ALLOW_STUB           = bool(int(_os.environ.get("ALLOW_STUB", "0")))  # explicit opt-in for smoke-test stubs only
if ALLOW_STUB:
    # Quarantine smoke-test stubs away from manuscript-facing exp4_results/.
    OUT_DIR  = Path("exp4_results_smoke")
    TRAJ_DIR = OUT_DIR / "trajectories"
    OUT_MAIN = OUT_DIR / "main_paper"
    OUT_APPX = OUT_DIR / "appendix"
    OUT_BLOCK = OUT_DIR / "block_robustness"
    OUT_SENS  = OUT_DIR / "neural_sensitivity"
    for _d in [OUT_DIR, TRAJ_DIR, OUT_MAIN, OUT_APPX, OUT_BLOCK, OUT_SENS]:
        _d.mkdir(parents=True, exist_ok=True)
    print("ALLOW_STUB=1: writing only to exp4_results_smoke/; not manuscript-facing outputs.")
# Neural k / filtration-threshold sensitivity diagnostics
K_SENS_VALUES          = [8, 10, 12, 16]
PCT_SENS_VALUES        = [90.0, 95.0, 99.0]
N_NULLS_SENS           = 200   # normal-d path (MLP2, SmallCNN)
N_NULLS_SENS_CHUNKED   = 50    # chunked path (ResNet-18): reduced to limit runtime
                                # 50 nulls → p-value resolution Δ=1/(N+1)≈0.02 — adequate
RUN_NEURAL_SENSITIVITY = bool(int(_os.environ.get("RUN_NEURAL_SENSITIVITY", "0")))
# PH settings
SUBSAMPLE_PH      = 200
TOTAL_PROJECTIONS = 100
ALPHA             = 0.05
ZROB_THRESH = 1.5
# Separation-flag diagnostic thresholds
DELTA_MIN  = 1e-3
LOBS_MULT  = 5.0
EPS        = 1e-12
EPS_SIG    = 1e-6
MAD_SCALE  = 1.4826
# Null collapse diagnostics
NULL_COLLAPSE_ZERO_FRAC = 0.7
NULL_COLLAPSE_LOBS_MIN  = 0.05
# Geodesic settings
GEODESIC_K     = 12
# Formal-run policy: geodesic statistics must be defined for every
# observed/null trajectory without sentinel fill. Some matched-step neural
# nulls are disconnected at kmax=50, so adaptive k may reach full connectivity
# on the T=200 checkpoint graph.
GEODESIC_MAX_K = SUBSAMPLE_PH - 1
GEODESIC_PCT   = 95.0
# PCA audit dimensions and epoch checkpoints
PCA_DIMS_FIXED = [3, 10, 50]
EPOCH_PREFIXES = [100, 150, 200]
assert EPOCH_PREFIXES[-1] == EPOCHS
# Parallelism
N_JOBS_FULL  = 1
N_JOBS_PCA   = 4
N_JOBS_BLOCK = 4   # block permutation parallelism (normal-d path)
D_THRESH     = 1000
# Chunked mmap path
LARGE_D_CHUNKED = 2_000_000
CHUNK_COLS      = 50_000
# Fast-path threshold
R_THRESH_FAST = 3.0
print("=" * 70)
print("Neural Network Trajectory Analysis")
print(f"  {len(ARCHITECTURES)} archs x {len(OPTIMIZERS)} opts x {N_SEEDS} seeds"
      f" = {len(ARCHITECTURES)*len(OPTIMIZERS)*N_SEEDS} runs")
print(f"  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}"
      f"  N_NULLS_FULL={N_NULLS_FULL}  N_NULLS_PCA={N_NULLS_PCA}")
print(f"  N_JOBS_PCA={N_JOBS_PCA}  GEODESIC_K={GEODESIC_K}")
print(f"  Formal trigger: pperm < {ALPHA}; separation flag is diagnostic only")
print(f"  RAM budget: N_JOBS_PCA={N_JOBS_PCA} × ~244MB/null ≈ ~1GB concurrent (SmallCNN)")
print(f"  Chunked path: CHUNK_COLS={CHUNK_COLS} → ~{CHUNK_COLS*200*4//1_000_000*2}MB peak/chunk (ResNet18)")
print(f"  RUN_BLOCK_ROBUSTNESS={RUN_BLOCK_ROBUSTNESS}  "
      f"BLOCK_SIZES={BLOCK_SIZES}  SKIP_MAIN={SKIP_MAIN}")
print(f"  RUN_NEURAL_SENSITIVITY={RUN_NEURAL_SENSITIVITY}  "
      f"K_SENS_VALUES={K_SENS_VALUES}  PCT_SENS_VALUES={PCT_SENS_VALUES}")
print(f"  N_NULLS_SENS={N_NULLS_SENS} (normal-d)  "
      f"N_NULLS_SENS_CHUNKED={N_NULLS_SENS_CHUNKED} (ResNet-18)")
print("=" * 70)
# =============================================================================
# STATISTICS
# =============================================================================
def _mad(a: np.ndarray) -> float:
    return float(np.median(np.abs(a - np.median(a))))
# =============================================================================
# KINEMATICS
# =============================================================================
def kinematic_ratio(X: np.ndarray) -> dict:
    steps    = np.diff(X, axis=0)
    path_len = float(np.sum(np.linalg.norm(steps, axis=1)))
    net_disp = float(np.linalg.norm(X[-1] - X[0]))
    return dict(path_length=path_len, net_displacement=net_disp,
                kinematic_R=path_len / max(net_disp, 1e-8))
# =============================================================================
# BLOCK-PERMUTATION INDEX UTILITY
# =============================================================================
# A block permutation of N step-increments {Δ_1, ..., Δ_N} works as follows:
#   1. Partition indices [0, N) into ceil(N / block_size) contiguous blocks.
#   2. Permute the block order uniformly at random.
#   3. Concatenate the permuted blocks (preserving within-block order).
# At block_size = 1 this reduces exactly to the matched-step permutation.
# Rows with block_size > 1 are dependence-preserving diagnostics.
# =============================================================================
def _block_permutation_indices(n_steps: int, block_size: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Block-step index permutation via audit_common.block_permutation_indices."""
    return block_permutation_indices(n_steps, block_size, rng)
# =============================================================================
# CHUNKED MMAP UTILITIES (large-d path)
# =============================================================================
def _kinematic_ratio_chunked(tp: Path) -> dict:
    X       = np.load(tp, mmap_mode='r')
    n, d    = X.shape
    step_sq = np.zeros(n - 1, dtype=np.float64)
    net_sq  = 0.0
    for start in range(0, d, CHUNK_COLS):
        end    = min(start + CHUNK_COLS, d)
        Xc     = X[:, start:end].astype(np.float64)
        diffs  = np.diff(Xc, axis=0)
        step_sq += np.einsum('ij,ij->i', diffs, diffs)
        ed      = Xc[-1] - Xc[0]
        net_sq += float(np.dot(ed, ed))
        del Xc, diffs, ed
    path_len = float(np.sum(np.sqrt(step_sq)))
    net_disp = float(np.sqrt(net_sq))
    return dict(path_length=path_len, net_displacement=net_disp,
                kinematic_R=path_len / max(net_disp, 1e-8))
def _distance_from_step_gram(step_gram: np.ndarray, perm: np.ndarray | None = None) -> np.ndarray:
    """Checkpoint distances from the Gram matrix of trajectory increments."""
    G = np.asarray(step_gram, dtype=np.float64)
    if perm is not None:
        p = np.asarray(perm, dtype=int)
        G = G[np.ix_(p, p)]
    prefix = np.pad(np.cumsum(np.cumsum(G, axis=0), axis=1), ((1, 0), (1, 0)))
    diag = np.diag(prefix)
    D2 = diag[:, None] + diag[None, :] - prefix - prefix.T
    D2 = 0.5 * (D2 + D2.T)
    D = np.sqrt(np.maximum(D2, 0.0))
    np.fill_diagonal(D, 0.0)
    return D


def _step_gram_from_array(X: np.ndarray) -> np.ndarray:
    """Float64 step Gram; avoids raw-parameter norm cancellation."""
    X64 = np.asarray(X, dtype=np.float64)
    steps = np.diff(X64, axis=0)
    G = steps @ steps.T
    return 0.5 * (G + G.T)


def _step_gram_chunked(X_mmap) -> np.ndarray:
    """Float64 increment Gram streamed over parameter columns."""
    n, d = X_mmap.shape
    G = np.zeros((n - 1, n - 1), dtype=np.float64)
    for start in range(0, d, CHUNK_COLS):
        end = min(start + CHUNK_COLS, d)
        Xc = np.asarray(X_mmap[:, start:end], dtype=np.float64)
        steps = np.diff(Xc, axis=0)
        G += steps @ steps.T
        del Xc, steps
    return 0.5 * (G + G.T)


def _gram_chunked(X_mmap, start_row: int = 0, n_rows: int | None = None) -> np.ndarray:
    Xview = X_mmap[start_row:] if n_rows is None else X_mmap[start_row:start_row + n_rows]
    return _distance_from_step_gram(_step_gram_chunked(Xview))


def _null_gram_chunked(X_mmap, idx: int, master: int, block_size: int = 1) -> np.ndarray:
    G = _step_gram_chunked(X_mmap)
    rng = np.random.default_rng([master, idx, block_size])
    perm = _block_permutation_indices(G.shape[0], block_size, rng)
    return _distance_from_step_gram(G, perm)


def _ph_from_dist(D: np.ndarray,
                  geodesic_k: int | None = None,
                  geodesic_pct: float | None = None) -> tuple[float, float, int]:
    """Return Euclidean H1, geodesic H1, and realized geodesic k."""
    pct = GEODESIC_PCT if geodesic_pct is None else float(geodesic_pct)
    complete = geodesic_pct is None
    euc = h1_lifetime_from_distance_matrix(
        D,
        pct=pct,
        eps=EPS,
        rescale=False,
        complete_filtration=complete,
    )
    geo, k_used = _geodesic_from_dist(
        D,
        geodesic_k=geodesic_k,
        geodesic_pct=geodesic_pct,
        return_k=True,
        complete_filtration=complete,
    )
    return euc, geo, int(k_used)


def _add_k_diagnostics(stats: dict, k_obs: int, k_null: np.ndarray, n_points: int) -> dict:
    k_null = np.asarray(k_null, dtype=float)
    stats = dict(stats)
    stats.update({
        "k_used_obs": int(k_obs),
        "null_k_min": float(np.min(k_null)),
        "null_k_median": float(np.median(k_null)),
        "null_k_max": float(np.max(k_null)),
        "null_k_near_complete_frac": float(np.mean(k_null >= 0.9 * max(n_points - 1, 1))),
    })
    return stats


def ambient_tests_chunked(tp: Path, seed: int, n_nulls: int,
                          block_size: int = 1,
                          geodesic_k: int | None = None,
                          geodesic_pct: float | None = None) -> tuple[dict, dict]:
    """Full-space audit from one streamed increment Gram matrix."""
    X_mmap = np.load(tp, mmap_mode="r")
    n = len(X_mmap)
    ns = seed + 100
    G = _step_gram_chunked(X_mmap)
    D_obs = _distance_from_step_gram(G)
    Lo_euc, Lo_geo, k_obs = _ph_from_dist(D_obs, geodesic_k, geodesic_pct)
    del D_obs
    H_euc = np.zeros(n_nulls, dtype=float)
    H_geo = np.zeros(n_nulls, dtype=float)
    K_geo = np.zeros(n_nulls, dtype=int)
    for i in range(n_nulls):
        rng = np.random.default_rng([ns, i, block_size])
        perm = _block_permutation_indices(n - 1, block_size, rng)
        D_null = _distance_from_step_gram(G, perm)
        H_euc[i], H_geo[i], K_geo[i] = _ph_from_dist(
            D_null, geodesic_k, geodesic_pct
        )
        del D_null
    s_euc = compute_stats(Lo_euc, H_euc)
    s_geo = _add_k_diagnostics(compute_stats(Lo_geo, H_geo), k_obs, K_geo, n)
    return s_euc, s_geo

def _pca_gram_chunked(tp: Path) -> tuple[np.ndarray, np.ndarray]:
    X_mmap = np.load(tp, mmap_mode="r")
    n, d = X_mmap.shape
    mean = np.zeros(d, dtype=np.float64)
    for start in range(0, d, CHUNK_COLS):
        end = min(start + CHUNK_COLS, d)
        mean[start:end] = np.asarray(X_mmap[:, start:end], dtype=np.float64).mean(axis=0)
    gram = np.zeros((n, n), dtype=np.float64)
    for start in range(0, d, CHUNK_COLS):
        end = min(start + CHUNK_COLS, d)
        Xc = np.asarray(X_mmap[:, start:end], dtype=np.float64) - mean[start:end]
        gram += Xc @ Xc.T
        del Xc
    return 0.5 * (gram + gram.T), mean

def pca_project_chunked(tp: Path, d_out: int) -> tuple[np.ndarray, float]:
    gram, _ = _pca_gram_chunked(tp)
    eigvals, U = np.linalg.eigh(gram)
    eigvals = eigvals[::-1]; U = U[:, ::-1]
    eigvals = np.maximum(eigvals, 0.0)
    var_ratio = eigvals[:d_out] / (eigvals.sum() + EPS)
    Xd = (U[:, :d_out] * np.sqrt(eigvals[:d_out])).astype(np.float32)
    return Xd, float(var_ratio.sum())
def pca_profile_chunked(tp: Path) -> dict:
    gram, _ = _pca_gram_chunked(tp)
    X_mmap  = np.load(tp, mmap_mode='r')
    n       = X_mmap.shape[0]
    eigvals, _ = np.linalg.eigh(gram)
    eigvals    = eigvals[::-1]
    eigvals    = np.maximum(eigvals, 0.0)
    cum        = np.cumsum(eigvals) / (eigvals.sum() + EPS)
    maxd       = n - 1
    d95 = min(int(np.searchsorted(cum, 0.95)) + 1, maxd)
    d99 = min(int(np.searchsorted(cum, 0.99)) + 1, maxd)
    return dict(d_95=d95, d_99=d99, max_d=maxd, cum=cum)
# =============================================================================
# DISTANCE COMPUTATION
# =============================================================================
def _gram_dist(X: np.ndarray) -> np.ndarray:
    """Centered float64 Gram distance for low-dimensional/PCA diagnostics."""
    X64 = np.asarray(X, dtype=np.float64)
    X64 = X64 - X64[:1]
    sq = np.einsum("ij,ij->i", X64, X64)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X64 @ X64.T)
    D2 = 0.5 * (D2 + D2.T)
    D = np.sqrt(np.maximum(D2, 0.0))
    np.fill_diagonal(D, 0.0)
    return D
# =============================================================================
# GPU ACCELERATION
# =============================================================================
GPU_D_THRESH = 200_000
_CUDA_OK = TORCH_AVAILABLE and _dev is not None and _dev.type == "cuda"
def _gram_dist_gpu(X: np.ndarray) -> np.ndarray:
    # Distance correctness takes priority; full-space neural tests use the
    # increment-Gram route, while PCA/projection diagnostics use this stable
    # centered float64 routine.
    return _gram_dist(X)
def _gpu_null_dist_loop(X: np.ndarray, n_nulls: int,
                        seed: int,
                        block_size: int = 1) -> list[np.ndarray] | None:
    """Compatibility wrapper using the stable increment-Gram construction."""
    G = _step_gram_from_array(X)
    ns = seed + 100
    dists = []
    for i in range(n_nulls):
        rng = np.random.default_rng([ns, i, block_size])
        perm = _block_permutation_indices(G.shape[0], block_size, rng)
        dists.append(_distance_from_step_gram(G, perm))
    return dists

def _gpu_batch_project(X: np.ndarray, ps: int,
                       ph_seed_base: int) -> list[np.ndarray]:
    n, d = X.shape
    cols = []
    for j in range(TOTAL_PROJECTIONS):
        P = np.linalg.qr(
            np.random.default_rng([ps, j]).standard_normal((d, 2))
        )[0][:, :2]
        cols.append(P)
    P_stacked = np.concatenate(cols, axis=1)
    if _CUDA_OK and d > GPU_D_THRESH:
        X_t  = torch.from_numpy(X.astype(np.float32)).to(_dev)
        P_t  = torch.from_numpy(P_stacked.astype(np.float32)).to(_dev)
        proj = torch.mm(X_t, P_t).cpu().numpy()
    else:
        proj = (X.astype(np.float32) @ P_stacked.astype(np.float32))
    return [proj[:, 2*j:2*j+2] for j in range(TOTAL_PROJECTIONS)]
# =============================================================================
# PH CORE
# =============================================================================
def _subsample(X: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    return X[rng.choice(len(X), cap, replace=False)] if len(X) > cap else X
def _geodesic_from_dist(D_full: np.ndarray,
                        geodesic_k: int | None = None,
                        geodesic_pct: float | None = None,
                        return_k: bool = False,
                        complete_filtration: bool = False):
    """Shared graph-geodesic H1, optionally returning realized k."""
    if len(D_full) < 4:
        return (0.0, 0) if return_k else 0.0
    k0 = GEODESIC_K if geodesic_k is None else int(geodesic_k)
    pct = GEODESIC_PCT if geodesic_pct is None else float(geodesic_pct)
    return geodesic_h1_lifetime(
        D_full,
        input_distance_matrix=True,
        k0=k0,
        kmax=GEODESIC_MAX_K,
        pct=pct,
        eps=EPS,
        sentinel_fill=False,
        return_k=return_k,
        complete_filtration=complete_filtration,
    )
def max_H1_euclidean(X: np.ndarray, cap: int, rng: np.random.Generator,
                     geodesic_pct: float | None = None) -> float:
    X = _subsample(X, cap, rng)
    pct = GEODESIC_PCT if geodesic_pct is None else float(geodesic_pct)
    D = _gram_dist(X)
    return h1_lifetime_from_distance_matrix(
        D,
        pct=pct,
        eps=EPS,
        rescale=False,
        complete_filtration=(geodesic_pct is None),
    )
def max_H1_geodesic(X: np.ndarray, cap: int, rng: np.random.Generator,
                    geodesic_k: int | None = None,
                    geodesic_pct: float | None = None) -> float:
    """Optional geodesic_k and geodesic_pct for sensitivity sweeps."""
    X = _subsample(X, cap, rng)
    D_full = _gram_dist_gpu(X)
    return _geodesic_from_dist(
        D_full,
        geodesic_k=geodesic_k,
        geodesic_pct=geodesic_pct,
        complete_filtration=(geodesic_pct is None),
    )
# =============================================================================
# NULL MODEL
# =============================================================================
def _make_null(X: np.ndarray, idx: int, master: int,
               block_size: int = 1) -> np.ndarray:
    """Matched/block-step null via audit_common.block_null."""
    rng = np.random.default_rng([master, idx, block_size])
    return block_null(X, block_size, rng)
# =============================================================================
# AMBIENT TESTS  (Euclidean + Geodesic)
# =============================================================================
def ambient_tests(X: np.ndarray, seed: int,
                  n_nulls: int, n_jobs: int,
                  block_size: int = 1,
                  geodesic_k: int | None = None,
                  geodesic_pct: float | None = None) -> tuple[dict, dict]:
    """Observed and null metrics from one float64 increment Gram matrix."""
    X = np.asarray(X)
    n = len(X)
    ns = seed + 100
    G = _step_gram_from_array(X)
    D_obs = _distance_from_step_gram(G)
    Lo_euc, Lo_geo, k_obs = _ph_from_dist(D_obs, geodesic_k, geodesic_pct)
    H_euc = np.zeros(n_nulls, dtype=float)
    H_geo = np.zeros(n_nulls, dtype=float)
    K_geo = np.zeros(n_nulls, dtype=int)
    for i in range(n_nulls):
        rng = np.random.default_rng([ns, i, block_size])
        perm = _block_permutation_indices(n - 1, block_size, rng)
        D_null = _distance_from_step_gram(G, perm)
        H_euc[i], H_geo[i], K_geo[i] = _ph_from_dist(
            D_null, geodesic_k, geodesic_pct
        )
    s_euc = compute_stats(Lo_euc, H_euc)
    s_geo = _add_k_diagnostics(compute_stats(Lo_geo, H_geo), k_obs, K_geo, n)
    return s_euc, s_geo
# =============================================================================
# PROJECTION TEST
# =============================================================================
def projection_test(X: np.ndarray, seed: int,
                    n_nulls: int, n_jobs: int) -> dict:
    _, d = X.shape
    ps   = seed + 8000
    def _proj_max(traj: np.ndarray, ph_seed_base: int) -> float:
        proj_list = _gpu_batch_project(traj, ps, ph_seed_base)
        return float(np.max([
            max_H1_euclidean(X2, SUBSAMPLE_PH,
                             np.random.default_rng(ph_seed_base + j))
            for j, X2 in enumerate(proj_list)
        ]))
    obs = _proj_max(X, seed + 3000)
    ns  = seed + 100
    if d > D_THRESH:
        H_null = np.array([
            _proj_max(_make_null(X, i, ns), seed + 4000 + i * 1000)
            for i in range(n_nulls)
        ])
    else:
        rng = np.random.default_rng(ns)
        nulls = matched_step_nulls(X, n_nulls, rng)
        def _null_worker(Xn: np.ndarray, i: int) -> float:
            proj_list = _gpu_batch_project(Xn, ps, seed + 4000 + i * 1000)
            return float(np.max([
                max_H1_euclidean(X2, SUBSAMPLE_PH,
                                 np.random.default_rng(seed + 4000 + i * 1000 + j))
                for j, X2 in enumerate(proj_list)
            ]))
        H_null = np.array(Parallel(n_jobs=n_jobs)(
            delayed(_null_worker)(nulls[i], i) for i in range(n_nulls)))
    return compute_stats(obs, H_null)
# =============================================================================
# CLASSIFICATION
# =============================================================================
def classify_geodesic(s_geo: dict, context: str = "matched_step") -> str:
    """Neutral per-row audit labels.

    These labels intentionally avoid the manuscript-level phrase
    overclaiming recurrence labels. A seed-level or diagnostic-grid p-value is
    not a final recurrence claim until downstream tables apply seed
    reproducibility, numerical-stability, block, and sensitivity audits.
    """
    fired = bool(s_geo["formal_trigger"])
    if context == "matched_step":
        return "matched_step_trigger" if fired else "matched_step_non_detection"
    if context == "pca_diagnostic":
        return "pca_diagnostic_trigger" if fired else "pca_diagnostic_non_trigger"
    if context == "sensitivity":
        return "sensitivity_nominal_trigger" if fired else "sensitivity_non_trigger"
    raise ValueError(f"Unknown decision context: {context}")


def classify_block_diagnostic(s_geo: dict, block_size: int) -> str:
    """Label block rows without promoting b>1 to formal recurrence evidence."""
    if int(block_size) == 1:
        return classify_geodesic(s_geo, context="matched_step")
    return (
        "block_diagnostic_trigger"
        if bool(s_geo["formal_trigger"])
        else "block_diagnostic_non_trigger"
    )
# =============================================================================
# PCA UTILITIES
# =============================================================================
def pca_project(X: np.ndarray, d: int) -> tuple[np.ndarray, float]:
    pca = PCA(n_components=d, svd_solver="full")
    Xd  = pca.fit_transform(X)
    return Xd, float(np.sum(pca.explained_variance_ratio_))
def pca_profile(X: np.ndarray) -> dict:
    n, d  = X.shape
    maxd  = min(n - 1, d)
    pca   = PCA(n_components=maxd, svd_solver="full")
    pca.fit(X)
    cum   = np.cumsum(pca.explained_variance_ratio_)
    d95   = int(np.searchsorted(cum, 0.95)) + 1
    d99   = int(np.searchsorted(cum, 0.99)) + 1
    return dict(d_95=min(d95, maxd), d_99=min(d99, maxd),
                max_d=maxd, cum=cum)
# =============================================================================
# PYTORCH MODELS + TRAINING
# =============================================================================
if TORCH_AVAILABLE:
    class DataPrefetcher:
        def __init__(self, loader: torch.utils.data.DataLoader) -> None:
            self._loader   = loader
            self._iterator = iter(loader)
            self._next     = None
            self._done     = False
            self._lock     = threading.Lock()
            self._ready    = threading.Event()
            self._fetch    = threading.Event()
            self._stop     = threading.Event()
            self._fetch.set()
            self._thread   = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        def _worker(self) -> None:
            while not self._stop.is_set():
                self._fetch.wait(timeout=0.1)
                if self._stop.is_set():
                    break
                if not self._fetch.is_set():
                    continue
                self._fetch.clear()
                with self._lock:
                    try:
                        self._next = next(self._iterator)
                        self._done = False
                    except StopIteration:
                        self._next = None
                        self._done = True
                self._ready.set()
        def close(self) -> None:
            self._stop.set()
            self._fetch.set()
            self._thread.join(timeout=2.0)
        def __enter__(self) -> "DataPrefetcher":
            return self
        def __exit__(self, *_) -> None:
            self.close()
        def __iter__(self) -> "DataPrefetcher":
            return self
        def __next__(self):
            self._ready.wait()
            self._ready.clear()
            if self._done:
                raise StopIteration
            batch = self._next
            self._fetch.set()
            return batch
        def __len__(self) -> int:
            return len(self._loader)
    class MLP2(nn.Module):
        def __init__(self, hidden: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(),
                nn.Linear(hidden, 10))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
            self.c = nn.Sequential(
                nn.Flatten(), nn.Linear(1024, 256), nn.ReLU(),
                nn.Linear(256, 10))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c(self.f(x))
    def get_model(arch: str) -> nn.Module:
        if arch == "mlp2_mnist":
            return MLP2()
        if arch == "smallcnn_cifar10":
            return SmallCNN()
        if arch == "resnet18_cifar10":
            import torchvision.models as models
            m          = models.resnet18(weights=None)
            m.fc       = nn.Linear(512, 10)
            m.conv1    = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            m.maxpool  = nn.Identity()
            return m
        raise ValueError(arch)
    def get_loader(arch: str, train: bool) -> torch.utils.data.DataLoader:
        root = "./data"
        if arch == "mlp2_mnist":
            tfm  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            data = torchvision.datasets.MNIST(
                root, train=train, download=True, transform=tfm)
        else:
            if train:
                tfm = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))])
            else:
                tfm = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))])
            data = torchvision.datasets.CIFAR10(
                root, train=train, download=True, transform=tfm)
        return torch.utils.data.DataLoader(
            data, batch_size=BATCH_SIZE, shuffle=train,
            num_workers=DATALOADER_WORKERS,
            pin_memory=(_dev.type == "cuda"),
            persistent_workers=(DATALOADER_WORKERS > 0))
    def get_optimizer(name: str, model: nn.Module) -> torch.optim.Optimizer:
        if name == "sgd":
            return optim.SGD(model.parameters(), lr=LR_SGD, momentum=0.0)
        if name == "sgd_momentum":
            return optim.SGD(model.parameters(), lr=LR_SGD, momentum=0.9)
        if name == "adam":
            return optim.Adam(model.parameters(), lr=LR_ADAM)
        if name == "adamw":
            return optim.AdamW(model.parameters(), lr=LR_ADAM, weight_decay=1e-2)
        raise ValueError(name)
    def flatten_params(model: nn.Module) -> np.ndarray:
        with torch.no_grad():
            vec = torch.cat([p.detach().view(-1) for p in model.parameters()])
            return vec.cpu().numpy().astype(np.float32)
    def train_and_record(arch: str, opt_name: str,
                         seed: int) -> tuple[np.ndarray, np.ndarray]:
        tp = TRAJ_DIR / f"{arch}__{opt_name}__seed{seed}.npy"
        mp = TRAJ_DIR / f"{arch}__{opt_name}__seed{seed}__metrics.npy"
        if tp.exists() and mp.exists():
            print(f"    Cached: {tp.name}")
            return np.load(tp, mmap_mode='r'), np.load(mp)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = get_model(arch).to(_dev)
        opt   = get_optimizer(opt_name, model)
        crit  = nn.CrossEntropyLoss()
        tr_lo = get_loader(arch, train=True)
        va_lo = get_loader(arch, train=False)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        d_params = sum(p.numel() for p in model.parameters())
        T   = np.lib.format.open_memmap(
                  tp, mode='w+', dtype=np.float32, shape=(EPOCHS, d_params))
        mets = []
        use_prefetch = (arch != "mlp2_mnist")
        def _iter_train():
            if use_prefetch:
                with DataPrefetcher(tr_lo) as pf:
                    yield from pf
            else:
                yield from tr_lo
        def _iter_val():
            if use_prefetch:
                with DataPrefetcher(va_lo) as pf:
                    yield from pf
            else:
                yield from va_lo
        pbar = tqdm(range(1, EPOCHS + 1),
                    desc=f"{arch[:8]}/{opt_name[:6]}",
                    unit="ep", leave=False)
        for ep in pbar:
            model.train()
            ep_loss = 0.0; nb = 0
            for xb, yb in _iter_train():
                xb, yb = xb.to(_dev, non_blocking=True), \
                          yb.to(_dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item(); nb += 1
            sched.step()
            with torch.no_grad():
                vec = torch.cat([p.detach().view(-1) for p in model.parameters()])
                T[ep - 1] = vec.cpu().numpy()
            model.eval(); correct = total = 0
            with torch.no_grad():
                for xb, yb in _iter_val():
                    xb, yb = xb.to(_dev, non_blocking=True), \
                              yb.to(_dev, non_blocking=True)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += yb.size(0)
            acc = 100.0 * correct / total
            mets.append([ep_loss / nb, acc])
            pbar.set_postfix(loss=f"{mets[-1][0]:.3f}", acc=f"{acc:.1f}%")
        M = np.array(mets, dtype=np.float32)
        T.flush()
        del T
        np.save(mp, M)
        del model, opt, crit, tr_lo, va_lo, mets
        gc.collect()
        if _CUDA_OK:
            torch.cuda.empty_cache()
        print(f"\n    Saved: {tp.name}  shape=({EPOCHS},{d_params})"
              f"  final_loss={M[-1,0]:.4f}  final_acc={M[-1,1]:.1f}%")
        return np.load(tp, mmap_mode='r'), M
# =============================================================================
# SEED UTILITIES
# =============================================================================
def get_seed(arch: str, opt: str, offset: int = 0) -> int:
    ai = ARCHITECTURES.index(arch)
    oi = OPTIMIZERS.index(opt)
    return MASTER_SEED + ai * 10000 + oi * 100 + offset
def ckpt_path(arch: str, opt: str, seed: int) -> Path:
    return OUT_DIR / f"ckpt_{arch}__{opt}__seed{seed}.csv"
def block_ckpt_path(arch: str, opt: str, seed: int) -> Path:
    """checkpoint for block-permutation results, separate from main."""
    return OUT_BLOCK / f"block_{arch}__{opt}__seed{seed}.csv"
# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================
def run_single(arch: str, opt: str, seed: int, rep: int) -> list[dict]:
    rows: list[dict] = []
    tp = TRAJ_DIR / f"{arch}__{opt}__seed{seed}.npy"
    used_stub = False
    try:
        if TORCH_AVAILABLE:
            X, metrics = train_and_record(arch, opt, seed)
        else:
            raise RuntimeError("PyTorch not available")
    except Exception as e:
        if not ALLOW_STUB:
            raise RuntimeError(
                "Training/data loading failed; refusing to generate a random-walk stub. "
                "Set ALLOW_STUB=1 only for smoke tests. Stub outputs must not be used "
                "for manuscript tables."
            ) from e
        print(f"  Training failed ({e}); ALLOW_STUB=1, using random-walk stub for smoke test only")
        used_stub = True
        rng     = np.random.default_rng(seed)
        X       = np.cumsum(rng.standard_normal((EPOCHS, 200)) * 0.05, axis=0)
        metrics = np.column_stack([np.linspace(2.0, 0.3, EPOCHS),
                                   np.linspace(20., 90., EPOCHS)])
    N, d = X.shape
    if d > LARGE_D_CHUNKED:
        del X; gc.collect()
        if _CUDA_OK: torch.cuda.empty_cache()
        kin = _kinematic_ratio_chunked(tp)
        print(f"    N={N}  d={d}  R={kin['kinematic_R']:.1f}  "
              f"loss={metrics[-1,0]:.4f}  acc={metrics[-1,1]:.1f}%  "
              f"[chunked-mmap, peak RAM ~80 MB/chunk]")
        print(f"  [Stage 2] Full-space geodesic  "
              f"({N_NULLS_FULL} nulls, chunked-mmap, d={d})")
        t0 = time.time()
        s_euc_f, s_geo_f = ambient_tests_chunked(tp, seed + 5000, N_NULLS_FULL)
        dec_f = classify_geodesic(s_geo_f)
        print(f"    zrob={s_geo_f['zrob']:+.2f}  p={s_geo_f['pperm']:.4f}  "
              f"formal={s_geo_f['formal_trigger']}  "
              f"Lobs={s_geo_f['Lobs']:.5f}  "
              f"zero_frac={s_geo_f['zero_frac']:.2f}  "
              f"R={kin['kinematic_R']:.1f}  "
              f"=> {dec_f}  ({time.time()-t0:.1f}s)")
        if s_geo_f["zero_frac"] > 0.5:
            print(f"    [diag] high zero_frac — geodesic PH below floor at N={N}")
        rows.append(dict(
            arch=arch, optimizer=opt, seed=seed, rep=rep,
            is_stub=used_stub,
            analysis_role="formal_matched_step",
            is_formal_inference=True,
            space="full", pca_d=d, pca_var=1.0,
            prefix_ep=EPOCHS, N_pts=N,
            **{f"geo_{k}": v for k, v in s_geo_f.items()},
            **{f"euc_{k}": v for k, v in s_euc_f.items()},
            decision=dec_f,
            train_loss=float(metrics[-1, 0]),
            val_acc=float(metrics[-1, 1]),
            kinematic_R=kin["kinematic_R"],
            path_length=kin["path_length"],
            net_displacement=kin["net_displacement"],
            pca_d95=None, pca_d99=None, d_full=d,
        ))
        geo_collapsed = (s_geo_f["zero_frac"] >= 0.9)
        traj_straight = (kin["kinematic_R"] < R_THRESH_FAST)
        fast_mode     = geo_collapsed and traj_straight
        n_pca_nulls   = N_NULLS_PCA_FAST if fast_mode else N_NULLS_PCA
        print(f"  [Stage 3] PCA stability  "
              f"({n_pca_nulls} nulls, chunked-mmap PCA, {N_JOBS_PCA} jobs)")
        prof = pca_profile_chunked(tp)
        print(f"    d_95={prof['d_95']}  d_99={prof['d_99']}  max_d={prof['max_d']}")
        dims = sorted(set(
            [pd_ for pd_ in PCA_DIMS_FIXED if pd_ <= prof["max_d"]]
            + [prof["d_95"], prof["d_99"]]
        ))
        pca_cache: dict[int, tuple[np.ndarray, float]] = {}
        for pca_d in dims:
            pca_cache[pca_d] = pca_project_chunked(tp, pca_d)
        for pca_d in dims:
            Xd, var = pca_cache[pca_d]
            for prefix_ep in EPOCH_PREFIXES:
                Xp = Xd[:prefix_ep]
                if len(Xp) < 5:
                    continue
                final_loss = float(metrics[prefix_ep - 1, 0])
                final_acc  = float(metrics[prefix_ep - 1, 1])
                kin_p      = kinematic_ratio(Xp)
                t0 = time.time()
                s_euc, s_geo = ambient_tests(
                    Xp, seed + pca_d * 1000 + prefix_ep,
                    n_nulls=n_pca_nulls, n_jobs=N_JOBS_PCA)
                dec = classify_geodesic(s_geo, context="pca_diagnostic")
                print(f"    d={pca_d:3d} ep={prefix_ep:3d}  "
                      f"geo={s_geo['formal_trigger']}  "
                      f"zrob={s_geo['zrob']:+.2f}  "
                      f"zf={s_geo['zero_frac']:.2f}  "
                      f"R={kin_p['kinematic_R']:.1f}  "
                      f"loss={final_loss:.4f}  acc={final_acc:.1f}%  "
                      f"=> {dec}  ({time.time()-t0:.1f}s)")
                rows.append(dict(
                    arch=arch, optimizer=opt, seed=seed, rep=rep,
                    is_stub=used_stub,
                    analysis_role="projection_diagnostic",
                    is_formal_inference=False,
                    space="pca", pca_d=pca_d, pca_var=var,
                    prefix_ep=prefix_ep, N_pts=len(Xp),
                    **{f"geo_{k}": v for k, v in s_geo.items()},
                    **{f"euc_{k}": v for k, v in s_euc.items()},
                    decision=dec,
                    train_loss=final_loss, val_acc=final_acc,
                    kinematic_R=kin_p["kinematic_R"],
                    path_length=kin_p["path_length"],
                    net_displacement=kin_p["net_displacement"],
                    pca_d95=prof["d_95"], pca_d99=prof["d_99"], d_full=d,
                ))
        return rows
    # NORMAL PATH
    kin = kinematic_ratio(X)
    print(f"    N={N}  d={d}  R={kin['kinematic_R']:.1f}  "
          f"loss={metrics[-1,0]:.4f}  acc={metrics[-1,1]:.1f}%")
    print(f"  [Stage 2] Full-space geodesic  "
          f"({N_NULLS_FULL} nulls, sequential, d={d})")
    t0 = time.time()
    s_euc_f, s_geo_f = ambient_tests(
        X, seed + 5000, n_nulls=N_NULLS_FULL, n_jobs=N_JOBS_FULL)
    dec_f = classify_geodesic(s_geo_f)
    print(f"    zrob={s_geo_f['zrob']:+.2f}  p={s_geo_f['pperm']:.4f}  "
          f"formal={s_geo_f['formal_trigger']}  "
          f"Lobs={s_geo_f['Lobs']:.5f}  "
          f"zero_frac={s_geo_f['zero_frac']:.2f}  "
          f"R={kin['kinematic_R']:.1f}  "
          f"=> {dec_f}  ({time.time()-t0:.1f}s)")
    if s_geo_f["zero_frac"] > 0.5:
        print(f"    [diag] high zero_frac={s_geo_f['zero_frac']:.2f} — "
              f"geodesic PH may be below detection floor at N={N}")
    rows.append(dict(
        arch=arch, optimizer=opt, seed=seed, rep=rep,
        is_stub=used_stub,
        analysis_role="formal_matched_step",
        is_formal_inference=True,
        space="full", pca_d=d, pca_var=1.0,
        prefix_ep=EPOCHS, N_pts=N,
        **{f"geo_{k}": v for k, v in s_geo_f.items()},
        **{f"euc_{k}": v for k, v in s_euc_f.items()},
        decision=dec_f,
        train_loss=float(metrics[-1, 0]),
        val_acc=float(metrics[-1, 1]),
        kinematic_R=kin["kinematic_R"],
        path_length=kin["path_length"],
        net_displacement=kin["net_displacement"],
        pca_d95=None, pca_d99=None, d_full=d,
    ))
    geo_collapsed = (s_geo_f["zero_frac"] >= 0.9)
    traj_straight = (kin["kinematic_R"] < R_THRESH_FAST)
    fast_mode     = geo_collapsed and traj_straight
    n_pca_nulls   = N_NULLS_PCA_FAST if fast_mode else N_NULLS_PCA
    if fast_mode:
        print(f"  [Stage 3] PCA stability  "
              f"({n_pca_nulls} nulls FAST-PATH: "
              f"R={kin['kinematic_R']:.1f}<{R_THRESH_FAST}, "
              f"zero_frac={s_geo_f['zero_frac']:.2f}, {N_JOBS_PCA} jobs)")
    else:
        print(f"  [Stage 3] PCA stability  "
              f"({n_pca_nulls} nulls, {N_JOBS_PCA} jobs)")
    prof = pca_profile(X)
    print(f"    d_95={prof['d_95']}  d_99={prof['d_99']}  max_d={prof['max_d']}")
    dims = sorted(set(
        [pd_ for pd_ in PCA_DIMS_FIXED if pd_ <= prof["max_d"]]
        + [prof["d_95"], prof["d_99"]]
    ))
    pca_cache: dict[int, tuple[np.ndarray, float]] = {}
    for pca_d in dims:
        pca_cache[pca_d] = pca_project(X, pca_d)
    del X
    gc.collect()
    if _CUDA_OK:
        torch.cuda.empty_cache()
    print(f"    [mem] trajectory freed — {d}D → PCA projections only in RAM")
    for pca_d in dims:
        Xd, var = pca_cache[pca_d]
        for prefix_ep in EPOCH_PREFIXES:
            Xp = Xd[:prefix_ep]
            if len(Xp) < 5:
                continue
            final_loss = float(metrics[prefix_ep - 1, 0])
            final_acc  = float(metrics[prefix_ep - 1, 1])
            kin_p      = kinematic_ratio(Xp)
            t0 = time.time()
            s_euc, s_geo = ambient_tests(
                Xp, seed + pca_d * 1000 + prefix_ep,
                n_nulls=n_pca_nulls, n_jobs=N_JOBS_PCA)
            dec = classify_geodesic(s_geo, context="pca_diagnostic")
            print(f"    d={pca_d:3d} ep={prefix_ep:3d}  "
                  f"geo={s_geo['formal_trigger']}  "
                  f"zrob={s_geo['zrob']:+.2f}  "
                  f"zf={s_geo['zero_frac']:.2f}  "
                  f"R={kin_p['kinematic_R']:.1f}  "
                  f"loss={final_loss:.4f}  acc={final_acc:.1f}%  "
                  f"=> {dec}  ({time.time()-t0:.1f}s)")
            rows.append(dict(
                arch=arch, optimizer=opt, seed=seed, rep=rep,
                is_stub=used_stub,
                analysis_role="projection_diagnostic",
                is_formal_inference=False,
                space="pca", pca_d=pca_d, pca_var=var,
                prefix_ep=prefix_ep, N_pts=len(Xp),
                **{f"geo_{k}": v for k, v in s_geo.items()},
                **{f"euc_{k}": v for k, v in s_euc.items()},
                decision=dec,
                train_loss=final_loss, val_acc=final_acc,
                kinematic_R=kin_p["kinematic_R"],
                path_length=kin_p["path_length"],
                net_displacement=kin_p["net_displacement"],
                pca_d95=prof["d_95"], pca_d99=prof["d_99"], d_full=d,
            ))
    return rows
# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================
def run_experiment() -> pd.DataFrame:
    all_rows: list[dict] = []
    t_exp = time.time()
    for arch in ARCHITECTURES:
        for opt in OPTIMIZERS:
            print(f"\n{'='*65}")
            print(f"  {arch}  /  {opt}")
            print(f"{'='*65}")
            arch_opt_rows: list[dict] = []
            for rep, offset in enumerate(SEED_OFFSETS):
                seed = get_seed(arch, opt, offset)
                cp   = ckpt_path(arch, opt, seed)
                if cp.exists():
                    print(f"  [rep {rep}] RESUME  seed={seed}  <- {cp.name}")
                    arch_opt_rows.extend(pd.read_csv(cp).to_dict("records"))
                    continue
                print(f"\n  [rep {rep}/{N_SEEDS-1}]  seed={seed}")
                rows = run_single(arch, opt, seed, rep)
                arch_opt_rows.extend(rows)
                pd.DataFrame(rows).to_csv(cp, index=False)
                print(f"  checkpoint: {cp.name}")
            all_rows.extend(arch_opt_rows)
            full_rows = [r for r in arch_opt_rows if r["space"] == "full"]
            fires     = sum(bool(r["geo_formal_trigger"]) for r in full_rows)
            rr_pca    = sum(
                r.get("decision") == "pca_diagnostic_trigger"
                for r in arch_opt_rows
                if r["space"] == "pca" and r["prefix_ep"] == EPOCHS)
            print(f"\n  Full-space geo: {fires}/{len(full_rows)} seeds fire  |  "
                  f"PCA diagnostic triggers at ep=200: {rr_pca} "
                  f"(arch={arch}, opt={opt})")
    df = pd.DataFrame(all_rows)
    _save_outputs(df)
    elapsed = (time.time() - t_exp) / 60
    print(f"\n{'='*65}")
    print(f"Complete in {elapsed:.1f} min")
    print(f"  Main text  -> {OUT_MAIN}")
    print(f"  Appendix   -> {OUT_APPX}")
    print(f"{'='*65}")
    return df
# =============================================================================
# STAGE 4 — BLOCK-PERMUTATION ROBUSTNESS
# =============================================================================
def run_block_robustness_single(arch: str, opt: str, seed: int,
                                rep: int) -> list[dict]:
    """
    Run full-space geodesic PH with block-permuted nulls for each
    block size in BLOCK_SIZES. Reads existing trajectory from disk.
    Returns one row per (arch, opt, seed, block_size).
    """
    rows: list[dict] = []
    tp = TRAJ_DIR / f"{arch}__{opt}__seed{seed}.npy"
    if not tp.exists():
        print(f"    [skip] trajectory file not found: {tp.name}")
        return rows
    if d_full := _peek_trajectory_dim(tp):
        pass
    use_chunked = d_full > LARGE_D_CHUNKED
    if use_chunked:
        kin = _kinematic_ratio_chunked(tp)
    else:
        X_full = np.load(tp, mmap_mode='r')
        kin = kinematic_ratio(np.asarray(X_full))
        del X_full
    print(f"    [block] {arch}/{opt}/seed{seed}  d={d_full}  "
          f"R={kin['kinematic_R']:.1f}  chunked={use_chunked}")
    for b in BLOCK_SIZES:
        t0 = time.time()
        if use_chunked:
            s_euc, s_geo = ambient_tests_chunked(
                tp, seed + 5000, n_nulls=N_NULLS_BLOCK, block_size=b)
        else:
            X_full = np.asarray(np.load(tp, mmap_mode='r'))
            s_euc, s_geo = ambient_tests(
                X_full, seed + 5000,
                n_nulls=N_NULLS_BLOCK, n_jobs=N_JOBS_BLOCK,
                block_size=b)
            del X_full
            gc.collect()
            if _CUDA_OK:
                torch.cuda.empty_cache()
        dec = classify_block_diagnostic(s_geo, b)
        elapsed = time.time() - t0
        print(f"      b={b:2d}  zrob={s_geo['zrob']:+.2f}  "
              f"p={s_geo['pperm']:.4f}  formal={s_geo['formal_trigger']}  "
              f"Lobs={s_geo['Lobs']:.5f}  zf={s_geo['zero_frac']:.2f}  "
              f"=> {dec}  ({elapsed:.1f}s)")
        rows.append(dict(
            arch=arch, optimizer=opt, seed=seed, rep=rep,
            block_size=b, d_full=d_full,
            analysis_role="block_diagnostic",
            is_formal_inference=False,
            kinematic_R=kin["kinematic_R"],
            **{f"geo_{k}": v for k, v in s_geo.items()},
            **{f"euc_{k}": v for k, v in s_euc.items()},
            decision=dec,
            elapsed_sec=elapsed,
        ))
    return rows
def _peek_trajectory_dim(tp: Path) -> int:
    """Return the dimension d of a saved (T, d) trajectory without loading it."""
    arr = np.load(tp, mmap_mode='r')
    d = int(arr.shape[1])
    del arr
    return d
def run_block_robustness() -> pd.DataFrame:
    print()
    print("=" * 70)
    print(f"  STAGE 4 — Block-Permutation Robustness")
    print(f"  block_sizes={BLOCK_SIZES}  n_nulls={N_NULLS_BLOCK}")
    print(f"  b=1 is the matched-step null used for formal inference.")
    print(f"  b>1 preserves local serial dependence as a robustness diagnostic.")
    print(f"  Type-I exchangeability (Prop 1) applies only at b=1.")
    print("=" * 70)
    all_rows: list[dict] = []
    t_exp = time.time()
    for arch in ARCHITECTURES:
        for opt in OPTIMIZERS:
            print(f"\n  {arch}  /  {opt}")
            for rep, offset in enumerate(SEED_OFFSETS):
                seed = get_seed(arch, opt, offset)
                cp   = block_ckpt_path(arch, opt, seed)
                if cp.exists():
                    print(f"    [rep {rep}] RESUME  seed={seed}  <- {cp.name}")
                    all_rows.extend(pd.read_csv(cp).to_dict("records"))
                    continue
                rows = run_block_robustness_single(arch, opt, seed, rep)
                if rows:
                    all_rows.extend(rows)
                    pd.DataFrame(rows).to_csv(cp, index=False)
                    print(f"    checkpoint: {cp.name}")
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("\n  [warn] No block-robustness rows produced. "
              "Are .npy trajectories missing?")
        return df
    out_full = OUT_BLOCK / "block_robustness_full.csv"
    df.to_csv(out_full, index=False)
    print(f"\n  Full block-robustness table -> {out_full} ({len(df)} rows)")
    # ----- Aggregate summary tables for the appendix -----
    summary_rows = []
    for (arch, opt), grp in df.groupby(["arch", "optimizer"]):
        for b in BLOCK_SIZES:
            sub = grp[grp["block_size"] == b]
            if sub.empty:
                continue
            n_seeds = len(sub)
            n_fire  = int(sub["geo_formal_trigger"].sum())
            p_min   = float(sub["geo_pperm"].min())
            p_med   = float(sub["geo_pperm"].median())
            zrob_med = float(sub["geo_zrob"].median())
            summary_rows.append(dict(
                arch=arch, optimizer=opt, block_size=b,
                n_seeds=n_seeds, seeds_fired=n_fire,
                p_min=p_min, p_median=p_med, zrob_median=zrob_med,
            ))
    summary = pd.DataFrame(summary_rows)
    out_summary = OUT_BLOCK / "block_robustness_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"  Summary table             -> {out_summary} ({len(summary)} rows)")
    # ----- Decision-stability cross-tab: do decisions change with b? -----
    pivot = df.pivot_table(
        index=["arch", "optimizer", "seed"],
        columns="block_size",
        values="geo_formal_trigger",
        aggfunc="first",
    )
    pivot["decision_stable_across_b"] = pivot.apply(
        lambda r: int(len(set(r.dropna().astype(bool))) <= 1), axis=1)
    out_pivot = OUT_BLOCK / "block_robustness_pivot.csv"
    pivot.to_csv(out_pivot)
    print(f"  Decision pivot            -> {out_pivot}")
    # ----- Console summary -----
    print("\n  Decision counts per (arch, block_size):")
    print(df.groupby(["arch", "block_size", "decision"]).size().to_string())
    print("\n  Decisions stable across block sizes?")
    n_total  = len(pivot)
    n_stable = int(pivot["decision_stable_across_b"].sum())
    print(f"    {n_stable}/{n_total} (arch, opt, seed) cells "
          f"have identical decisions across b ∈ {BLOCK_SIZES}")
    elapsed = (time.time() - t_exp) / 60
    print(f"\n  Block-robustness sweep complete in {elapsed:.1f} min")
    return df
# =============================================================================
# STAGE 5 — NEURAL k / FILTRATION-THRESHOLD SENSITIVITY
# =============================================================================
def sens_ckpt_path(arch: str, opt: str, seed: int) -> Path:
    return OUT_SENS / f"sens_{arch}__{opt}__seed{seed}.csv"
def run_neural_sensitivity_single(arch: str, opt: str,
                                   seed: int, rep: int) -> list[dict]:
    """
    Full-space geodesic sensitivity sweep over k and filtration percentile.
    Reads existing trajectory from disk. No retraining.
    Loads X_full once per seed outside the k/pct loop to avoid redundant
    disk reads. Uses N_JOBS_BLOCK for normal-dimensional trajectories and
    N_NULLS_SENS_CHUNKED for ResNet-18 to limit runtime.
    """
    rows: list[dict] = []
    tp = TRAJ_DIR / f"{arch}__{opt}__seed{seed}.npy"
    if not tp.exists():
        print(f"    [skip] trajectory file not found: {tp.name}")
        return rows
    d_full      = _peek_trajectory_dim(tp)
    use_chunked = d_full > LARGE_D_CHUNKED
    if use_chunked:
        kin    = _kinematic_ratio_chunked(tp)
        X_full = None  # stays on disk; ambient_tests_chunked reads via mmap
        n_nulls_to_use = N_NULLS_SENS_CHUNKED
        print(f"    [sens] {arch}/{opt}/seed{seed}  d={d_full}  "
              f"R={kin['kinematic_R']:.1f}  chunked=True  "
              f"n_nulls={n_nulls_to_use} (reduced for ResNet-18)")
    else:
        X_mmap = np.load(tp, mmap_mode="r")
        X_full = np.asarray(X_mmap)   # load ONCE for all k/pct iterations
        del X_mmap
        kin    = kinematic_ratio(X_full)
        n_nulls_to_use = N_NULLS_SENS
        print(f"    [sens] {arch}/{opt}/seed{seed}  d={d_full}  "
              f"R={kin['kinematic_R']:.1f}  chunked=False  "
              f"n_nulls={n_nulls_to_use}")
    for k_val in K_SENS_VALUES:
        for pct_val in PCT_SENS_VALUES:
            t0        = time.time()
            sens_seed = seed + 9000 + 100 * int(k_val) + int(pct_val)
            if use_chunked:
                s_euc, s_geo = ambient_tests_chunked(
                    tp, sens_seed, n_nulls=n_nulls_to_use,
                    block_size=1,
                    geodesic_k=k_val, geodesic_pct=pct_val)
            else:
                s_euc, s_geo = ambient_tests(
                    X_full, sens_seed, n_nulls=n_nulls_to_use,
                    n_jobs=N_JOBS_BLOCK,  # FIX: use 4 workers, not 1
                    block_size=1,
                    geodesic_k=k_val, geodesic_pct=pct_val)
            dec     = classify_geodesic(s_geo, context="sensitivity")
            elapsed = time.time() - t0
            print(f"      k={k_val:2d}  pct={pct_val:4.1f}  "
                  f"zrob={s_geo['zrob']:+.2f}  p={s_geo['pperm']:.4f}  "
                  f"formal={s_geo['formal_trigger']}  "
                  f"Lobs={s_geo['Lobs']:.5f}  "
                  f"zf={s_geo['zero_frac']:.2f}  "
                  f"=> {dec}  ({elapsed:.1f}s)")
            rows.append(dict(
                arch=arch, optimizer=opt, seed=seed, rep=rep,
                k_value=k_val, filtration_pct=pct_val,
                analysis_role="metric_sensitivity",
                is_formal_inference=False,
                d_full=d_full, kinematic_R=kin["kinematic_R"],
                n_nulls_used=n_nulls_to_use,
                **{f"geo_{kk}": vv for kk, vv in s_geo.items()},
                **{f"euc_{kk}": vv for kk, vv in s_euc.items()},
                decision=dec, elapsed_sec=elapsed,
            ))
    if X_full is not None:
        del X_full
        gc.collect()
        if _CUDA_OK:
            torch.cuda.empty_cache()
    return rows
def run_neural_sensitivity() -> pd.DataFrame:
    print()
    print("=" * 70)
    print("  STAGE 5 — Neural k / Filtration-Threshold Sensitivity")
    print(f"  k values: {K_SENS_VALUES}  (main pipeline uses k={GEODESIC_K})")
    print(f"  filtration percentiles: {PCT_SENS_VALUES}  (main uses {GEODESIC_PCT})")
    print(f"  n_nulls (normal-d): {N_NULLS_SENS}   n_nulls (chunked/ResNet-18): {N_NULLS_SENS_CHUNKED}")
    print(f"  X loaded once per seed (no redundant disk reads).")
    print("=" * 70)
    all_rows: list[dict] = []
    t_exp = time.time()
    for arch in ARCHITECTURES:
        for opt in OPTIMIZERS:
            print(f"\n  {arch}  /  {opt}")
            for rep, offset in enumerate(SEED_OFFSETS):
                seed = get_seed(arch, opt, offset)
                cp   = sens_ckpt_path(arch, opt, seed)
                if cp.exists():
                    print(f"    [rep {rep}] RESUME  seed={seed}  <- {cp.name}")
                    all_rows.extend(pd.read_csv(cp).to_dict("records"))
                    continue
                rows = run_neural_sensitivity_single(arch, opt, seed, rep)
                if rows:
                    all_rows.extend(rows)
                    pd.DataFrame(rows).to_csv(cp, index=False)
                    print(f"    checkpoint: {cp.name}")
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("\n  [warn] No sensitivity rows. Are .npy trajectories present?")
        return df
    out_full = OUT_SENS / "neural_sensitivity_full.csv"
    df.to_csv(out_full, index=False)
    print(f"\n  Full sensitivity table -> {out_full} ({len(df)} rows)")
    summary_rows = []
    for (arch, opt, k_val, pct_val), grp in df.groupby(
            ["arch", "optimizer", "k_value", "filtration_pct"]):
        n_seeds = len(grp)
        n_fire  = int(grp["geo_formal_trigger"].sum())
        summary_rows.append(dict(
            arch=arch, optimizer=opt, k_value=k_val, filtration_pct=pct_val,
            n_seeds=n_seeds, seeds_fired=n_fire,
            p_min=float(grp["geo_pperm"].min()),
            p_median=float(grp["geo_pperm"].median()),
            zrob_median=float(grp["geo_zrob"].median()),
            Lobs_median=float(grp["geo_Lobs"].median()),
        ))
    summary = pd.DataFrame(summary_rows)
    out_summary = OUT_SENS / "neural_sensitivity_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"  Summary table         -> {out_summary} ({len(summary)} rows)")
    pivot = df.pivot_table(
        index=["arch", "optimizer", "seed"],
        columns=["k_value", "filtration_pct"],
        values="geo_formal_trigger", aggfunc="first")
    pivot["decision_stable_across_grid"] = pivot.apply(
        lambda r: int(len(set(r.dropna().astype(bool))) <= 1), axis=1)
    out_pivot = OUT_SENS / "neural_sensitivity_pivot.csv"
    pivot.to_csv(out_pivot)
    print(f"  Decision pivot        -> {out_pivot}")
    n_total  = len(pivot)
    n_stable = int(pivot["decision_stable_across_grid"].sum())
    print(f"\n  Decisions stable across k/pct grid: {n_stable}/{n_total}")
    print("\n  Decision counts per k/pct:")
    print(df.groupby(["k_value", "filtration_pct", "decision"]).size().to_string())
    elapsed = (time.time() - t_exp) / 60
    print(f"\n  Neural sensitivity sweep complete in {elapsed:.1f} min")
    return df
# =============================================================================
# OUTPUT ROUTING
# =============================================================================
def _save_outputs(df: pd.DataFrame) -> None:
    if "is_stub" in df.columns and df["is_stub"].fillna(False).astype(bool).any():
        if OUT_DIR.name != "exp4_results_smoke":
            raise RuntimeError(
                "Refusing to write manuscript-facing outputs because stub rows are present."
            )
        print("  [smoke] Stub rows present; outputs are quarantined under exp4_results_smoke/.")
    df.to_csv(OUT_DIR / "exp4_results_full.csv", index=False)
    main_cols = [
        "arch", "optimizer", "seed", "rep", "analysis_role",
        "is_formal_inference", "space", "pca_d",
        "geo_Lobs", "geo_zrob", "geo_pperm", "geo_formal_trigger",
        "geo_zero_frac", "geo_null_collapsed",
        "geo_k_used_obs", "geo_null_k_min", "geo_null_k_median",
        "geo_null_k_max", "geo_null_k_near_complete_frac", "decision",
        "train_loss", "val_acc", "kinematic_R", "d_full",
    ]
    df_full = df[df["space"] == "full"][
        [c for c in main_cols if c in df.columns]]
    df_full.to_csv(OUT_MAIN / "table_main_topology.csv", index=False)
    print(f"  Main table: {len(df_full)} rows -> "
          f"{OUT_MAIN/'table_main_topology.csv'}")
    df.to_csv(OUT_APPX / "table_appendix_full.csv", index=False)
    print(f"  Appendix:   {len(df)} rows -> "
          f"{OUT_APPX/'table_appendix_full.csv'}")
    _plot_phase_map(df)
    _plot_training_curves(df)
def _plot_phase_map(df: pd.DataFrame) -> None:
    pca_full = df[
        (df["space"] == "pca") &
        (df["prefix_ep"] == EPOCHS) &
        df["kinematic_R"].notna()
    ]
    if pca_full.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    colours = {
        "matched_step_trigger": "#2ca02c",
        "matched_step_non_detection": "#1f77b4",
        "pca_diagnostic_trigger": "#2ca02c",
        "pca_diagnostic_non_trigger": "#1f77b4",
        "pca_nominal_trigger": "#2ca02c",
        "pca_non_trigger": "#1f77b4",
        "sensitivity_nominal_trigger": "#9467bd",
        "sensitivity_non_trigger": "#7f7f7f",
        "block_diagnostic_trigger": "#ff7f0e",
        "block_diagnostic_non_trigger": "#8c564b",
    }
    markers = {
        "mlp2_mnist":       "o",
        "smallcnn_cifar10": "s",
        "resnet18_cifar10": "^",
    }
    for (dec, arch), grp in pca_full.groupby(["decision", "arch"]):
        ax.scatter(
            grp["kinematic_R"], grp["geo_zrob"],
            c=colours.get(dec, "grey"),
            marker=markers.get(arch, "o"),
            label=f"{arch[:10]} / {dec[:20]}",
            alpha=0.75, s=70, edgecolors="k", linewidths=0.5)
    ax.axhline(ZROB_THRESH, color="green", ls="--", lw=1.0, alpha=0.5,
               label=f"z_rob={ZROB_THRESH} (descriptive reference)")
    ax.set_xscale("log")
    ax.set_xlabel("Kinematic Ratio  R = path / displacement", fontsize=11)
    ax.set_ylabel("Geodesic effect size  z_rob  [descriptive]", fontsize=11)
    ax.set_title("Appendix — PCA Stability Phase Map\n"
                 f"(Geodesic formal trigger on PCA trajectories, ep={EPOCHS})",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = OUT_APPX / "fig_phase_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Phase map  -> {out}")
def _plot_training_curves(df: pd.DataFrame) -> None:
    full = df[(df["space"] == "full") & df["train_loss"].notna()]
    if full.empty:
        return
    for arch in full["arch"].unique():
        arch_df = full[full["arch"] == arch]
        fig, axes = plt.subplots(1, len(OPTIMIZERS), figsize=(14, 3.5),
                                 sharey=False)
        fig.suptitle(f"{arch} — Training dynamics", fontsize=11)
        for ax, opt in zip(axes, OPTIMIZERS):
            sub = arch_df[arch_df["optimizer"] == opt]
            ax.set_title(opt, fontsize=9)
            ax.set_xlabel("epoch"); ax.set_ylabel("train loss")
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                mp = TRAJ_DIR / f"{arch}__{opt}__seed{int(row['seed'])}__metrics.npy"
                if not mp.exists():
                    continue
                M  = np.load(mp)
                ep = np.arange(1, len(M) + 1)
                ax.plot(ep, M[:, 0], alpha=0.7,
                        label=f"seed{int(row['seed'])}")
                ax.text(0.98, 0.95,
                        f"dec={row.get('decision','?')}\n"
                        f"R={row.get('kinematic_R', 0):.0f}",
                        transform=ax.transAxes, fontsize=6,
                        va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            ax.legend(fontsize=6)
        fig.tight_layout()
        out = OUT_APPX / f"training_{arch}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Training curves -> {out}")
# =============================================================================
# SYNTHETIC LOOP SANITY CHECK
# =============================================================================
CIRCLE_NOISE_BASE = 0.05
CIRCLE_RADIUS     = 1.0
CIRCLE_N_CYCLES   = 4
def make_circle_trajectory(n_pts: int, n_cycles: int, r: float,
                           d_embed: int, seed: int) -> np.ndarray:
    t       = np.linspace(0, n_cycles * 2 * np.pi, n_pts, endpoint=False)
    X       = np.zeros((n_pts, d_embed), dtype=np.float32)
    X[:, 0] = r * np.cos(t)
    X[:, 1] = r * np.sin(t)
    if d_embed > 2:
        sigma    = CIRCLE_NOISE_BASE / np.sqrt(d_embed - 2)
        X[:, 2:] = np.random.default_rng(seed).standard_normal(
                       (n_pts, d_embed - 2)).astype(np.float32) * sigma
    return X
def run_sanity_circle(d_embed: int = 2048, n_cycles: int = CIRCLE_N_CYCLES,
                      label: str = "") -> dict:
    tag  = label or f"d={d_embed} n_cycles={n_cycles}"
    seed = 9999
    X    = make_circle_trajectory(EPOCHS, n_cycles, CIRCLE_RADIUS, d_embed, seed)
    kin  = kinematic_ratio(X)
    print(f"  [{tag}]  shape={X.shape}  R={kin['kinematic_R']:.1f}")
    t0 = time.time()
    _, s_geo = ambient_tests(X, seed, n_nulls=N_NULLS_FULL, n_jobs=N_JOBS_FULL)
    dec_time = time.time() - t0
    decision = "synthetic_loop_detected" if s_geo["formal_trigger"] else "FAILED_SANITY"
    print(f"    zrob={s_geo['zrob']:+.2f}  p={s_geo['pperm']:.4f}  "
          f"formal={s_geo['formal_trigger']}  Lobs={s_geo['Lobs']:.5f}  "
          f"zero_frac={s_geo['zero_frac']:.2f}  => {decision}  ({dec_time:.1f}s)")
    if decision == "FAILED_SANITY":
        print("    SANITY FAILURE: formal trigger did not detect the synthetic loop.")
    return dict(label=tag, d_embed=d_embed, n_cycles=n_cycles,
                decision=decision, **{f"geo_{k}": v for k, v in s_geo.items()},
                kinematic_R=kin["kinematic_R"])
def run_sanity_negcontrol() -> dict:
    seed = 9998
    rng  = np.random.default_rng(seed)
    X    = np.cumsum(rng.standard_normal((EPOCHS, 2048)).astype(np.float32) * 0.05,
                     axis=0)
    kin  = kinematic_ratio(X)
    print(f"  [random walk d=2048]  shape={X.shape}  R={kin['kinematic_R']:.1f}")
    t0 = time.time()
    _, s_geo = ambient_tests(X, seed, n_nulls=N_NULLS_FULL, n_jobs=N_JOBS_FULL)
    dec_time = time.time() - t0
    decision = "random_walk_non_trigger" if not s_geo["formal_trigger"] else "FALSE_POSITIVE"
    print(f"    zrob={s_geo['zrob']:+.2f}  p={s_geo['pperm']:.4f}  "
          f"formal={s_geo['formal_trigger']}  => {decision}  ({dec_time:.1f}s)")
    if decision == "FALSE_POSITIVE":
        print("    FALSE POSITIVE: formal trigger fired on a random walk.")
    return dict(label="random_walk_d2048", d_embed=2048, n_cycles=0,
                decision=decision, **{f"geo_{k}": v for k, v in s_geo.items()},
                kinematic_R=kin["kinematic_R"])
def run_experiment_0c() -> pd.DataFrame:
    sanity_dir = OUT_DIR / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("=" * 70)
    print("  Synthetic loop sanity check")
    print(f"  {CIRCLE_N_CYCLES} cycles, N={EPOCHS}, noise=CIRCLE_NOISE_BASE/sqrt(d-2)")
    print("=" * 70)
    rows = []
    print("\n  Test A: single loop (n_cycles=1, d=2048)")
    rows.append(run_sanity_circle(d_embed=2048, n_cycles=1, label="1-cycle d=2048"))
    print("\n  Test B: 4 cycles matching cosine-LR schedule (n_cycles=4, d=2048)")
    rows.append(run_sanity_circle(d_embed=2048, n_cycles=4, label="4-cycle d=2048"))
    mlp2_d = 203530
    print(f"\n  Test C: 4 cycles at real MLP2 dimension (n_cycles=4, d={mlp2_d})")
    rows.append(run_sanity_circle(d_embed=mlp2_d, n_cycles=4,
                                  label=f"4-cycle d={mlp2_d}"))
    print("\n  Test D: negative control — random walk (expect non-trigger)")
    rows.append(run_sanity_negcontrol())
    df = pd.DataFrame(rows)
    print("\n  Sanity check summary:")
    for _, row in df.iterrows():
        status = "pass" if row["decision"] in (
            "synthetic_loop_detected", "random_walk_non_trigger") else "FAIL"
        print(f"    [{status}]  {row['label']:30s}  => {row['decision']}")
    all_pass = all(
        row["decision"] in ("synthetic_loop_detected", "random_walk_non_trigger")
        for _, row in df.iterrows())
    print()
    if all_pass:
        print("  All sanity checks passed — neural audit outputs are interpretable.")
    else:
        print("  SANITY FAILURE — investigate before trusting neural audit outputs.")
    out = sanity_dir / "sanity_results.csv"
    df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    return df
# =============================================================================
# UTILITY
# =============================================================================
# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if not SKIP_MAIN:
        df_sanity = run_experiment_0c()
        sanity_ok = all(
            row["decision"] in ("synthetic_loop_detected", "random_walk_non_trigger")
            for _, row in df_sanity.iterrows())
        if not sanity_ok:
            raise RuntimeError(
                "Synthetic sanity check failed. Refusing to run neural trajectory audit."
            )
        else:
            df = run_experiment()
            print("\nDecision summary (full-space):")
            if not df.empty:
                fs = df[df["space"] == "full"]
                print(fs.groupby(["arch", "optimizer", "decision"]).size()
                        .to_string())
                print("\nDecision summary (PCA stability, ep=200):")
                pca_ep = df[(df["space"] == "pca") & (df["prefix_ep"] == EPOCHS)]
                print(pca_ep.groupby(["arch", "decision"]).size().to_string())
    else:
        print("\n[SKIP_MAIN=1] Skipping synthetic sanity check and neural trajectory audit.")
    if RUN_BLOCK_ROBUSTNESS:
        df_block = run_block_robustness()
    else:
        print("\n[RUN_BLOCK_ROBUSTNESS=0] Skipping block-permutation "
              "robustness diagnostics. Set RUN_BLOCK_ROBUSTNESS=1 to enable.")
    if RUN_NEURAL_SENSITIVITY:
        df_sens = run_neural_sensitivity()
    else:
        print("\n[RUN_NEURAL_SENSITIVITY=0] Skipping neural k / threshold "
              "sensitivity diagnostics. Set RUN_NEURAL_SENSITIVITY=1 to enable.")
