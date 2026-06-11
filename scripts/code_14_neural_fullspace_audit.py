#!/usr/bin/env python3
"""Optimization-stress neural full-space audit.

Purpose
-------
Stress-test whether highly tortuous SmallCNN/CIFAR-10 optimization trajectories
produce detectable full-space graph-geodesic H1 recurrence under the matched-step
null.

Stress regimes
--------------
1. Edge of stability: large batch, high learning rate, no momentum.
2. Extreme momentum: momentum = 0.99 with elevated learning rate.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Outputs
-------
Results are written under exp5_results/.
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
# OMP thread pinning — must happen before heavy NumPy/SciPy imports.
# ---------------------------------------------------------------------------
import os as _os
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_env_var, "1")
import gc
import random
import time
import warnings
from pathlib import Path
import platform as _platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from ripser import ripser
from sklearn.decomposition import PCA
# =============================================================================
# TORCH
# =============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"PyTorch device: {_dev}"
        + (f"  ({torch.cuda.get_device_name(0)})" if _dev.type == "cuda" else "")
    )
    if _dev.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA capability: {cap[0]}.{cap[1]}  VRAM: {vram:.1f} GB")
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError as exc:
    TORCH_AVAILABLE = False
    _dev = None
    raise RuntimeError("PyTorch/torchvision is required for Experiment 5.") from exc
# =============================================================================
# CONFIG
# =============================================================================
EPOCHS = 200
MASTER_SEED = 42
SEED_OFFSETS = [0, 1000, 2000]
N_SEEDS = len(SEED_OFFSETS)
FORCE_RETRAIN = bool(int(_os.environ.get("FORCE_RETRAIN", "0")))
# Stress regimes
STRESS_CONFIGS = [
    {
        "name": "edge_of_stability",
        "label": "EoS",
        "batch_size": 2048,
        "lr": 0.08,
        "momentum": 0.0,
        "weight_decay": 0.0,
    },
    {
        "name": "extreme_momentum",
        "label": "Momentum99",
        "batch_size": 512,
        "lr": 0.03,
        "momentum": 0.99,
        "weight_decay": 0.0,
    },
]
# Topology / PH settings
N_NULLS_FULL = 200
N_NULLS_PCA = 200
N_NULLS_PCA_FAST = 100
SUBSAMPLE_PH = 200
ALPHA = 0.05
DELTA_MIN = 1e-3
LOBS_MULT = 5.0
EPS = 1e-12
EPS_SIG = 1e-6
MAD_SCALE = 1.4826
GEODESIC_K = 12
GEODESIC_MAX_K = 50
GEODESIC_PCT = 95.0
PCA_DIMS_FIXED = [3, 10, 50]
EPOCH_PREFIXES = [100, 150, 200]
N_JOBS_FULL = 1
N_JOBS_PCA = 4
D_THRESH = 1000
R_THRESH_FAST = 3.0
DATALOADER_WORKERS = 4
if _platform.system() == "Windows":
    DATALOADER_WORKERS = 0
OUT_DIR = Path("exp5_results")
TRAJ_DIR = OUT_DIR / "trajectories"
OUT_MAIN = OUT_DIR / "main_paper"
OUT_APPX = OUT_DIR / "appendix"
for _d in [OUT_DIR, TRAJ_DIR, OUT_MAIN, OUT_APPX]:
    _d.mkdir(parents=True, exist_ok=True)
print("=" * 70)
print("CODE 14 — Optimization Stress-Test Neural Full-Space Audit")
print("=" * 70)
print(f"{len(STRESS_CONFIGS)} stress configs x {N_SEEDS} seeds = {len(STRESS_CONFIGS) * N_SEEDS} runs")
print(f"EPOCHS={EPOCHS}")
print(f"N_NULLS_FULL={N_NULLS_FULL}")
print(f"N_NULLS_PCA={N_NULLS_PCA}")
print(f"SUBSAMPLE_PH={SUBSAMPLE_PH}")
print(f"GEODESIC_K={GEODESIC_K}")
print(f"GEODESIC_PCT={GEODESIC_PCT}")
print(f"N_JOBS_PCA={N_JOBS_PCA}")
print(f"DATALOADER_WORKERS={DATALOADER_WORKERS}")
print(f"FORCE_RETRAIN={FORCE_RETRAIN}")
print(f"Formal trigger: pperm < {ALPHA}; separation flag is diagnostic only")
print("zrob and TSR are descriptive only.")
print("=" * 70)
# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_all_seeds(seed: int) -> None:
    """
    Set Python, NumPy, and PyTorch seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic settings for reproducibility.
    # This may slightly slow training, but improves auditability.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# =============================================================================
# STATISTICS
# =============================================================================
def _mad(a: np.ndarray) -> float:
    """
    Median absolute deviation.
    """
    a = np.asarray(a, dtype=float)
    return float(np.median(np.abs(a - np.median(a))))
# =============================================================================
# KINEMATICS
# =============================================================================
def kinematic_ratio(X: np.ndarray) -> dict:
    """
    Path length / net displacement.
    """
    steps = np.diff(X, axis=0)
    path_len = float(np.sum(np.linalg.norm(steps, axis=1)))
    net_disp = float(np.linalg.norm(X[-1] - X[0]))
    return dict(
        path_length=path_len,
        net_displacement=net_disp,
        kinematic_R=path_len / max(net_disp, 1e-8),
    )
# =============================================================================
# DISTANCE + PH
# =============================================================================
def gram_dist(X: np.ndarray) -> np.ndarray:
    """
    Pairwise Euclidean distance matrix using float64 arithmetic.
    This matches the non-GPU Experiment 4 path and avoids float32
    cancellation near numerical floors.
    """
    X64 = X.astype(np.float64, copy=False)
    sq = np.einsum("ij,ij->i", X64, X64).reshape(-1, 1)
    D2 = sq + sq.T - 2.0 * (X64 @ X64.T)
    return np.sqrt(np.maximum(D2, 0.0))
def _subsample(X: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random subsampling.
    In Experiment 5, full-space trajectory length is EPOCHS=200 and
    SUBSAMPLE_PH=200, so full-space PH uses all epochs. PCA prefixes are
    also <= 200, so no subsampling occurs there either.
    """
    if len(X) > cap:
        idx = rng.choice(len(X), cap, replace=False)
        return X[idx]
    return X
def _geodesic_from_dist(D_full: np.ndarray) -> float:
    """Shared graph-geodesic max H1 lifetime from an ambient distance matrix."""
    if len(D_full) < 4:
        return 0.0
    return geodesic_h1_lifetime(
        D_full,
        input_distance_matrix=True,
        k0=GEODESIC_K,
        kmax=GEODESIC_MAX_K,
        pct=GEODESIC_PCT,
        eps=EPS,
        sentinel_fill=False,
    )
def max_H1_euclidean(
    X: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> float:
    """
    Euclidean VR PH maximum finite H1 lifetime.
    """
    X = _subsample(X, cap, rng)
    D = gram_dist(X)
    iu = np.triu_indices_from(D, k=1)
    thresh = float(np.percentile(D[iu], GEODESIC_PCT)) if len(D[iu]) else 1.0
    dgms = ripser(
        D,
        maxdim=1,
        distance_matrix=True,
        thresh=thresh,
    )["dgms"]
    if len(dgms) < 2 or len(dgms[1]) == 0:
        return 0.0
    H1 = dgms[1]
    finite_h1 = H1[np.isfinite(H1[:, 1])]
    if len(finite_h1) == 0:
        return 0.0
    return float(np.max(finite_h1[:, 1] - finite_h1[:, 0]))
def max_H1_geodesic(
    X: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> float:
    """
    Geodesic VR PH maximum finite H1 lifetime.
    """
    X = _subsample(X, cap, rng)
    D = gram_dist(X)
    return _geodesic_from_dist(D)
def make_null(X: np.ndarray, seed: int) -> np.ndarray:
    """Matched-step permutation null via audit_common.matched_step_null."""
    rng = np.random.default_rng(seed)
    return matched_step_null(X, rng).astype(np.float32)
def _one_null_stats(
    X: np.ndarray,
    idx: int,
    master: int,
) -> tuple[float, float]:
    """
    One null statistic pair: Euclidean and geodesic.
    """
    Xn = make_null(
        X,
        master + idx,
    )
    e = max_H1_euclidean(
        Xn,
        SUBSAMPLE_PH,
        np.random.default_rng(master + 200 + idx),
    )
    g = max_H1_geodesic(
        Xn,
        SUBSAMPLE_PH,
        np.random.default_rng(master + 400 + idx),
    )
    del Xn
    return e, g
def ambient_tests(
    X: np.ndarray,
    seed: int,
    n_nulls: int,
    n_jobs: int,
) -> tuple[dict, dict]:
    """
    Ambient Euclidean and graph-geodesic PH under matched-step null.
    """
    _, d = X.shape
    ns = seed + 100
    # Large-d full-space: do one null at a time to avoid memory blow-up.
    if d > D_THRESH or n_jobs == 1:
        H_euc = np.empty(n_nulls, dtype=np.float64)
        H_geo = np.empty(n_nulls, dtype=np.float64)
        for i in range(n_nulls):
            Xn = make_null(
                X,
                ns + i,
            )
            H_euc[i] = max_H1_euclidean(
                Xn,
                SUBSAMPLE_PH,
                np.random.default_rng(seed + 200 + i),
            )
            H_geo[i] = max_H1_geodesic(
                Xn,
                SUBSAMPLE_PH,
                np.random.default_rng(seed + 400 + i),
            )
            del Xn
            if i % 10 == 0:
                gc.collect()
    else:
        pairs = Parallel(n_jobs=n_jobs)(
            delayed(_one_null_stats)(
                X,
                i,
                ns,
            )
            for i in range(n_nulls)
        )
        H_euc = np.array(
            [p[0] for p in pairs],
            dtype=np.float64,
        )
        H_geo = np.array(
            [p[1] for p in pairs],
            dtype=np.float64,
        )
    Lo_euc = max_H1_euclidean(
        X,
        SUBSAMPLE_PH,
        np.random.default_rng(seed + 10),
    )
    Lo_geo = max_H1_geodesic(
        X,
        SUBSAMPLE_PH,
        np.random.default_rng(seed + 300),
    )
    return compute_stats(Lo_euc, H_euc), compute_stats(Lo_geo, H_geo)
def classify_geodesic(s_geo: dict, *, space: str) -> str:
    """
    Neutral raw-output label based only on the geodesic formal trigger.

    These labels deliberately avoid the manuscript-level phrase
    ``robust_recurrence``. Seed reproducibility, numerical checks, and
    paper-level audit labels must be reconstructed downstream.
    """
    fired = bool(s_geo["formal_trigger"])
    if space == "full":
        return "matched_step_trigger" if fired else "matched_step_non_detection"
    if space == "pca":
        return "pca_nominal_trigger" if fired else "pca_non_trigger"
    return "nominal_trigger" if fired else "non_trigger"
# =============================================================================
# PCA UTILITIES
# =============================================================================
def pca_project(X: np.ndarray, d: int) -> tuple[np.ndarray, float]:
    """
    PCA projection to d dimensions.
    """
    pca = PCA(
        n_components=d,
        svd_solver="full",
    )
    Xd = pca.fit_transform(X)
    return Xd.astype(np.float32), float(np.sum(pca.explained_variance_ratio_))
def pca_profile(X: np.ndarray) -> dict:
    """
    PCA cumulative explained variance profile.
    """
    n, d = X.shape
    maxd = min(n - 1, d)
    pca = PCA(
        n_components=maxd,
        svd_solver="full",
    )
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d95 = int(np.searchsorted(cum, 0.95)) + 1
    d99 = int(np.searchsorted(cum, 0.99)) + 1
    return dict(
        d_95=min(d95, maxd),
        d_99=min(d99, maxd),
        max_d=maxd,
        cum=cum,
    )
# =============================================================================
# MODEL + TRAINING
# =============================================================================
class SmallCNN(nn.Module):
    """
    Same SmallCNN architecture used in Experiment 4.
    """
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.c = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(self.f(x))
def get_loader(
    train: bool,
    batch_size: int,
    seed: int,
) -> torch.utils.data.DataLoader:
    """
    CIFAR-10 loader.
    Training transform matches Experiment 4:
        RandomCrop + RandomHorizontalFlip + ToTensor + Normalize.
    Validation transform:
        ToTensor + Normalize.
    """
    root = "./data"
    if train:
        tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])
    else:
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])
    data = torchvision.datasets.CIFAR10(
        root,
        train=train,
        download=True,
        transform=tfm,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=train,
        num_workers=DATALOADER_WORKERS,
        pin_memory=(_dev.type == "cuda"),
        persistent_workers=(DATALOADER_WORKERS > 0),
        generator=generator,
    )
def get_optimizer(
    config: dict,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """
    SGD optimizer for stress regimes.
    """
    return optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config.get("weight_decay", 0.0),
        nesterov=False,
    )
def flatten(model: nn.Module) -> np.ndarray:
    """
    Flatten model parameters.
    """
    with torch.no_grad():
        return torch.cat(
            [p.detach().reshape(-1) for p in model.parameters()]
        ).cpu().numpy().astype(np.float32)
def _validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    crit: nn.Module,
) -> tuple[float, float]:
    """
    Validation loss and accuracy.
    """
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(
                _dev,
                non_blocking=True,
            )
            yb = yb.to(
                _dev,
                non_blocking=True,
            )
            out = model(xb)
            loss = crit(out, yb)
            loss_sum += float(loss.item()) * int(yb.size(0))
            pred = out.argmax(1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
    val_loss = loss_sum / max(total, 1)
    val_acc = 100.0 * correct / max(total, 1)
    return val_loss, val_acc
def train_stress_model(
    config: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Train one SmallCNN stress-test trajectory.
    Returns:
        trajectory, metrics, status
    """
    tp = TRAJ_DIR / f"smallcnn__{config['name']}__seed{seed}.npy"
    mp = TRAJ_DIR / f"smallcnn__{config['name']}__seed{seed}__metrics.npy"
    if (not FORCE_RETRAIN) and tp.exists() and mp.exists():
        print(f"    Cached: {tp.name}")
        return np.load(tp, mmap_mode="r"), np.load(mp), "ok"
    if FORCE_RETRAIN:
        if tp.exists():
            tp.unlink()
        if mp.exists():
            mp.unlink()
    set_all_seeds(seed)
    model = SmallCNN().to(_dev)
    opt = get_optimizer(config, model)
    crit = nn.CrossEntropyLoss()
    tr_lo = get_loader(
        train=True,
        batch_size=config["batch_size"],
        seed=seed + 100,
    )
    va_lo = get_loader(
        train=False,
        batch_size=config["batch_size"],
        seed=seed + 200,
    )
    d_params = sum(p.numel() for p in model.parameters())
    traj = np.empty(
        (EPOCHS, d_params),
        dtype=np.float32,
    )
    metrics = np.empty(
        (EPOCHS, 3),
        dtype=np.float32,
    )
    status = "ok"
    done = 0
    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        nb = 0
        for xb, yb in tr_lo:
            xb = xb.to(
                _dev,
                non_blocking=True,
            )
            yb = yb.to(
                _dev,
                non_blocking=True,
            )
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        train_loss = ep_loss / max(nb, 1)
        flat = flatten(model)
        if not np.isfinite(flat).all():
            status = "diverged"
            done = ep
            print(f"    Diverged at epoch {ep + 1}; stopping early.")
            break
        val_loss, val_acc = _validate(
            model,
            va_lo,
            crit,
        )
        if not (np.isfinite(val_loss) and np.isfinite(val_acc)):
            status = "diverged"
            done = ep
            print(f"    Validation blew up at epoch {ep + 1}; stopping early.")
            break
        traj[ep] = flat
        metrics[ep] = np.array(
            [train_loss, val_loss, val_acc],
            dtype=np.float32,
        )
        done = ep + 1
        if (ep + 1) % 25 == 0 or ep == 0:
            print(
                f"    ep={ep + 1:3d}/{EPOCHS}"
                f" train_loss={train_loss:.4f}"
                f" val_loss={val_loss:.4f}"
                f" val_acc={val_acc:.1f}%"
            )
    traj = traj[:done].copy()
    metrics = metrics[:done].copy()
    if status == "ok":
        np.save(tp, traj)
        np.save(mp, metrics)
        print(
            f"\n    Saved: {tp.name}"
            f" shape={traj.shape}"
            f" final_train_loss={metrics[-1, 0]:.4f}"
            f" final_val_loss={metrics[-1, 1]:.4f}"
            f" final_val_acc={metrics[-1, 2]:.1f}%"
        )
    else:
        print(
            f"    Status={status}; trajectory length={len(traj)}."
        )
    del model, opt, crit, tr_lo, va_lo
    gc.collect()
    if TORCH_AVAILABLE and _dev is not None and _dev.type == "cuda":
        torch.cuda.empty_cache()
    return traj, metrics, status
# =============================================================================
# EXPERIMENT LOOP
# =============================================================================
def get_seed(
    config_idx: int,
    rep: int,
) -> int:
    return MASTER_SEED + config_idx * 10000 + SEED_OFFSETS[rep]
def ckpt_path(
    config_name: str,
    seed: int,
) -> Path:
    return OUT_DIR / f"ckpt_{config_name}__seed{seed}.csv"
def run_single(
    config: dict,
    seed: int,
    rep: int,
) -> list[dict]:
    """
    Run one stress config / seed.
    """
    rows: list[dict] = []
    print(f"  [rep {rep}/{N_SEEDS - 1}] seed={seed}  {config['label']}")
    X, M, status = train_stress_model(
        config,
        seed,
    )
    if status != "ok" or len(X) < 5:
        rows.append(dict(
            experiment="stress",
            stress_name=config["name"],
            stress_label=config["label"],
            seed=seed,
            rep=rep,
            status=status,
            space="full",
            pca_d=np.nan,
            pca_var=np.nan,
            prefix_ep=np.nan,
            N_pts=len(X),
            decision="diverged",
            train_loss=np.nan,
            val_loss=np.nan,
            val_acc=np.nan,
            kinematic_R=np.nan,
            path_length=np.nan,
            net_displacement=np.nan,
            d_full=np.nan,
        ))
        return rows
    N, d = X.shape
    kin = kinematic_ratio(X)
    print(
        f"    N={N} d={d} R={kin['kinematic_R']:.1f}"
        f" train_loss={M[-1, 0]:.4f}"
        f" val_loss={M[-1, 1]:.4f}"
        f" val_acc={M[-1, 2]:.1f}%"
    )
    # -----------------------------------------------------------------------
    # Stage 2 — full-space geodesic
    # -----------------------------------------------------------------------
    print(f"  [Stage 2] Full-space geodesic ({N_NULLS_FULL} nulls, sequential)")
    t0 = time.time()
    s_euc_f, s_geo_f = ambient_tests(
        X,
        seed + 5000,
        n_nulls=N_NULLS_FULL,
        n_jobs=N_JOBS_FULL,
    )
    dec_f = classify_geodesic(s_geo_f, space="full")
    print(
        f"    zrob={s_geo_f['zrob']:+.2f}"
        f" p={s_geo_f['pperm']:.4f}"
        f" formal={s_geo_f['formal_trigger']}"
        f" Lobs={s_geo_f['Lobs']:.5f}"
        f" zf={s_geo_f['zero_frac']:.2f}"
        f" R={kin['kinematic_R']:.1f}"
        f" => {dec_f}"
        f" ({time.time() - t0:.1f}s)"
    )
    rows.append(dict(
        experiment="stress",
        stress_name=config["name"],
        stress_label=config["label"],
        seed=seed,
        rep=rep,
        status=status,
        space="full",
        pca_d=d,
        pca_var=1.0,
        prefix_ep=EPOCHS,
        N_pts=N,
        **{f"geo_{k}": v for k, v in s_geo_f.items()},
        **{f"euc_{k}": v for k, v in s_euc_f.items()},
        decision=dec_f,
        train_loss=float(M[-1, 0]),
        val_loss=float(M[-1, 1]),
        val_acc=float(M[-1, 2]),
        kinematic_R=kin["kinematic_R"],
        path_length=kin["path_length"],
        net_displacement=kin["net_displacement"],
        d_full=d,
    ))
    # -----------------------------------------------------------------------
    # Stage 3 — PCA stability
    # -----------------------------------------------------------------------
    prof = pca_profile(X)
    dims = sorted(set(
        [pd_ for pd_ in PCA_DIMS_FIXED if pd_ <= prof["max_d"]]
        + [prof["d_95"], prof["d_99"]]
    ))
    pca_cache: dict[int, tuple[np.ndarray, float]] = {}
    for pca_d in dims:
        pca_cache[pca_d] = pca_project(
            X,
            pca_d,
        )
    geo_collapsed = (s_geo_f["zero_frac"] >= 0.9)
    traj_straight = (kin["kinematic_R"] < R_THRESH_FAST)
    n_pca_nulls = (
        N_NULLS_PCA_FAST
        if (geo_collapsed and traj_straight)
        else N_NULLS_PCA
    )
    print(f"  [Stage 3] PCA stability ({n_pca_nulls} nulls, {N_JOBS_PCA} jobs)")
    print(f"    d_95={prof['d_95']} d_99={prof['d_99']} max_d={prof['max_d']}")
    del X
    gc.collect()
    if TORCH_AVAILABLE and _dev is not None and _dev.type == "cuda":
        torch.cuda.empty_cache()
    for pca_d in dims:
        Xd, var = pca_cache[pca_d]
        for prefix_ep in EPOCH_PREFIXES:
            if prefix_ep > len(Xd):
                continue
            Xp = Xd[:prefix_ep]
            if len(Xp) < 5:
                continue
            final_train_loss = float(M[prefix_ep - 1, 0])
            final_val_loss = float(M[prefix_ep - 1, 1])
            final_val_acc = float(M[prefix_ep - 1, 2])
            kin_p = kinematic_ratio(Xp)
            t0 = time.time()
            s_euc, s_geo = ambient_tests(
                Xp,
                seed + pca_d * 1000 + prefix_ep,
                n_nulls=n_pca_nulls,
                n_jobs=N_JOBS_PCA,
            )
            dec = classify_geodesic(s_geo, space="pca")
            print(
                f"    d={pca_d:3d}"
                f" ep={prefix_ep:3d}"
                f" geo={s_geo['formal_trigger']}"
                f" zrob={s_geo['zrob']:+.2f}"
                f" zf={s_geo['zero_frac']:.2f}"
                f" R={kin_p['kinematic_R']:.1f}"
                f" => {dec}"
                f" ({time.time() - t0:.1f}s)"
            )
            rows.append(dict(
                experiment="stress",
                stress_name=config["name"],
                stress_label=config["label"],
                seed=seed,
                rep=rep,
                status=status,
                space="pca",
                pca_d=pca_d,
                pca_var=var,
                prefix_ep=prefix_ep,
                N_pts=len(Xp),
                **{f"geo_{k}": v for k, v in s_geo.items()},
                **{f"euc_{k}": v for k, v in s_euc.items()},
                decision=dec,
                train_loss=final_train_loss,
                val_loss=final_val_loss,
                val_acc=final_val_acc,
                kinematic_R=kin_p["kinematic_R"],
                path_length=kin_p["path_length"],
                net_displacement=kin_p["net_displacement"],
                d_full=d,
            ))
    return rows
def run_experiment() -> pd.DataFrame:
    """
    Run all stress configs and seeds.
    """
    all_rows: list[dict] = []
    t_exp = time.time()
    for ci, config in enumerate(STRESS_CONFIGS):
        print()
        print("=" * 70)
        print(
            f"STRESS CONFIG: {config['name']}"
            f" [LR={config['lr']} MOM={config['momentum']} BS={config['batch_size']}]"
        )
        print("=" * 70)
        cfg_rows: list[dict] = []
        for rep in range(N_SEEDS):
            seed = get_seed(
                ci,
                rep,
            )
            cp = ckpt_path(
                config["name"],
                seed,
            )
            if FORCE_RETRAIN and cp.exists():
                cp.unlink()
            if cp.exists():
                print(f"  [rep {rep}] RESUME seed={seed} <- {cp.name}")
                cfg_rows.extend(
                    pd.read_csv(cp).to_dict("records")
                )
                continue
            rows = run_single(
                config,
                seed,
                rep,
            )
            cfg_rows.extend(rows)
            pd.DataFrame(rows).to_csv(
                cp,
                index=False,
            )
            print(f"  checkpoint: {cp.name}")
        all_rows.extend(cfg_rows)
        full_rows = [
            r for r in cfg_rows
            if r["space"] == "full"
        ]
        full_fires = sum(
            bool(r.get("geo_formal_trigger"))
            for r in full_rows
            if r.get("status") == "ok"
        )
        pca_ep200 = [
            r for r in cfg_rows
            if r["space"] == "pca"
            and r["prefix_ep"] == EPOCHS
        ]
        pca_fires = sum(
            bool(r.get("geo_formal_trigger"))
            for r in pca_ep200
        )
        print()
        print(
            f"Summary for {config['name']}:"
            f" full-space fires {full_fires}/{len(full_rows)} seeds;"
            f" PCA ep=200 nominal triggers {pca_fires}/{len(pca_ep200)} rows"
        )
    df = pd.DataFrame(all_rows)
    save_outputs(df)

    # Manuscript-facing safety check: this stress-test table supports the claim
    # that full-space matched-step geodesic detections remain absent in these
    # optimization stress regimes. Save outputs first, then fail loudly if the
    # expected absence of full-space formal triggers is violated.
    full_ok = df[(df["space"] == "full") & (df["status"] == "ok")].copy()
    bad_full = full_ok[full_ok["geo_formal_trigger"].astype(bool)]
    if len(bad_full):
        raise RuntimeError(
            "Optimization stress test produced full-space matched-step formal "
            "geodesic triggers; this contradicts the expected manuscript claim "
            "of absent full-space detections. Outputs have been saved for audit.\n"
            + bad_full[
                [
                    "stress_name", "seed", "geo_Lobs", "geo_pperm",
                    "geo_TSR", "geo_zero_frac", "kinematic_R", "decision",
                ]
            ].to_string(index=False)
        )

    elapsed_min = (time.time() - t_exp) / 60.0
    print()
    print("=" * 70)
    print(f"Complete in {elapsed_min:.1f} min")
    print(f"Main text -> {OUT_MAIN}")
    print(f"Appendix  -> {OUT_APPX}")
    print("=" * 70)
    return df
# =============================================================================
# OUTPUTS / FIGURES
# =============================================================================
def save_outputs(df: pd.DataFrame) -> None:
    """
    Save full, main, and appendix outputs.
    """
    df.to_csv(
        OUT_DIR / "exp5_results_full.csv",
        index=False,
    )
    main_cols = [
        "stress_name",
        "stress_label",
        "seed",
        "rep",
        "status",
        "space",
        "pca_d",
        "geo_Lobs",
        "geo_null_med",
        "geo_null_max",
        "geo_TSR",
        "geo_zrob",
        "geo_pperm",
        "geo_delta",
        "geo_zero_frac",
        "geo_fallback",
        "geo_formal_trigger",
        "decision",
        "train_loss",
        "val_loss",
        "val_acc",
        "kinematic_R",
        "path_length",
        "net_displacement",
        "d_full",
    ]
    present_main_cols = [
        c for c in main_cols
        if c in df.columns
    ]
    df_full = df[df["space"] == "full"][present_main_cols]
    df_full.to_csv(
        OUT_MAIN / "table_stress_main.csv",
        index=False,
    )
    df.to_csv(
        OUT_APPX / "table_stress_appendix.csv",
        index=False,
    )
    plot_phase_map(df)
    plot_loss_curves(df)
    print(f"Main table -> {OUT_MAIN / 'table_stress_main.csv'}")
    print(f"Appendix   -> {OUT_APPX / 'table_stress_appendix.csv'}")
def plot_phase_map(df: pd.DataFrame) -> None:
    """
    Phase map: kinematic R versus descriptive zrob.
    """
    pca_full = df[
        (df["space"] == "pca")
        & (df["prefix_ep"] == EPOCHS)
    ]
    if pca_full.empty:
        return
    fig, ax = plt.subplots(
        figsize=(9, 6),
    )
    colours = {
        "matched_step_trigger": "tab:green",
        "matched_step_non_detection": "tab:blue",
        "pca_nominal_trigger": "tab:green",
        "pca_non_trigger": "tab:blue",
        "nominal_trigger": "tab:green",
        "non_trigger": "tab:blue",
        "diverged": "tab:red",
    }
    markers = {
        "edge_of_stability": "o",
        "extreme_momentum": "s",
    }
    for (dec, stress), grp in pca_full.groupby(
        ["decision", "stress_name"]
    ):
        ax.scatter(
            grp["kinematic_R"],
            grp["geo_zrob"],
            c=colours.get(dec, "grey"),
            marker=markers.get(stress, "o"),
            label=f"{stress} / {dec}",
            alpha=0.75,
            s=65,
            edgecolors="k",
            linewidths=0.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Kinematic ratio R = path length / displacement")
    ax.set_ylabel("Geodesic effect size zrob [descriptive]")
    ax.set_title("Experiment 5 — Stress-test phase map, epoch 200")
    ax.grid(True, alpha=0.2)
    ax.legend(
        fontsize=8,
        loc="upper left",
    )
    fig.tight_layout()
    out = OUT_APPX / "stress_phase_map.png"
    fig.savefig(
        out,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Phase map -> {out}")
def plot_loss_curves(df: pd.DataFrame) -> None:
    """
    Save train/validation loss curves for stress runs.
    """
    full = df[
        (df["space"] == "full")
        & df["status"].eq("ok")
    ]
    if full.empty:
        return
    for stress in full["stress_name"].unique():
        fig, ax = plt.subplots(
            figsize=(7.5, 4),
        )
        sub = full[
            full["stress_name"] == stress
        ]
        for _, row in sub.iterrows():
            mp = TRAJ_DIR / f"smallcnn__{stress}__seed{int(row['seed'])}__metrics.npy"
            if not mp.exists():
                continue
            M = np.load(mp)
            ep = np.arange(1, len(M) + 1)
            ax.plot(
                ep,
                M[:, 0],
                alpha=0.8,
                label=f"train seed{int(row['seed'])}",
            )
            ax.plot(
                ep,
                M[:, 1],
                alpha=0.8,
                linestyle="--",
                label=f"val seed{int(row['seed'])}",
            )
        ax.set_title(
            stress.replace("_", " ").title()
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(
            fontsize=6,
            ncol=2,
        )
        ax.grid(
            True,
            alpha=0.2,
        )
        fig.tight_layout()
        out = OUT_APPX / f"loss_{stress}.png"
        fig.savefig(
            out,
            dpi=140,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"Loss curves -> {out}")
# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    df_stress = run_experiment()
    print()
    print("Decision summary (full-space):")
    fs = df_stress[
        df_stress["space"] == "full"
    ]
    if not fs.empty:
        print(
            fs.groupby(
                ["stress_name", "decision"]
            ).size().to_string()
        )
    print()
    print("Decision summary (PCA stability, ep=200):")
    pca_ep = df_stress[
        (df_stress["space"] == "pca")
        & (df_stress["prefix_ep"] == EPOCHS)
    ]
    if not pca_ep.empty:
        print(
            pca_ep.groupby(
                ["stress_name", "decision"]
            ).size().to_string()
        )
