#!/usr/bin/env python3
"""Forced-recurrence positive control.

Purpose
-------
Verify that the graph-geodesic matched-step audit detects a trajectory with an
explicitly injected recurrent signal.

Experiment
----------
A linear model is trained to follow a slowly rotating target function. The
target weight vector traces a circle in two coordinates with slow drift in a
third coordinate.

Decision convention
-------------------
Formal triggers are defined by pperm < ALPHA. The separation flag, TSR, zrob,
delta, and null-collapse diagnostics are descriptive only.

Implementation convention
-------------------------
The observed statistic and every null statistic use the same adaptive
graph-geodesic max-H1-lifetime rule.

Outputs
-------
Writes forced_recurrence_results.csv, null lifetimes, null k-values, and the
matched-step null histogram under results/forced_recurrence_positive_control/.
"""

from __future__ import annotations

# Shared audit primitives.
from pathlib import Path as _AuditPath
import sys as _audit_sys

_audit_sys.path.insert(0, str(_AuditPath(__file__).resolve().parents[1]))

from audit_common import (
    compute_stats,
    matched_step_null,
    geodesic_distance_matrix,
    h1_lifetime_from_distance_matrix,
)

from pathlib import Path
import time
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = Path("results") / "forced_recurrence_positive_control"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42

D_IN = 50
D_OUT = 1

NUM_STEPS = 800
NUM_CYCLES = 2.0

LR = 0.05
MOMENTUM = 0.9
BATCH_SIZE = 64

TARGET_RADIUS = 2.0
DRIFT_SCALE = 0.001
TARGET_NOISE = 0.1

SUBSAMPLE = 400

# Default null count.
N_NULLS = 200

GEODESIC_K = 8
# Formal-run policy: observed/null geodesic statistic must be defined
# without sentinel fill. Some matched-step nulls are disconnected at
# kmax=50, so allow adaptive k up to full SUBSAMPLE connectivity.
GEODESIC_MAX_K = SUBSAMPLE - 1
GEODESIC_PCT = 95.0

ALPHA = 0.05
DELTA_MIN = 1e-3
LOBS_MULT = 5.0

EPS = 1e-12
EPS_SIG = 1e-6
MAD_SCALE = 1.4826


print("=" * 70)
print("Forced-recurrence positive control")
print("=" * 70)
print(f"RNG_SEED={RNG_SEED}")
print(f"D_IN={D_IN}, D_OUT={D_OUT}")
print(f"NUM_STEPS={NUM_STEPS}, NUM_CYCLES={NUM_CYCLES}")
print(f"SUBSAMPLE={SUBSAMPLE}, N_NULLS={N_NULLS}")
print(f"GEODESIC_K={GEODESIC_K}, GEODESIC_MAX_K={GEODESIC_MAX_K}")
print(f"GEODESIC_PCT={GEODESIC_PCT}")
print(f"Formal trigger: pperm < {ALPHA}; separation flag is diagnostic only")
print("Observed/null statistic: same adaptive-k geodesic H1 lifetime")
print("zrob and TSR are descriptive only.")
print("=" * 70)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """
    Set python, numpy, and torch RNG seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# POSITIVE CONTROL TRAJECTORY
# =============================================================================

def generate_robust_positive_control(
    d_in: int = D_IN,
    d_out: int = D_OUT,
    num_steps: int = NUM_STEPS,
    num_cycles: float = NUM_CYCLES,
    lr: float = LR,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """
    Generate a trajectory forced to trace a loop.

    A linear model follows a target weight vector w_star whose first two
    coordinates rotate on a circle and whose third coordinate drifts.

    Returns:
        trajectory: array of shape (num_steps, d_in * d_out)
    """
    set_all_seeds(seed)

    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)

    model = nn.Linear(d_in, d_out, bias=False)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=MOMENTUM,
    )

    trajectory = []

    for step in range(num_steps):
        theta = 2.0 * np.pi * (step / num_steps) * num_cycles

        w_star = torch.zeros(d_in, d_out)
        w_star[0, 0] = TARGET_RADIUS * np.cos(theta)
        w_star[1, 0] = TARGET_RADIUS * np.sin(theta)
        w_star[2, 0] = DRIFT_SCALE * step

        X = torch.randn(
            BATCH_SIZE,
            d_in,
            generator=torch_gen,
        )

        with torch.no_grad():
            Y_target = (
                X @ w_star
                + TARGET_NOISE * torch.randn(
                    BATCH_SIZE,
                    d_out,
                    generator=torch_gen,
                )
            )

        loss = nn.functional.mse_loss(
            model(X),
            Y_target,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        trajectory.append(
            model.weight.detach().clone().flatten().cpu().numpy()
        )

    return np.asarray(trajectory, dtype=np.float32)


# =============================================================================
# SUBSAMPLING
# =============================================================================

def stride_subsample_indices(n: int, cap: int) -> np.ndarray:
    """
    Deterministic stride subsampling indices for ordered trajectory.
    """
    if n <= cap:
        return np.arange(n, dtype=int)

    return np.linspace(0, n - 1, cap, dtype=int)


# =============================================================================
# GEODESIC PH
# =============================================================================

def max_H1_from_geodesic_distance(D_geo: np.ndarray | None) -> float:
    """
    Max finite H1 lifetime from a geodesic distance matrix.

    The lifetime extractor performs median rescaling internally using positive
    finite distances only, then applies the GEODESIC_PCT filtration threshold.
    """
    if D_geo is None:
        return 0.0

    return h1_lifetime_from_distance_matrix(
        D_geo,
        pct=GEODESIC_PCT,
        eps=EPS,
        rescale=True,
    )


def compute_geodesic_once_strict(
    traj: np.ndarray,
    k: int,
) -> float | None:
    """
    Strict fixed-k geodesic PH.

    Returns:
        Lmax if the graph is connected at this fixed k.
        None if the graph is disconnected.

    No sentinel fill is allowed here.
    """
    if len(traj) < 4:
        return 0.0

    try:
        D_geo, _, _ = geodesic_distance_matrix(
            traj,
            k0=k,
            kmax=k,
            sentinel_fill=False,
            rescale=False,
            eps=EPS,
            context_label=f"forced_recurrence_fixed_k={k}",
        )

    except RuntimeError:
        return None

    return max_H1_from_geodesic_distance(D_geo)


def compute_geodesic_Lmax_adaptive(
    traj: np.ndarray,
    k_start: int = GEODESIC_K,
    k_max: int = GEODESIC_MAX_K,
) -> tuple[float, int]:
    """
    Adaptive geodesic PH with geometric k-growth.

    This is the single statistic used for BOTH observed and null trajectories.

    Returns:
        Lmax, k_used
    """
    if len(traj) < 4:
        return 0.0, min(k_start, max(1, len(traj) - 1))

    hard_max = min(int(k_max), len(traj) - 1)
    k = min(max(1, int(k_start)), hard_max)

    while True:
        L_max = compute_geodesic_once_strict(
            traj,
            k,
        )

        if L_max is not None:
            return float(L_max), int(k)

        if k >= hard_max:
            break

        k = min(2 * k, hard_max)

    raise RuntimeError(
        f"Graph remained disconnected up to k={hard_max}; "
        "cannot assign Lmax=0 in a formal positive-control audit."
    )


# =============================================================================
# MATCHED-STEP PERMUTATION NULL
# =============================================================================

def permute_steps(
    traj: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Matched-step permutation null via audit_common.matched_step_null.
    """
    return matched_step_null(traj, rng)


# =============================================================================
# MAIN
# =============================================================================

def main() -> pd.DataFrame:
    t0 = time.time()

    print()
    print("1. Generating positive-control trajectory...")

    full_trajectory = generate_robust_positive_control(
        d_in=D_IN,
        d_out=D_OUT,
        num_steps=NUM_STEPS,
        num_cycles=NUM_CYCLES,
        lr=LR,
        seed=RNG_SEED,
    )

    traj_path = OUT_DIR / "forced_recurrence_trajectory.npy"
    np.save(traj_path, full_trajectory)

    print(f"   Saved trajectory: {traj_path}")
    print(f"   Trajectory shape: {full_trajectory.shape}")

    indices = stride_subsample_indices(
        len(full_trajectory),
        SUBSAMPLE,
    )

    X_obs = full_trajectory[indices]

    print()
    print("2. Computing observed adaptive-k geodesic PH...")

    L_obs, k_obs = compute_geodesic_Lmax_adaptive(
        X_obs,
        k_start=GEODESIC_K,
        k_max=GEODESIC_MAX_K,
    )

    print(f"   Observed Lmax : {L_obs:.6f}")
    print(f"   Observed k_used: {k_obs}")

    print()
    print(f"3. Generating matched-step null distribution: N_NULLS={N_NULLS}")
    print("   Nulls use the SAME adaptive-k statistic as the observed trajectory.")

    rng_null_master = np.random.default_rng(RNG_SEED + 1000)

    L_null: list[float] = []
    null_k_used: list[int] = []

    for i in tqdm(range(N_NULLS)):
        null_seed = int(rng_null_master.integers(0, 2**32 - 1))
        rng_null = np.random.default_rng(null_seed)

        traj_null = permute_steps(
            full_trajectory,
            rng_null,
        )

        X_null = traj_null[indices]

        L_max_null, k_null = compute_geodesic_Lmax_adaptive(
            X_null,
            k_start=GEODESIC_K,
            k_max=GEODESIC_MAX_K,
        )

        L_null.append(float(L_max_null))
        null_k_used.append(int(k_null))

    L_null_arr = np.asarray(
        L_null,
        dtype=float,
    )

    null_k_used_arr = np.asarray(
        null_k_used,
        dtype=int,
    )

    null_path = OUT_DIR / "forced_recurrence_null_lifetimes.npy"
    null_k_path = OUT_DIR / "forced_recurrence_null_k_used.npy"

    np.save(null_path, L_null_arr)
    np.save(null_k_path, null_k_used_arr)

    stats = compute_stats(
        L_obs,
        L_null_arr,
        alpha=ALPHA,
        delta_min=DELTA_MIN,
        lobs_mult=LOBS_MULT,
        eps=EPS,
        eps_sig=EPS_SIG,
        mad_scale=MAD_SCALE,
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    print()
    print("=" * 70)
    print("FORMAL-TRIGGER RESULTS")
    print("=" * 70)
    print(f"Observed Lmax    : {stats['Lobs']:.6f}")
    print(f"Observed k_used  : {k_obs}")
    print(
        "Null k_used      : "
        f"min={int(null_k_used_arr.min())}, "
        f"median={float(np.median(null_k_used_arr)):.1f}, "
        f"max={int(null_k_used_arr.max())}"
    )
    print(f"Null median      : {stats['null_med']:.6f}")
    print(f"Null max         : {stats['null_max']:.6f}")
    print(f"TSR              : {stats['TSR']:.3f}")
    print(f"Delta            : {stats['delta']:.6f}")
    print(f"Percentile       : {stats['pct']:.1f}%")
    print(f"Robust z-score   : {stats['zrob']:+.2f}")
    print(f"Zero fraction    : {stats['zero_frac']:.3f}")
    print(f"P-value          : {stats['pperm']:.4f}")
    print(f"Separation flag  : {stats['separation_flag']}")
    print(f"Formal trigger   : {stats['formal_trigger']}")
    print(f"Nominal trigger  : {stats['nominal_trigger']}  [diagnostic only]")
    print("=" * 70)

    if stats["formal_trigger"]:
        print(">>> SIGNIFICANT GEODESIC STRUCTURE DETECTED <<<")
    else:
        print(">>> NO FORMAL RECURRENCE DETECTED <<<")

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    row = dict(
        experiment="forced_recurrence_positive_control",
        RNG_SEED=RNG_SEED,
        D_IN=D_IN,
        D_OUT=D_OUT,
        NUM_STEPS=NUM_STEPS,
        NUM_CYCLES=NUM_CYCLES,
        LR=LR,
        MOMENTUM=MOMENTUM,
        BATCH_SIZE=BATCH_SIZE,
        TARGET_RADIUS=TARGET_RADIUS,
        DRIFT_SCALE=DRIFT_SCALE,
        TARGET_NOISE=TARGET_NOISE,
        SUBSAMPLE=SUBSAMPLE,
        N_NULLS=N_NULLS,
        GEODESIC_K=GEODESIC_K,
        GEODESIC_MAX_K=GEODESIC_MAX_K,
        GEODESIC_PCT=GEODESIC_PCT,
        ALPHA=ALPHA,
        DELTA_MIN=DELTA_MIN,
        LOBS_MULT=LOBS_MULT,
        k_obs=int(k_obs),
        null_k_min=int(null_k_used_arr.min()),
        null_k_median=float(np.median(null_k_used_arr)),
        null_k_max=int(null_k_used_arr.max()),
        **stats,
    )

    df = pd.DataFrame([row])

    csv_path = OUT_DIR / "forced_recurrence_results.csv"

    df.to_csv(
        csv_path,
        index=False,
    )

    print()
    print(f"Saved results CSV    : {csv_path}")
    print(f"Saved null lifetimes : {null_path}")
    print(f"Saved null k-values  : {null_k_path}")

    # =========================================================================
    # HISTOGRAM
    # =========================================================================

    plt.figure(figsize=(8, 5))

    plt.hist(
        L_null_arr,
        bins=20,
        alpha=0.7,
        edgecolor="black",
        label="Matched-step null",
    )

    plt.axvline(
        L_obs,
        linestyle="dashed",
        linewidth=2,
        label=f"Observed Lmax = {L_obs:.3f}",
    )

    plt.title("Forced-recurrence positive control")
    plt.xlabel("Geodesic H1 maximum lifetime")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = OUT_DIR / "forced_recurrence_null_histogram.png"

    plt.savefig(
        fig_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved histogram      : {fig_path}")

    if not stats["formal_trigger"]:
        raise RuntimeError(
            "Forced-recurrence positive control failed: expected formal "
            "geodesic trigger, but got "
            f"pperm={stats['pperm']:.4f}, Lobs={stats['Lobs']:.6f}. "
            f"See {csv_path}"
        )

    elapsed = (time.time() - t0) / 60.0
    print(f"Finished in {elapsed:.1f} minutes.")

    return df


if __name__ == "__main__":
    main()
