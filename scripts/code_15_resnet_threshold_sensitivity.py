#!/usr/bin/env python3
"""Filtration-percentile sensitivity for the three ResNet-18/SGD trajectories.

This is a diagnostic-only rerun of the historical Stage E calculation.  It
does not define recurrence evidence: the formal neural result remains the
matched-step b=1 test in ``code_10_neural_trajectory_pipeline.py`` at the
frozen 95-percentile setting.  The script exists because the old release
contained its outputs but accidentally omitted its source.

For each ResNet-18/SGD trajectory and each requested percentile, the observed
trajectory and 50 matched-step permutations are analysed with the same
increment-Gram distance routine used by the main audit.  The null seeds match
the ResNet branch of code_10's sensitivity stage so the calculation is
reproducible from the supplied trajectories.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_common import (  # noqa: E402
    block_permutation_indices,
    compute_stats,
    distance_from_step_gram,
    geodesic_h1_lifetime,
    sha256_file,
)


ARCHITECTURE = "resnet18_cifar10"
OPTIMIZER = "sgd"
DEFAULT_SEEDS = (20042, 21042, 22042)
DEFAULT_PCTS = (90.0, 95.0, 99.0)
DEFAULT_K0 = 12
DEFAULT_N_NULLS = 50
DEFAULT_CHUNK_COLS = 20_000
ALPHA = 0.05
EPS = 1e-12


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=Path("exp4_results") / "trajectories",
        help="Directory containing the three cached ResNet-18/SGD arrays.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exp4_results") / "threshold_sensitivity_resnet18_sgd",
        help="Directory for regenerated diagnostic CSV and TeX outputs.",
    )
    parser.add_argument("--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--percentiles", type=parse_csv_floats, default=DEFAULT_PCTS)
    parser.add_argument("--n-nulls", type=int, default=DEFAULT_N_NULLS)
    parser.add_argument("--k0", type=int, default=DEFAULT_K0)
    parser.add_argument(
        "--kmax",
        type=int,
        default=0,
        help="Maximum adaptive k; 0 resolves to n_checkpoints - 1.",
    )
    parser.add_argument("--chunk-cols", type=int, default=DEFAULT_CHUNK_COLS)
    parser.add_argument(
        "--hash-trajectories",
        action="store_true",
        help="Record each array SHA-256 (extra I/O for large files).",
    )
    return parser.parse_args()


def trajectory_path(traj_dir: Path, seed: int) -> Path:
    return traj_dir / f"{ARCHITECTURE}__{OPTIMIZER}__seed{seed}.npy"


def step_gram_chunked(path: Path, chunk_cols: int) -> np.ndarray:
    """Accumulate the float64 increment Gram matrix without loading all columns."""
    if chunk_cols <= 0:
        raise ValueError("chunk_cols must be positive")
    X = np.load(path, mmap_mode="r")
    if X.ndim != 2 or X.shape[0] < 4:
        raise ValueError(f"Expected a 2D trajectory with at least four checkpoints: {path}")
    n_points, d_full = map(int, X.shape)
    gram = np.zeros((n_points - 1, n_points - 1), dtype=np.float64)
    for start in range(0, d_full, chunk_cols):
        stop = min(start + chunk_cols, d_full)
        chunk = np.asarray(X[:, start:stop], dtype=np.float64)
        steps = np.diff(chunk, axis=0)
        gram += steps @ steps.T
        del chunk, steps
    return 0.5 * (gram + gram.T)


def geodesic_lifetime(
    ambient_distance: np.ndarray,
    *,
    k0: int,
    kmax: int,
    percentile: float,
) -> tuple[float, int]:
    """Run the same precomputed-distance graph-geodesic PH path as code_10."""
    lifetime, k_used = geodesic_h1_lifetime(
        ambient_distance,
        input_distance_matrix=True,
        k0=k0,
        kmax=kmax,
        pct=percentile,
        eps=EPS,
        sentinel_fill=False,
        return_k=True,
        complete_filtration=False,
    )
    return float(lifetime), int(k_used)


def sensitivity_null_seed(seed: int, k0: int, percentile: float) -> int:
    """Match code_10's ResNet sensitivity seed convention exactly."""
    return int(seed + 9000 + 100 * int(k0) + int(round(percentile)))


def decision_from_stats(stats: dict) -> str:
    return "sensitivity_trigger" if bool(stats["formal_trigger"]) else "sensitivity_non_trigger"


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    result_path = out_dir / "threshold_sensitivity_results.csv"
    summary_path = out_dir / "threshold_sensitivity_summary.csv"
    tex_path = out_dir / "threshold_sensitivity_results.tex"
    frame.to_csv(result_path, index=False)

    summary = (
        frame.groupby("threshold_pct", as_index=False)
        .agg(
            n_rows=("seed", "count"),
            n_positive=("formal_trigger", "sum"),
            positive_rate=("formal_trigger", "mean"),
            mean_pperm=("pperm", "mean"),
            mean_zrob=("zrob", "mean"),
            mean_Lobs=("Lobs", "mean"),
        )
        .sort_values("threshold_pct")
    )
    summary.to_csv(summary_path, index=False)

    display = frame[
        ["seed", "threshold_pct", "Lobs", "null_med", "null_max", "pperm", "zrob", "decision"]
    ].copy()
    display.to_latex(tex_path, index=False, float_format=lambda value: f"{value:.6g}")
    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {tex_path}")


def main() -> None:
    args = parse_args()
    if args.n_nulls <= 0:
        raise SystemExit("--n-nulls must be positive")
    if not all(0.0 < pct <= 100.0 for pct in args.percentiles):
        raise SystemExit("All --percentiles values must lie in (0, 100].")

    rows: list[dict] = []
    start_time = time.time()
    print("=" * 78)
    print("ResNet-18/SGD filtration-percentile sensitivity (diagnostic only)")
    print(f"seeds={args.seeds}; percentiles={args.percentiles}; n_nulls={args.n_nulls}")
    print("Formal b=1 recurrence evidence remains code_10 at the frozen 95% setting.")
    print("=" * 78)

    for seed in args.seeds:
        path = trajectory_path(args.traj_dir, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached trajectory: {path}")
        shape = np.load(path, mmap_mode="r").shape
        n_points, d_full = map(int, shape)
        kmax = min(n_points - 1, args.kmax if args.kmax > 0 else n_points - 1)
        if kmax < 1:
            raise ValueError(f"Need at least two checkpoints: {path}")
        print(f"\nseed={seed} shape=({n_points}, {d_full}) k0={args.k0} kmax={kmax}")
        gram = step_gram_chunked(path, args.chunk_cols)
        observed_distance = distance_from_step_gram(gram)

        for percentile in args.percentiles:
            observed, k_used_obs = geodesic_lifetime(
                observed_distance, k0=args.k0, kmax=kmax, percentile=percentile
            )
            null_seed = sensitivity_null_seed(seed, args.k0, percentile)
            null_lifetimes = np.empty(args.n_nulls, dtype=np.float64)
            null_k = np.empty(args.n_nulls, dtype=np.int64)
            for null_index in range(args.n_nulls):
                rng = np.random.default_rng([null_seed + 100, null_index, 1])
                perm = block_permutation_indices(n_points - 1, 1, rng)
                null_distance = distance_from_step_gram(gram, perm=perm)
                null_lifetimes[null_index], null_k[null_index] = geodesic_lifetime(
                    null_distance, k0=args.k0, kmax=kmax, percentile=percentile
                )

            stats = compute_stats(observed, null_lifetimes, alpha=ALPHA)
            row = {
                "architecture": ARCHITECTURE,
                "optimizer": OPTIMIZER,
                "seed": int(seed),
                "threshold_pct": float(percentile),
                "GEODESIC_K": int(args.k0),
                "GEODESIC_MAX_K": int(kmax),
                "N_NULL": int(args.n_nulls),
                "SUBSAMPLE": int(n_points),
                "CHUNK_COLS": int(args.chunk_cols),
                "ALPHA": ALPHA,
                "analysis_role": "metric_sensitivity",
                "is_formal_inference": False,
                "is_default_threshold": bool(np.isclose(percentile, 95.0)),
                "k_used_obs": int(k_used_obs),
                "null_k_min": int(np.min(null_k)),
                "null_k_median": float(np.median(null_k)),
                "null_k_max": int(np.max(null_k)),
                "d_full": int(d_full),
                "trajectory_file": str(path),
                "trajectory_sha256": sha256_file(path) if args.hash_trajectories else "",
                **stats,
            }
            row["decision"] = decision_from_stats(stats)
            rows.append(row)
            print(
                f"  pct={percentile:5.1f} Lobs={observed:.6g} "
                f"p={stats['pperm']:.6g} k={k_used_obs} "
                f"decision={row['decision']}"
            )

        del gram, observed_distance

    write_outputs(rows, args.out_dir)
    print(f"Elapsed: {(time.time() - start_time) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
