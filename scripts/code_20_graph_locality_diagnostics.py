#!/usr/bin/env python3
"""Graph-locality diagnostics for the main neural trajectory audit.

Purpose
-------
The manuscript uses an adaptive k-NN graph-geodesic metric: start at k0 and
increase k until the symmetrized k-NN graph is connected. This script records
whether that adaptive rule stayed local for the 36 main neural trajectories.

It does not compute persistent homology, null distributions, or recurrence
claims. It only audits the observed trajectory graphs used by the main neural
experiment.

Outputs
-------
Default outputs are written to exp4_results/appendix/:
    graph_locality_diagnostics_main.csv
    graph_locality_summary_by_arch_optimizer.csv
    graph_locality_diagnostics_main.tex

Run from repository root:
    python scripts/code_20_graph_locality_diagnostics.py

Optional:
    python scripts/code_20_graph_locality_diagnostics.py \
        --traj-dir exp4_results/trajectories \
        --main-results exp4_results/main_paper/table_main_topology.csv
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# BLAS/OpenMP thread pinning. Keep this before NumPy/SciPy-heavy work.
# ---------------------------------------------------------------------------
import os as _os
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_key, "1")

import argparse
import gc
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

# Shared graph-construction convention. Do not rebuild kNN logic locally.
# Repository import path. This must appear before importing audit_common.
import sys
from pathlib import Path as _RepoPath

_REPO_ROOT = _RepoPath(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audit_common import symmetrized_knn_graph_from_distance_matrix, sha256_file


ARCH_ORDER = {
    "mlp2_mnist": 0,
    "smallcnn_cifar10": 1,
    "resnet18_cifar10": 2,
}
OPT_ORDER = {
    "sgd": 0,
    "sgd_momentum": 1,
    "adam": 2,
    "adamw": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=Path("exp4_results") / "trajectories",
        help="Directory containing neural trajectory .npy files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exp4_results") / "appendix",
        help="Directory where diagnostic CSV/TEX outputs are written.",
    )
    parser.add_argument(
        "--main-results",
        type=Path,
        default=Path("exp4_results") / "main_paper" / "table_main_topology.csv",
        help="Optional main topology table to merge p-value/Lobs/decision columns.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*__*__seed*.npy",
        help="Glob pattern for trajectory arrays.",
    )
    parser.add_argument(
        "--k0",
        type=int,
        default=12,
        help="Initial k used by the main neural graph-geodesic audit.",
    )
    parser.add_argument(
        "--kmax",
        type=int,
        default=0,
        help="Maximum k. Use 0 to mean n_points - 1, matching the main neural run.",
    )
    parser.add_argument(
        "--chunk-cols",
        type=int,
        default=50_000,
        help="Column chunk size for high-dimensional pairwise distances.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=36,
        help="Expected number of main neural trajectories. Set 0 to disable the warning.",
    )
    parser.add_argument(
        "--write-latex",
        action="store_true",
        default=True,
        help="Write a compact LaTeX table for manuscript appendix use.",
    )
    return parser.parse_args()


def parse_trajectory_name(path: Path) -> tuple[str, str, int] | None:
    """Parse names like arch__optimizer__seed2042.npy.

    Metrics files such as *__metrics.npy are ignored.
    """
    if path.name.endswith("__metrics.npy"):
        return None
    m = re.match(r"^(?P<arch>.+?)__(?P<optimizer>.+?)__seed(?P<seed>\d+)\.npy$", path.name)
    if not m:
        return None
    return m.group("arch"), m.group("optimizer"), int(m.group("seed"))


def iter_trajectory_files(traj_dir: Path, pattern: str) -> list[tuple[str, str, int, Path]]:
    rows: list[tuple[str, str, int, Path]] = []
    for path in sorted(traj_dir.glob(pattern)):
        parsed = parse_trajectory_name(path)
        if parsed is None:
            continue
        arch, opt, seed = parsed
        rows.append((arch, opt, seed, path))
    rows.sort(key=lambda r: (ARCH_ORDER.get(r[0], 99), OPT_ORDER.get(r[1], 99), r[2]))
    return rows


def pairwise_distance_chunked(path: Path, chunk_cols: int) -> np.ndarray:
    """Observed checkpoint distances via a streamed float64 increment Gram."""
    X = np.load(path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D trajectory array, got shape {X.shape} for {path}")
    n, d = X.shape
    G = np.zeros((n - 1, n - 1), dtype=np.float64)
    for start in range(0, d, chunk_cols):
        end = min(start + chunk_cols, d)
        Xc = np.asarray(X[:, start:end], dtype=np.float64)
        steps = np.diff(Xc, axis=0)
        G += steps @ steps.T
        del Xc, steps
    G = 0.5 * (G + G.T)
    prefix = np.pad(np.cumsum(np.cumsum(G, axis=0), axis=1), ((1, 0), (1, 0)))
    diag = np.diag(prefix)
    D2 = diag[:, None] + diag[None, :] - prefix - prefix.T
    D2 = 0.5 * (D2 + D2.T)
    D = np.sqrt(np.maximum(D2, 0.0))
    np.fill_diagonal(D, 0.0)
    return D

def upper_triangle_values_from_sparse_graph(G) -> np.ndarray:
    """Return one copy of each undirected edge weight from a symmetric sparse graph."""
    C = G.tocoo()
    mask = C.row < C.col
    return np.asarray(C.data[mask], dtype=float)


def dense_upper_values(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(D, k=1)
    vals = np.asarray(D[iu], dtype=float)
    return vals[np.isfinite(vals)]


def graph_diagnostics_from_distance_matrix(
    D: np.ndarray,
    *,
    k0: int,
    kmax: int,
) -> dict:
    """Adaptive-k graph diagnostics using the shared audit graph convention."""
    n = int(D.shape[0])
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be square.")
    if n < 2:
        raise ValueError("Need at least two trajectory checkpoints.")

    hard_max = min(n - 1, int(kmax) if int(kmax) > 0 else n - 1)
    k_initial = min(max(1, int(k0)), hard_max)

    all_pair = dense_upper_values(D)
    if all_pair.size == 0:
        raise ValueError("No finite off-diagonal pairwise distances were found.")

    # Check the initial graph separately so the table can report whether
    # adaptive growth was required.
    G0 = symmetrized_knn_graph_from_distance_matrix(D, k_initial)
    n_comp0, _ = connected_components(G0, directed=False, return_labels=True)
    connected_at_initial = bool(n_comp0 == 1)

    k = k_initial
    G_final = G0
    n_comp = int(n_comp0)
    while n_comp != 1:
        if k >= hard_max:
            break
        k = min(2 * k, hard_max)
        G_final = symmetrized_knn_graph_from_distance_matrix(D, k)
        n_comp, _ = connected_components(G_final, directed=False, return_labels=True)

    connected_final = bool(n_comp == 1)
    edge_weights = upper_triangle_values_from_sparse_graph(G_final)
    edge_count = int(edge_weights.size)
    possible_edges = n * (n - 1) / 2.0
    graph_density = edge_count / possible_edges if possible_edges > 0 else np.nan

    edge_median = float(np.median(edge_weights)) if edge_weights.size else np.nan
    edge_p95 = float(np.percentile(edge_weights, 95)) if edge_weights.size else np.nan
    edge_p99 = float(np.percentile(edge_weights, 99)) if edge_weights.size else np.nan
    edge_max = float(np.max(edge_weights)) if edge_weights.size else np.nan

    all_median = float(np.median(all_pair))
    all_p95 = float(np.percentile(all_pair, 95))
    all_p99 = float(np.percentile(all_pair, 99))
    all_max = float(np.max(all_pair))

    eps = 1e-12
    return {
        "n_points": n,
        "k_initial": int(k_initial),
        "k_used": int(k),
        "kmax": int(hard_max),
        "k_used_over_n_minus_1": float(k / max(n - 1, 1)),
        "adaptive_growth_used": bool(k > k_initial),
        "connected_at_initial_k": connected_at_initial,
        "initial_components": int(n_comp0),
        "connected_at_final_k": connected_final,
        "final_components": int(n_comp),
        "edge_count": edge_count,
        "possible_edges": int(possible_edges),
        "graph_density": float(graph_density),
        "edge_median": edge_median,
        "edge_p95": edge_p95,
        "edge_p99": edge_p99,
        "edge_max": edge_max,
        "edge_max_over_median": float(edge_max / (edge_median + eps)) if edge_weights.size else np.nan,
        "allpair_median": all_median,
        "allpair_p95": all_p95,
        "allpair_p99": all_p99,
        "allpair_max": all_max,
        "edge_max_over_allpair_median": float(edge_max / (all_median + eps)) if edge_weights.size else np.nan,
        "edge_max_over_allpair_p95": float(edge_max / (all_p95 + eps)) if edge_weights.size else np.nan,
        "edge_p95_over_allpair_p95": float(edge_p95 / (all_p95 + eps)) if edge_weights.size else np.nan,
        "near_complete_graph_flag": bool((graph_density >= 0.5) or (k / max(n - 1, 1) >= 0.5)),
    }


def load_main_results(path: Path) -> pd.DataFrame | None:
    """Load manuscript-facing main table for optional result annotation."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"arch", "optimizer", "seed"}
    if not required.issubset(df.columns):
        return None

    keep = ["arch", "optimizer", "seed"]
    optional = [
        "geo_Lobs", "geo_null_med", "geo_null_max", "geo_pperm",
        "geo_formal_trigger", "geo_zero_frac", "geo_null_collapsed",
        "decision", "kinematic_R", "val_acc", "d_full",
    ]
    keep.extend([c for c in optional if c in df.columns])
    return df[keep].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["arch", "optimizer"], dropna=False)
    out = grouped.agg(
        n_paths=("seed", "count"),
        connected_initial_count=("connected_at_initial_k", "sum"),
        adaptive_growth_count=("adaptive_growth_used", "sum"),
        k_used_min=("k_used", "min"),
        k_used_median=("k_used", "median"),
        k_used_max=("k_used", "max"),
        graph_density_median=("graph_density", "median"),
        graph_density_max=("graph_density", "max"),
        edge_max_over_median_median=("edge_max_over_median", "median"),
        edge_max_over_median_max=("edge_max_over_median", "max"),
        edge_max_over_allpair_p95_max=("edge_max_over_allpair_p95", "max"),
        near_complete_graph_count=("near_complete_graph_flag", "sum"),
    ).reset_index()
    return out


def write_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    """Compact appendix table; use the CSV for the full 36-row audit."""
    if summary.empty:
        return
    tex_df = summary.copy()
    tex_df = tex_df.rename(columns={
        "arch": "Architecture",
        "optimizer": "Optimizer",
        "n_paths": "Paths",
        "connected_initial_count": "Conn. at $k_0$",
        "adaptive_growth_count": "Growth",
        "k_used_min": "$k_{\\min}$",
        "k_used_median": "$k_{\\mathrm{med}}$",
        "k_used_max": "$k_{\\max}$",
        "graph_density_median": "Med. density",
        "graph_density_max": "Max density",
        "near_complete_graph_count": "Near-complete",
    })
    cols = [
        "Architecture", "Optimizer", "Paths", "Conn. at $k_0$", "Growth",
        "$k_{\\min}$", "$k_{\\mathrm{med}}$", "$k_{\\max}$",
        "Med. density", "Max density", "Near-complete",
    ]
    tex_df = tex_df[[c for c in cols if c in tex_df.columns]]
    for col in ["Med. density", "Max density"]:
        if col in tex_df.columns:
            tex_df[col] = tex_df[col].map(lambda x: f"{float(x):.4f}")
    out_path.write_text(
        tex_df.to_latex(index=False, escape=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    t_start = time.time()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    trajectories = iter_trajectory_files(args.traj_dir, args.pattern)
    if not trajectories:
        raise SystemExit(
            f"No trajectory .npy files found in {args.traj_dir} with pattern {args.pattern}. "
            "Run code_10 first or point --traj-dir to the saved trajectory archive."
        )

    if args.expected_count and len(trajectories) != args.expected_count:
        print(
            f"[warn] Expected {args.expected_count} trajectories, found {len(trajectories)}. "
            "Continuing so partial archives can still be audited."
        )

    print("=" * 78)
    print("Graph-locality diagnostics for main neural trajectories")
    print("=" * 78)
    print(f"Trajectory directory : {args.traj_dir}")
    print(f"Trajectory count     : {len(trajectories)}")
    print(f"k0                   : {args.k0}")
    print(f"kmax                 : {'n_points-1' if args.kmax == 0 else args.kmax}")
    print(f"chunk_cols           : {args.chunk_cols}")
    print("This script computes observed-trajectory graph diagnostics only; no PH/nulls.")
    print("=" * 78)

    rows: list[dict] = []
    for idx, (arch, opt, seed, path) in enumerate(trajectories, start=1):
        one_t0 = time.time()
        X_mmap = np.load(path, mmap_mode="r")
        n_points, d_full = map(int, X_mmap.shape)
        del X_mmap

        print(
            f"[{idx:02d}/{len(trajectories):02d}] {arch}/{opt}/seed{seed} "
            f"shape=({n_points}, {d_full})"
        )

        D = pairwise_distance_chunked(path, args.chunk_cols)
        diag = graph_diagnostics_from_distance_matrix(
            D,
            k0=args.k0,
            kmax=(args.kmax if args.kmax > 0 else n_points - 1),
        )
        del D
        gc.collect()

        row = {
            "arch": arch,
            "optimizer": opt,
            "seed": seed,
            "trajectory_file": str(path),
            "trajectory_sha256": sha256_file(path),
            "d_full": d_full,
            **diag,
            "elapsed_sec": time.time() - one_t0,
        }
        rows.append(row)

        print(
            f"    k_used={diag['k_used']}  "
            f"growth={diag['adaptive_growth_used']}  "
            f"density={diag['graph_density']:.4f}  "
            f"edge_max/edge_med={diag['edge_max_over_median']:.3f}  "
            f"edge_max/all_p95={diag['edge_max_over_allpair_p95']:.3f}  "
            f"near_complete={diag['near_complete_graph_flag']}  "
            f"({row['elapsed_sec']:.1f}s)"
        )

    df = pd.DataFrame(rows)

    main_results = load_main_results(args.main_results)
    if main_results is not None:
        df = df.merge(
            main_results,
            on=["arch", "optimizer", "seed"],
            how="left",
            suffixes=("", "_main"),
        )
    else:
        print(f"[info] Main topology table not found or not mergeable: {args.main_results}")

    # Stable column order for appendix inspection.
    first_cols = [
        "arch", "optimizer", "seed", "n_points", "d_full",
        "k_initial", "k_used", "kmax", "k_used_over_n_minus_1",
        "connected_at_initial_k", "initial_components",
        "connected_at_final_k", "final_components", "adaptive_growth_used",
        "edge_count", "possible_edges", "graph_density",
        "edge_median", "edge_p95", "edge_p99", "edge_max",
        "edge_max_over_median", "allpair_median", "allpair_p95",
        "allpair_p99", "allpair_max", "edge_max_over_allpair_median",
        "edge_max_over_allpair_p95", "edge_p95_over_allpair_p95",
        "near_complete_graph_flag",
        "geo_Lobs", "geo_pperm", "geo_formal_trigger", "decision",
        "kinematic_R", "val_acc", "trajectory_file", "trajectory_sha256", "elapsed_sec",
    ]
    ordered_cols = [c for c in first_cols if c in df.columns] + [
        c for c in df.columns if c not in first_cols
    ]
    df = df[ordered_cols]

    out_csv = args.out_dir / "graph_locality_diagnostics_main.csv"
    df.to_csv(out_csv, index=False)

    summary = summarize(df)
    out_summary = args.out_dir / "graph_locality_summary_by_arch_optimizer.csv"
    summary.to_csv(out_summary, index=False)

    if args.write_latex:
        out_tex = args.out_dir / "graph_locality_diagnostics_main.tex"
        write_latex_table(summary, out_tex)
    else:
        out_tex = None

    print("=" * 78)
    print(f"Wrote full 36-row diagnostics : {out_csv}")
    print(f"Wrote grouped summary         : {out_summary}")
    if out_tex is not None:
        print(f"Wrote compact LaTeX summary   : {out_tex}")
    print("\nKey audit counts:")
    print(f"  connected at initial k0 : {int(df['connected_at_initial_k'].sum())}/{len(df)}")
    print(f"  adaptive growth used    : {int(df['adaptive_growth_used'].sum())}/{len(df)}")
    print(f"  near-complete flags     : {int(df['near_complete_graph_flag'].sum())}/{len(df)}")
    print(f"  max k_used              : {int(df['k_used'].max())}")
    print(f"  max graph density       : {float(df['graph_density'].max()):.4f}")
    print(f"Complete in {(time.time() - t_start) / 60.0:.1f} min")
    print("=" * 78)


if __name__ == "__main__":
    main()
