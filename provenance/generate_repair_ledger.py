#!/usr/bin/env python3
"""Generate the historical-versus-corrected path-fidelity repair ledger.

This is a provenance-only utility.  It is deliberately outside the production
rerun path and is not called by ``reproduce_all.sh``.  It requires the matching
Experiment 4 trajectory arrays because the public derived CSVs do not contain
the historical per-row certificate constants.

For each of the 36 neural trajectories, the utility computes the graph-
certificate quantities twice from the same increment Gram matrix:

* historical: the retained double-prefix inclusion--exclusion reconstruction;
* corrected: the shared diagonal-anchored reconstruction used in production.

Delta columns are ``corrected - historical``.  No PH calculation, permutation
null, formal decision, or training step is performed.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_code21():
    source = ROOT / "code_21_path_fidelity_certificate.py"
    spec = importlib.util.spec_from_file_location("code21_repair_provenance", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=ROOT / "exp4_results" / "trajectories",
        help="Directory containing the 36 matching Experiment 4 .npy arrays.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            ROOT
            / "exp4_results"
            / "appendix"
            / "path_fidelity"
            / "path_fidelity_repair_comparison_main.csv"
        ),
        help="Output path for the 36-row repair ledger.",
    )
    parser.add_argument("--k0", type=int, default=12)
    parser.add_argument(
        "--kmax",
        type=int,
        default=199,
        help="Maximum k for the frozen n=200 neural specification.",
    )
    parser.add_argument("--chunk-cols", type=int, default=50_000)
    parser.add_argument("--expected-count", type=int, default=36)
    parser.add_argument(
        "--no-hash-trajectories",
        action="store_true",
        help="Skip trajectory SHA-256 values; not appropriate for the final ledger.",
    )
    return parser.parse_args()


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _finite_float(value: object) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"Expected a finite certificate quantity, got {value!r}")
    return result


def legacy_summary(module, gram: np.ndarray, *, k0: int, kmax: int) -> dict:
    """Compute the historical certificate quantities without production gates."""
    step_lengths = module.shared_step_lengths_from_gram(gram)
    distance = module._legacy_distance_from_step_gram_prefix(gram)
    if not np.isfinite(distance).all():
        raise RuntimeError("Historical distance reconstruction produced non-finite values")

    graph, geodesic, graph_meta = module.adaptive_graph(
        distance,
        k0=k0,
        kmax=kmax,
        mode=module.MODE_KNN,
    )
    if not np.isfinite(geodesic).all():
        raise RuntimeError("Historical graph is disconnected at the requested kmax")

    rows, cols, weights = module.edge_arrays(graph)
    reference = module.path_metric_from_step_lengths(step_lengths)
    edge_reference = reference[rows, cols] if len(rows) else np.array([], dtype=float)
    if not len(weights) or not np.all(edge_reference > module.EPS):
        raise RuntimeError("Historical graph contains no usable positive reference edges")

    ratios = weights / edge_reference
    finite = np.isfinite(ratios)
    if not np.any(finite):
        raise RuntimeError("Historical edge ratios are non-finite")
    valid_indices = np.flatnonzero(finite)
    selected = int(valid_indices[int(np.argmin(ratios[finite]))])
    mu_raw = _finite_float(ratios[selected])
    mu_g = float(np.clip(mu_raw, 0.0, 1.0))

    edge_set = set(zip(rows.tolist(), cols.tolist()))
    adjacent_present = np.array(
        [(r, r + 1) in edge_set for r in range(len(step_lengths))],
        dtype=bool,
    )
    eta_values = np.diag(geodesic, k=1) / step_lengths
    if not np.isfinite(eta_values).all():
        raise RuntimeError("Historical eta values are non-finite")
    eta_g = _finite_float(np.max(eta_values))
    eta_argmax = int(np.argmax(eta_values))

    return {
        **graph_meta,
        "mu_raw": mu_raw,
        "mu_G": mu_g,
        "eta_G": eta_g,
        "kappa_G": _finite_float(eta_g / mu_g),
        "mu_argmin_i": int(rows[selected]),
        "mu_argmin_j": int(cols[selected]),
        "mu_argmin_lag": int(abs(int(cols[selected]) - int(rows[selected]))),
        "eta_argmax_index": eta_argmax,
        "adjacent_edge_count": int(np.sum(adjacent_present)),
        "missing_adjacent_edge_count": int(np.sum(~adjacent_present)),
        "c_adj": float(np.mean(adjacent_present)),
        "legacy_recon_rel_err": float(
            module.gram_reconstruction_error(gram, distance)
        ),
    }


def corrected_summary(module, gram: np.ndarray, *, k0: int, kmax: int) -> dict:
    """Call the released production certificate for the corrected quantities."""
    row, _graph, _geodesic, _reference, _locality = module.compute_certificate(
        gram,
        k0=k0,
        kmax=kmax,
        mode=module.MODE_KNN,
        shortcut_ratio=0.5,
    )
    if not bool(row["assessable"]):
        raise RuntimeError(
            "Corrected certificate is not assessable: "
            f"{row['not_assessable_reason']}"
        )
    if not bool(row["impl_reconstruction_exact"]):
        raise RuntimeError("Corrected reconstruction invariant is not exact")
    return row


def make_record(module, path: Path, *, k0: int, kmax: int, chunk_cols: int, hash_path: bool) -> dict:
    parsed = module.parse_trajectory_name(path)
    if parsed is None:
        raise RuntimeError(f"Could not parse trajectory filename: {path.name}")
    arch, optimizer, seed = parsed
    shape = np.load(path, mmap_mode="r").shape
    if len(shape) != 2:
        raise RuntimeError(f"Expected a 2D trajectory array: {path} has shape {shape}")
    n_points, d_full = map(int, shape)
    gram = module.step_gram_chunked(path, chunk_cols)

    historical = legacy_summary(module, gram, k0=k0, kmax=kmax)
    corrected = corrected_summary(module, gram, k0=k0, kmax=kmax)

    def delta(name: str) -> float:
        return float(corrected[name] - historical[name])

    record = {
        "group": "neural",
        "arch": arch,
        "optimizer": optimizer,
        "seed": int(seed),
        "n_points": n_points,
        "d_full": d_full,
        "k0": int(k0),
        "kmax": int(kmax),
        "trajectory_file": _relative_path(path),
        "trajectory_sha256": module.sha256_file(path) if hash_path else "",
        "historical_reconstruction": "legacy_double_prefix_inclusion_exclusion",
        "corrected_reconstruction": "audit_common_diagonal_anchored",
        "historical_mu_raw": historical["mu_raw"],
        "corrected_mu_raw": corrected["mu_raw"],
        "delta_mu_raw": delta("mu_raw"),
        "historical_mu_G": historical["mu_G"],
        "corrected_mu_G": corrected["mu_G"],
        "delta_mu_G": delta("mu_G"),
        "historical_eta_G": historical["eta_G"],
        "corrected_eta_G": corrected["eta_G"],
        "delta_eta_G": delta("eta_G"),
        "historical_kappa_G": historical["kappa_G"],
        "corrected_kappa_G": corrected["kappa_G"],
        "delta_kappa_G": delta("kappa_G"),
        "historical_mu_argmin_i": historical["mu_argmin_i"],
        "corrected_mu_argmin_i": corrected["mu_argmin_i"],
        "historical_mu_argmin_j": historical["mu_argmin_j"],
        "corrected_mu_argmin_j": corrected["mu_argmin_j"],
        "historical_mu_argmin_lag": historical["mu_argmin_lag"],
        "corrected_mu_argmin_lag": corrected["mu_argmin_lag"],
        "historical_eta_argmax_index": historical["eta_argmax_index"],
        "corrected_eta_argmax_index": corrected["eta_argmax_index"],
        "historical_k_used": historical["k_used"],
        "corrected_k_used": corrected["k_used"],
        "historical_c_adj": historical["c_adj"],
        "corrected_c_adj": corrected["c_adj"],
        "historical_adjacent_edge_count": historical["adjacent_edge_count"],
        "corrected_adjacent_edge_count": corrected["adjacent_edge_count"],
        "historical_missing_adjacent_edge_count": historical[
            "missing_adjacent_edge_count"
        ],
        "corrected_missing_adjacent_edge_count": corrected[
            "missing_adjacent_edge_count"
        ],
        "historical_recon_rel_err": historical["legacy_recon_rel_err"],
        "corrected_recon_rel_err": corrected["impl_recon_rel_err"],
        "corrected_reconstruction_exact": bool(corrected["impl_reconstruction_exact"]),
        "corrected_eta_consistency_ok": bool(corrected["impl_eta_consistency_ok"]),
        "corrected_assessable": bool(corrected["assessable"]),
    }
    del gram
    return record


def main() -> None:
    args = parse_args()
    module = load_code21()
    trajectories = module.iter_trajectory_files(args.traj_dir, "*__*__seed*.npy")
    if len(trajectories) != args.expected_count:
        raise SystemExit(
            f"Expected exactly {args.expected_count} Experiment 4 trajectories, "
            f"found {len(trajectories)} in {args.traj_dir}."
        )

    rows = []
    for index, (_arch, _optimizer, _seed, path) in enumerate(trajectories, start=1):
        print(f"[{index:02d}/{len(trajectories):02d}] {path.name}")
        rows.append(
            make_record(
                module,
                path,
                k0=args.k0,
                kmax=args.kmax,
                chunk_cols=args.chunk_cols,
                hash_path=not args.no_hash_trajectories,
            )
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["arch", "optimizer", "seed"], kind="mergesort")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} rows to {args.out}")
    print("Delta convention: corrected - historical")
    print(
        "No training, persistent homology, permutation nulls, or formal "
        "decisions were computed."
    )


if __name__ == "__main__":
    main()
