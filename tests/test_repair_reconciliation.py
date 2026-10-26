#!/usr/bin/env python3
"""Validate the released numerical-repair ledger and preserved outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_common import sha256_file  # noqa: E402


LEDGER = ROOT / "exp4_results" / "appendix" / "path_fidelity" / "path_fidelity_repair_comparison_main.csv"
CERTIFICATE = ROOT / "exp4_results" / "appendix" / "path_fidelity" / "path_fidelity_certificate_main.csv"
BARS = ROOT / "exp4_results" / "appendix" / "path_fidelity" / "path_fidelity_h1_bars_main.csv"

KEYS = {
    "exp4_main": [
        "arch", "optimizer", "seed", "rep", "analysis_role", "space",
        "pca_d", "prefix_ep", "N_pts",
    ],
    "neural_sensitivity": ["arch", "optimizer", "seed", "rep", "k_value", "filtration_pct"],
    "block_robustness": ["arch", "optimizer", "seed", "rep", "block_size"],
    "exp5": [
        "experiment", "stress_name", "seed", "rep", "analysis_role", "space",
        "pca_d", "prefix_ep", "N_pts",
    ],
    "stress_appendix": [
        "experiment", "stress_name", "seed", "rep", "analysis_role", "space",
        "pca_d", "prefix_ep", "N_pts",
    ],
}

PRESERVED_TABLES = {
    "exp4_main": (
        ROOT / "exp4_results" / "exp4_results_full.csv",
        ROOT / "legacy_release_outputs" / "exp4_results" / "exp4_results_full.csv",
        534,
    ),
    "neural_sensitivity": (
        ROOT / "exp4_results" / "neural_sensitivity" / "neural_sensitivity_full.csv",
        ROOT / "legacy_release_outputs" / "exp4_results" / "neural_sensitivity" / "neural_sensitivity_full.csv",
        432,
    ),
    "block_robustness": (
        ROOT / "exp4_results" / "block_robustness" / "block_robustness_full.csv",
        ROOT / "legacy_release_outputs" / "exp4_results" / "block_robustness" / "block_robustness_full.csv",
        144,
    ),
    "exp5": (
        ROOT / "exp5_results" / "exp5_results_full.csv",
        ROOT / "legacy_release_outputs" / "exp5_results" / "exp5_results_full.csv",
        96,
    ),
    "stress_appendix": (
        ROOT / "exp5_results" / "appendix" / "table_stress_appendix.csv",
        ROOT / "legacy_release_outputs" / "exp5_results" / "appendix" / "table_stress_appendix.csv",
        96,
    ),
}

CHANGE_TOL = 1e-12


def require_file(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"required release file is missing: {path.relative_to(ROOT)}")


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise AssertionError(f"{label} is missing columns: {missing}")


def key_index(frame: pd.DataFrame, keys: list[str], label: str) -> pd.Index:
    require_columns(frame, keys, label)
    index = pd.MultiIndex.from_frame(frame[keys])
    if index.has_duplicates:
        raise AssertionError(f"{label} has duplicate row keys")
    return index


def compare_preserved_table(name: str, current_path: Path, legacy_path: Path, expected_rows: int) -> None:
    require_file(current_path)
    require_file(legacy_path)
    current = pd.read_csv(current_path)
    legacy = pd.read_csv(legacy_path)
    if len(current) != expected_rows or len(legacy) != expected_rows:
        raise AssertionError(
            f"{name}: expected {expected_rows} rows in both tables, "
            f"got {len(current)} and {len(legacy)}"
        )

    keys = KEYS[name]
    current_index = key_index(current, keys, f"current {name}")
    legacy_index = key_index(legacy, keys, f"legacy {name}")
    if set(current_index) != set(legacy_index):
        raise AssertionError(f"{name}: current and historical row keys differ")

    check_columns = [
        "geo_pperm",
        "geo_formal_trigger",
        "geo_nominal_trigger",
        "euc_pperm",
        "euc_formal_trigger",
        "euc_nominal_trigger",
        "geo_Lobs",
        "euc_Lobs",
        "geo_null_collapsed",
        "decision",
    ]
    check_columns = [
        column
        for column in check_columns
        if column in current.columns and column in legacy.columns
    ]
    if not check_columns:
        raise AssertionError(f"{name}: no inferential columns were available for comparison")

    current = current.set_index(keys).sort_index()
    legacy = legacy.set_index(keys).sort_index()
    for column in check_columns:
        left = current[column]
        right = legacy[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            left_text = left.astype("string").fillna("<NA>").to_numpy()
            right_text = right.astype("string").fillna("<NA>").to_numpy()
            equal = np.array_equal(left_text, right_text)
        if not equal:
            raise AssertionError(f"{name}: preserved column changed: {column}")

    print(f"{name}: {expected_rows} preserved rows and inferential columns identical")


def check_ledger() -> None:
    require_file(LEDGER)
    ledger = pd.read_csv(LEDGER)
    required = [
        "arch", "optimizer", "seed", "trajectory_file", "trajectory_sha256",
        "historical_reconstruction", "corrected_reconstruction",
        "historical_mu_G", "corrected_mu_G", "delta_mu_G",
        "historical_eta_G", "corrected_eta_G", "delta_eta_G",
        "historical_kappa_G", "corrected_kappa_G", "delta_kappa_G",
        "historical_mu_argmin_lag", "corrected_mu_argmin_lag",
        "historical_c_adj", "corrected_c_adj",
        "historical_missing_adjacent_edge_count",
        "corrected_missing_adjacent_edge_count",
        "corrected_reconstruction_exact", "corrected_recon_rel_err",
    ]
    require_columns(ledger, required, "repair ledger")
    if len(ledger) != 36:
        raise AssertionError(f"repair ledger must contain 36 rows, got {len(ledger)}")
    keys = key_index(ledger, ["arch", "optimizer", "seed"], "repair ledger")
    if len(keys) != 36:
        raise AssertionError("repair ledger does not contain 36 unique neural cells")
    if ledger["trajectory_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").sum() != 36:
        raise AssertionError("repair ledger must contain a SHA-256 value for every trajectory")

    certificate = pd.read_csv(CERTIFICATE)
    certificate = certificate[
        (certificate["group"].astype(str) == "neural")
        & (certificate["graph_mode"].astype(str) == "or_knn")
    ].copy()
    if len(certificate) != 36:
        raise AssertionError(f"corrected certificate must contain 36 neural OR-kNN rows, got {len(certificate)}")
    cert_index = key_index(certificate, ["arch", "optimizer", "seed"], "corrected certificate")
    if set(keys) != set(cert_index):
        raise AssertionError("repair ledger and corrected certificate keys differ")
    # Align both tables by their semantic neural-cell key before comparing
    # NumPy arrays.  The production certificate preserves its optimizer loop
    # order, whereas the provenance ledger is written in lexicographic order.
    certificate = certificate.set_index(["arch", "optimizer", "seed"]).sort_index()
    ledger = ledger.set_index(["arch", "optimizer", "seed"]).sort_index()

    for ledger_name, cert_name in (
        ("corrected_mu_G", "mu_G"),
        ("corrected_eta_G", "eta_G"),
        ("corrected_kappa_G", "kappa_G"),
        ("corrected_c_adj", "c_adj"),
        ("corrected_mu_argmin_lag", "mu_argmin_lag"),
        ("corrected_missing_adjacent_edge_count", "missing_adjacent_edge_count"),
        ("corrected_k_used", "k_used"),
    ):
        left = ledger[ledger_name].to_numpy(dtype=float)
        right = certificate[cert_name].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=2e-12, equal_nan=True):
            raise AssertionError(f"repair ledger does not reproduce certificate column {cert_name}")

    for column in ("corrected_reconstruction_exact", "corrected_eta_consistency_ok"):
        if column in ledger.columns and not ledger[column].astype(str).str.lower().eq("true").all():
            raise AssertionError(f"repair ledger has a false corrected invariant: {column}")
    if float(pd.to_numeric(ledger["corrected_recon_rel_err"]).max()) > 1e-12:
        raise AssertionError("corrected reconstruction residual exceeds tolerance")

    expected_change_counts = {
        "delta_mu_G": 14,
        "delta_eta_G": 34,
        "delta_kappa_G": 34,
    }
    expected_max_abs = {
        "delta_mu_G": 3.59e-5,
        "delta_eta_G": 5.79e-3,
        "delta_kappa_G": 8.15e-3,
    }

    # The released headline maxima above are recorded to three significant
    # figures.  Compare at that same precision rather than applying the
    # near-exact tolerance used for row-wise certificate equality.
    def assert_reported_max(actual: float, expected: float, label: str) -> None:
        actual_text = f"{actual:.2e}"
        expected_text = f"{expected:.2e}"
        if actual_text != expected_text:
            raise AssertionError(
                f"{label}: got {actual_text} (full value {actual!r}), "
                f"expected reported value {expected_text}"
            )

    for column, expected_count in expected_change_counts.items():
        values = pd.to_numeric(ledger[column]).to_numpy(dtype=float)
        actual_count = int(np.sum(np.abs(values) > CHANGE_TOL))
        if actual_count != expected_count:
            raise AssertionError(f"{column}: got {actual_count} changed rows, expected {expected_count}")
        assert_reported_max(
            float(np.max(np.abs(values))),
            expected_max_abs[column],
            f"max |{column}|",
        )

    if not np.array_equal(
        ledger["historical_mu_argmin_lag"].to_numpy(),
        ledger["corrected_mu_argmin_lag"].to_numpy(),
    ):
        raise AssertionError("the mu-minimising-edge lag changed during the repair")

    missing = ledger[ledger["corrected_missing_adjacent_edge_count"] == 1]
    missing_keys = set(missing.index.tolist())
    expected_missing = {
        ("mlp2_mnist", "adam", 1242),
        ("mlp2_mnist", "adamw", 2342),
    }
    if missing_keys != expected_missing:
        raise AssertionError(f"unexpected missing-adjacency rows: {missing_keys}")
    if int((ledger["corrected_missing_adjacent_edge_count"] > 0).sum()) != 2:
        raise AssertionError("expected exactly two corrected rows with a missing adjacent edge")

    fixed = certificate[certificate["k_used"] == 12]
    adaptive = certificate[certificate["k_used"] != 12]
    if len(fixed) != 32 or len(adaptive) != 4:
        raise AssertionError("unexpected fixed/adaptive k partition")
    fixed_lags = set(fixed["mu_argmin_lag"].astype(int))
    adaptive_lags = set(adaptive["mu_argmin_lag"].astype(int))
    if not fixed_lags or min(fixed_lags) < 2 or max(fixed_lags) > 10:
        raise AssertionError(f"fixed-k lag set is outside [2, 10]: {fixed_lags}")
    if adaptive_lags != {18, 33, 36, 51}:
        raise AssertionError(f"unexpected adaptive-growth lag set: {adaptive_lags}")
    if fixed_lags & adaptive_lags:
        raise AssertionError("fixed-k and adaptive-growth lag sets overlap")

    bars = pd.read_csv(BARS)
    if len(bars) != 51:
        raise AssertionError(f"expected 51 H1 bars, got {len(bars)}")
    if set(bars["bound_status"].astype(str)) != {"within_bound_confirmed"}:
        raise AssertionError("not all H1 bars are within_bound_confirmed")
    if bars["right_censored"].astype(str).str.lower().ne("false").any():
        raise AssertionError("a path-fidelity H1 bar is right-censored")
    print("repair ledger: 36 rows, expected deltas, adjacency exceptions, k-lag partition, and 51 bars validated")


def check_topology_identity() -> None:
    current = ROOT / "exp4_results" / "main_paper" / "table_main_topology.csv"
    historical = ROOT / "legacy_release_outputs" / "exp4_results" / "main_paper" / "table_main_topology.csv"
    require_file(current)
    require_file(historical)
    current_hash = sha256_file(current)
    historical_hash = sha256_file(historical)
    if current_hash != historical_hash:
        raise AssertionError("current and historical 36-cell topology tables are not byte-identical")
    print(f"topology table SHA-256 identical: {current_hash}")


def main() -> None:
    check_ledger()
    check_topology_identity()
    for name, (current, historical, rows) in PRESERVED_TABLES.items():
        compare_preserved_table(name, current, historical, rows)
    print("Repair reconciliation test passed: 1,302 preserved rows have no inferential change.")


if __name__ == "__main__":
    main()
