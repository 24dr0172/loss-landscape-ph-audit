#!/usr/bin/env python3
"""Build release tables from archived experiment CSVs.

The builder reconstructs the formal p-value trigger directly from the stored
permutation p-value whenever such a column is present. Rows with block size
larger than one are labelled as block-null diagnostics rather than formal
recurrence claims.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
ALPHA = 0.05

P_VALUE_NAMES = {"pperm", "p_perm", "perm_p", "p_value", "pvalue"}
BLOCK_NAMES = {"block_size", "block", "b"}
SUMMARY_KEYS = [
    "source_csv", "arch", "architecture", "model", "optimizer", "seed",
    "metric", "k", "pct", "block_size_normalized", "n_nulls",
]


def first_matching_column(columns: Iterable[str], names: set[str]) -> str | None:
    lower_to_original = {str(c).lower(): str(c) for c in columns}
    for name in sorted(names):
        if name in lower_to_original:
            return lower_to_original[name]
    return None


def read_csvs() -> dict[str, pd.DataFrame]:
    files = sorted(p for p in RESULTS.rglob("*.csv") if p.is_file()) if RESULTS.exists() else []
    out: dict[str, pd.DataFrame] = {}
    for path in files:
        rel = str(path.relative_to(ROOT))
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Could not read {rel}: {exc}")
            continue
        df["source_csv"] = rel
        out[rel] = df
    return out


def normalize_formal_rule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pcol = first_matching_column(df.columns, P_VALUE_NAMES)
    bcol = first_matching_column(df.columns, BLOCK_NAMES)

    if pcol is not None:
        df["pperm_normalized"] = pd.to_numeric(df[pcol], errors="coerce")
        df["raw_pperm_trigger"] = df["pperm_normalized"] < ALPHA
    else:
        df["pperm_normalized"] = pd.NA
        df["raw_pperm_trigger"] = pd.NA

    if bcol is not None:
        df["block_size_normalized"] = pd.to_numeric(df[bcol], errors="coerce")
        df["is_block_diagnostic"] = df["block_size_normalized"] > 1
    else:
        df["block_size_normalized"] = pd.NA
        source = df.get("source_csv", pd.Series([""] * len(df), index=df.index)).astype(str).str.lower()
        df["is_block_diagnostic"] = source.str.contains("block")

    df["block_null_diagnostic"] = (
        df["raw_pperm_trigger"].fillna(False) & df["is_block_diagnostic"].fillna(False)
    )
    df["formal_trigger_normalized"] = (
        df["raw_pperm_trigger"].fillna(False) & ~df["is_block_diagnostic"].fillna(False)
    )

    if "formal_trigger" in df.columns:
        df["source_formal_trigger_raw"] = df["formal_trigger"]
        source_bool = df["source_formal_trigger_raw"].astype(str).str.lower().isin({"true", "1", "yes"})
        df["source_formal_trigger_mismatch"] = df["raw_pperm_trigger"].notna() & (source_bool != df["raw_pperm_trigger"])
    else:
        df["source_formal_trigger_raw"] = pd.NA
        df["source_formal_trigger_mismatch"] = False

    return df


def release_inventory(all_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, raw in all_dfs.items():
        df = normalize_formal_rule(raw)
        row = {
            "source_csv": name,
            "n_rows": len(raw),
            "columns": ";".join(map(str, raw.columns)),
            "has_pperm": first_matching_column(raw.columns, P_VALUE_NAMES) is not None,
            "has_block_column": first_matching_column(raw.columns, BLOCK_NAMES) is not None,
            "raw_pperm_triggers": int(df["raw_pperm_trigger"].fillna(False).sum()),
            "formal_b1_or_nonblock_triggers": int(df["formal_trigger_normalized"].fillna(False).sum()),
            "block_diagnostic_triggers": int(df["block_null_diagnostic"].fillna(False).sum()),
            "source_formal_trigger_mismatches": int(df["source_formal_trigger_mismatch"].fillna(False).sum()),
        }
        for key in ["seed", "model", "arch", "architecture", "optimizer", "config", "metric", "k", "pct", "n_nulls", "block_size"]:
            hits = [c for c in raw.columns if str(c).lower() == key]
            if hits:
                row[f"n_unique_{key}"] = raw[hits[0]].nunique(dropna=True)
        rows.append(row)
    return pd.DataFrame(rows)


def trigger_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in SUMMARY_KEYS if c in df.columns]
    if not group_cols:
        group_cols = ["source_csv"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("source_csv", "size"),
            raw_pperm_triggers=("raw_pperm_trigger", lambda s: int(s.fillna(False).sum())),
            formal_triggers=("formal_trigger_normalized", lambda s: int(s.fillna(False).sum())),
            block_diagnostic_triggers=("block_null_diagnostic", lambda s: int(s.fillna(False).sum())),
            source_formal_trigger_mismatches=("source_formal_trigger_mismatch", lambda s: int(s.fillna(False).sum())),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-empty", action="store_true", help="Write empty inventory files instead of failing when results/ has no CSV files.")
    args = parser.parse_args()

    TABLES.mkdir(exist_ok=True)
    all_dfs = read_csvs()
    if not all_dfs:
        message = "No CSV files found under results/. Run the experiments before building manuscript tables."
        if not args.allow_empty:
            raise SystemExit(message)
        pd.DataFrame(columns=["source_csv", "n_rows", "columns"]).to_csv(TABLES / "release_csv_inventory.csv", index=False)
        pd.DataFrame().to_csv(TABLES / "all_results_normalized.csv", index=False)
        pd.DataFrame().to_csv(TABLES / "trigger_summary.csv", index=False)
        print(message)
        return

    inventory = release_inventory(all_dfs)
    inventory.to_csv(TABLES / "release_csv_inventory.csv", index=False)

    normalized = pd.concat([normalize_formal_rule(df) for df in all_dfs.values()], ignore_index=True, sort=False)
    normalized.to_csv(TABLES / "all_results_normalized.csv", index=False)
    trigger_summary(normalized).to_csv(TABLES / "trigger_summary.csv", index=False)

    print(f"Wrote release tables to {TABLES}")


if __name__ == "__main__":
    main()
