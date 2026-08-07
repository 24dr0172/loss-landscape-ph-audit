#!/usr/bin/env python3
"""Build release-wide inventory and trigger tables from archived CSV outputs.

Only rows explicitly identified as the primary matched-step analysis may become
formal triggers. Block, projection, sensitivity, control, benchmark, and static
Gaussian-null rows remain diagnostic even when a raw tail probability is below
alpha.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
ALPHA = 0.05

OUTPUT_ROOT_NAMES = (
    "results",
    "exp4_results",
    "exp5_results",
    "acf_results",
)

P_VALUE_PRIORITY = (
    "geo_pperm",
    "pperm",
    "p_perm",
    "perm_p",
    "p_value",
    "pvalue",
    "geo_pnull",
    "pnull",
)
FORMAL_FIELD_PRIORITY = (
    "geo_formal_trigger",
    "formal_trigger",
)
BLOCK_NAMES = {"block_size", "block", "b"}
ROLE_NAMES = {
    "analysis_role", "inference_role", "inferential_role", "role",
    "test_role", "analysis_type",
}
FORMAL_FLAG_NAMES = {"is_formal_inference", "is_primary_test", "formal_family"}

SUMMARY_KEYS = [
    "source_csv", "arch", "architecture", "model", "optimizer", "seed",
    "metric", "k", "k_value", "pct", "filtration_pct",
    "block_size_normalized", "n_nulls", "n_nulls_used",
    "analysis_role_normalized",
]


def first_matching_column(columns: Iterable[str], names: set[str]) -> str | None:
    lower_to_original = {str(c).lower(): str(c) for c in columns}
    for name in sorted(names):
        if name in lower_to_original:
            return lower_to_original[name]
    return None


def priority_column(columns: Iterable[str], priority: Iterable[str]) -> str | None:
    lower_to_original = {str(c).lower(): str(c) for c in columns}
    for name in priority:
        if name in lower_to_original:
            return lower_to_original[name]
    return None


def read_csvs() -> dict[str, pd.DataFrame]:
    files: list[Path] = []
    for name in OUTPUT_ROOT_NAMES:
        base = ROOT / name
        if base.exists():
            files.extend(p for p in base.rglob("*.csv") if p.is_file())

    out: dict[str, pd.DataFrame] = {}
    for path in sorted(set(files)):
        rel = str(path.relative_to(ROOT))
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Could not read {rel}: {exc}")
            continue
        df["source_csv"] = rel
        out[rel] = df
    return out


def _to_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "formal", "primary"}
    )


def _normalize_role_text(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    out = pd.Series("unclassified", index=series.index, dtype="object")

    out[text.str.contains("block", na=False)] = "block_diagnostic"
    out[text.str.contains("projection|pca", regex=True, na=False)] = "projection_diagnostic"
    out[text.str.contains("sensitivity|threshold|off.default|robust", regex=True, na=False)] = "metric_sensitivity"
    out[text.str.contains("control|synthetic|benchmark|manifold|archetype|acf", regex=True, na=False)] = "control_or_benchmark"
    out[
        text.str.contains(
            "formal.*matched|matched.*formal|primary.*matched|main.*neural|neural.*fullspace",
            regex=True,
            na=False,
        )
    ] = "formal_matched_step"
    return out


def _infer_role_from_source(df: pd.DataFrame) -> pd.Series:
    source = df.get(
        "source_csv",
        pd.Series([""] * len(df), index=df.index),
    ).astype(str).str.lower()
    role = pd.Series("unclassified", index=df.index, dtype="object")

    role[source.str.contains("block", na=False)] = "block_diagnostic"
    role[source.str.contains("projection|pca|acf", regex=True, na=False)] = "projection_diagnostic"
    role[
        source.str.contains(
            "sensitivity|threshold|k_sensitivity|robustness|off_default",
            regex=True,
            na=False,
        )
    ] = "metric_sensitivity"
    role[
        source.str.contains(
            "synthetic|control|benchmark|manifold|archetype|helix|circle|curvature|sanity",
            regex=True,
            na=False,
        )
    ] = "control_or_benchmark"
    role[
        source.str.contains(
            "main_paper/table_main_topology|neural_fullspace|fullspace_audit|main_neural|neural_audit",
            regex=True,
            na=False,
        )
    ] = "formal_matched_step"
    return role


def classify_analysis_role(df: pd.DataFrame) -> pd.Series:
    """Conservatively classify each row by inferential role."""
    role_col = first_matching_column(df.columns, ROLE_NAMES)
    formal_col = first_matching_column(df.columns, FORMAL_FLAG_NAMES)

    role = (
        _normalize_role_text(df[role_col])
        if role_col is not None
        else _infer_role_from_source(df)
    )

    # Row-level space is more reliable than a mixed file name.
    if "space" in df.columns:
        space = df["space"].astype(str).str.strip().str.lower()
        role.loc[space.eq("full")] = "formal_matched_step"
        role.loc[space.eq("pca")] = "projection_diagnostic"

    # Sensitivity-grid columns override generic full-space classification.
    if any(c in df.columns for c in ("k_value", "filtration_pct", "threshold_pct")):
        role.loc[:] = "metric_sensitivity"

    if formal_col is not None:
        explicit_formal = _to_bool_series(df[formal_col])
        role.loc[explicit_formal] = "formal_matched_step"
        role.loc[~explicit_formal & role.eq("formal_matched_step")] = "unclassified"

    # Experiment scope overrides a row-level formal flag. A b=1 positive
    # control may be a valid matched-step comparison within that control, but
    # it must not enter the release-wide inventory as a neural recurrence
    # trigger. The block override below still labels b>1 control rows as block
    # diagnostics.
    source = df.get(
        "source_csv",
        pd.Series([""] * len(df), index=df.index),
    ).astype(str).str.lower()
    control_source = source.str.contains(
        "synthetic|control|benchmark|manifold|archetype|helix|circle|curvature|sanity",
        regex=True,
        na=False,
    )
    role.loc[control_source] = "control_or_benchmark"

    block_col = first_matching_column(df.columns, BLOCK_NAMES)
    if block_col is not None:
        block_values = pd.to_numeric(df[block_col], errors="coerce")
        # A dedicated block-robustness file is diagnostic as a whole, including
        # its b=1 baseline row. Other files use b>1 as the diagnostic override.
        role.loc[source.str.contains("block", na=False)] = "block_diagnostic"
        role.loc[block_values > 1] = "block_diagnostic"

    return role


def normalize_formal_rule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pcol = priority_column(df.columns, P_VALUE_PRIORITY)
    bcol = first_matching_column(df.columns, BLOCK_NAMES)

    if pcol is not None:
        df["pvalue_source_column"] = pcol
        df["pperm_normalized"] = pd.to_numeric(df[pcol], errors="coerce")
        df["raw_pvalue_trigger"] = df["pperm_normalized"] < ALPHA
    else:
        df["pvalue_source_column"] = pd.NA
        df["pperm_normalized"] = pd.NA
        df["raw_pvalue_trigger"] = pd.NA

    if bcol is not None:
        df["block_size_normalized"] = pd.to_numeric(df[bcol], errors="coerce")
        df["is_block_gt1"] = df["block_size_normalized"] > 1
    else:
        df["block_size_normalized"] = pd.NA
        df["is_block_gt1"] = False

    df["analysis_role_normalized"] = classify_analysis_role(df)
    df["is_formal_matched_step_row"] = df["analysis_role_normalized"].eq(
        "formal_matched_step"
    )
    df["block_null_diagnostic"] = (
        df["raw_pvalue_trigger"].fillna(False)
        & df["analysis_role_normalized"].eq("block_diagnostic")
    )
    df["formal_trigger_normalized"] = (
        df["raw_pvalue_trigger"].fillna(False)
        & df["is_formal_matched_step_row"]
    )

    source_formal_col = priority_column(df.columns, FORMAL_FIELD_PRIORITY)
    if source_formal_col is not None:
        df["source_formal_column"] = source_formal_col
        df["source_formal_trigger_raw"] = df[source_formal_col]
        source_bool = _to_bool_series(df[source_formal_col])
        df["source_formal_trigger_mismatch"] = (
            df["pperm_normalized"].notna()
            & (source_bool != df["formal_trigger_normalized"])
        )
    else:
        df["source_formal_column"] = pd.NA
        df["source_formal_trigger_raw"] = pd.NA
        df["source_formal_trigger_mismatch"] = False

    return df


def release_inventory(all_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, raw in all_dfs.items():
        df = normalize_formal_rule(raw)
        rows.append({
            "source_csv": name,
            "n_rows": len(raw),
            "columns": ";".join(map(str, raw.columns)),
            "pvalue_column": (
                None if df.empty else df["pvalue_source_column"].iloc[0]
            ),
            "raw_pvalue_triggers": int(df["raw_pvalue_trigger"].fillna(False).sum()),
            "formal_matched_step_triggers": int(df["formal_trigger_normalized"].fillna(False).sum()),
            "block_diagnostic_triggers": int(df["block_null_diagnostic"].fillna(False).sum()),
            "unclassified_rows": int(df["analysis_role_normalized"].eq("unclassified").sum()),
            "source_formal_trigger_mismatches": int(df["source_formal_trigger_mismatch"].fillna(False).sum()),
        })
    return pd.DataFrame(rows)


def trigger_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in SUMMARY_KEYS if c in df.columns] or ["source_csv"]
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("source_csv", "size"),
            raw_pvalue_triggers=("raw_pvalue_trigger", lambda s: int(s.fillna(False).sum())),
            formal_triggers=("formal_trigger_normalized", lambda s: int(s.fillna(False).sum())),
            block_diagnostic_triggers=("block_null_diagnostic", lambda s: int(s.fillna(False).sum())),
            source_formal_trigger_mismatches=("source_formal_trigger_mismatch", lambda s: int(s.fillna(False).sum())),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write empty inventory files instead of failing when no output CSVs exist.",
    )
    args = parser.parse_args()

    TABLES.mkdir(exist_ok=True)
    all_dfs = read_csvs()
    if not all_dfs:
        message = "No CSV files found under the release output directories."
        if not args.allow_empty:
            raise SystemExit(message)
        pd.DataFrame(columns=["source_csv", "n_rows", "columns"]).to_csv(
            TABLES / "release_csv_inventory.csv", index=False
        )
        pd.DataFrame().to_csv(TABLES / "all_results_normalized.csv", index=False)
        pd.DataFrame().to_csv(TABLES / "trigger_summary.csv", index=False)
        print(message)
        return

    inventory = release_inventory(all_dfs)
    inventory.to_csv(TABLES / "release_csv_inventory.csv", index=False)

    normalized = pd.concat(
        [normalize_formal_rule(df) for df in all_dfs.values()],
        ignore_index=True,
        sort=False,
    )
    normalized.to_csv(TABLES / "all_results_normalized.csv", index=False)
    trigger_summary(normalized).to_csv(TABLES / "trigger_summary.csv", index=False)

    unclassified = int(normalized["analysis_role_normalized"].eq("unclassified").sum())
    if unclassified:
        print(
            f"[WARN] {unclassified} rows remain unclassified and cannot become "
            "formal triggers. Add an explicit analysis_role column if needed."
        )
    print(f"Wrote release-wide tables to {TABLES}")


if __name__ == "__main__":
    main()
