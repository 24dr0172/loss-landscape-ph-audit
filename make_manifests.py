#!/usr/bin/env python3
"""Create SHA-256 manifests for release artifacts and support files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_common import sha256_file

ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "manifests"
MANIFEST_DIR.mkdir(exist_ok=True)

TOP_LEVEL_PATTERNS = [
    "audit_common.py",
    "environment*.yml",
    "requirements*.txt",
    "README.md",
    "reproduce_all.sh",
    "make_manuscript_tables.py",
    "make_manifests.py",
    "lint_syntax.py",
    "verify_release_consistency.py",
    "sitecustomize.py",
]


def write_manifest(rows: list[dict], out_name: str) -> None:
    pd.DataFrame(rows, columns=["relative_path", "sha256", "size_bytes"]).to_csv(
        MANIFEST_DIR / out_name, index=False
    )


def rows_for_paths(paths: list[Path]) -> list[dict]:
    rows = []
    seen: set[Path] = set()
    for path in sorted(paths):
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        rows.append({
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    return rows


def manifest_for(
    folder: str,
    out_name: str,
    *,
    exclude_top_level: tuple[str, ...] = (),
) -> None:
    root = ROOT / folder
    if not root.exists():
        write_manifest([], out_name)
        return
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in exclude_top_level:
            continue
        paths.append(path)
    write_manifest(rows_for_paths(paths), out_name)


def manifest_for_top_level() -> None:
    paths: list[Path] = []
    for pattern in TOP_LEVEL_PATTERNS:
        paths.extend(ROOT.glob(pattern))
    write_manifest(rows_for_paths(paths), "release_root_sha256.csv")


def main() -> None:
    manifest_for_top_level()
    manifest_for("scripts", "scripts_sha256.csv")
    manifest_for("results", "results_sha256.csv")
    # The large raw trajectory archive is intentionally excluded from the
    # public release manifest; derived exp4 outputs remain covered.
    manifest_for(
        "exp4_results",
        "exp4_results_sha256.csv",
        exclude_top_level=("trajectories",),
    )
    manifest_for("exp5_results", "exp5_results_sha256.csv")
    manifest_for("acf_results", "acf_results_sha256.csv")
    manifest_for("figures", "figures_sha256.csv")
    manifest_for("tables", "tables_sha256.csv")
    print(f"Wrote manifests to {MANIFEST_DIR}")


if __name__ == "__main__":
    main()
