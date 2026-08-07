#!/usr/bin/env python3
"""Syntax-check repository Python files without importing heavy dependencies.

The check parses and compiles top-level support files and `scripts/**/*.py`.
It does not import modules or run experiments.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".venv", "venv", "env", "build", "dist",
}


def iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            continue
        yield path


def main() -> None:
    bad: list[tuple[Path, SyntaxError]] = []
    checked = 0
    for path in iter_python_files(ROOT):
        checked += 1
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            bad.append((path, exc))

    if bad:
        print("Python syntax check failed:", file=sys.stderr)
        for path, exc in bad:
            print(f"  {path.relative_to(ROOT)}:{exc.lineno}:{exc.offset}: {exc.msg}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Python syntax check passed for {checked} files.")


if __name__ == "__main__":
    main()
