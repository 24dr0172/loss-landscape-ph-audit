#!/usr/bin/env python3
"""Catch missing release files, stale hashes, and incomplete summary tables."""
from __future__ import annotations

import csv
import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import verify_release_consistency as verifier  # noqa: E402


class ReleaseArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(verifier, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        verifier.errors.clear()
        folders = (
            "release_root", "scripts", "tests", "provenance", "results",
            "exp4_results", "exp5_results", "acf_results", "figures", "tables",
            "legacy_release_outputs",
        )
        for folder in folders:
            self.write_csv(f"manifests/{folder}_sha256.csv",
                           ["relative_path", "sha256", "size_bytes"], [])
        self.file = self.root / "results/control.csv"
        self.write_csv("results/control.csv", ["value"], [[1], [2]])
        data = self.file.read_bytes()
        self.write_csv("manifests/results_sha256.csv",
                       ["relative_path", "sha256", "size_bytes"],
                       [["results/control.csv", hashlib.sha256(data).hexdigest(), len(data)]])

    def write_csv(self, relative, fields, rows):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_valid_manifest(self):
        verifier.check_manifest_integrity()
        self.assertEqual(verifier.errors, [])

    def test_missing_file(self):
        self.file.unlink()
        verifier.check_manifest_integrity()
        self.assertTrue(any("lists missing file: results/control.csv" in e for e in verifier.errors))

    def test_modified_file_same_size(self):
        self.file.write_bytes(self.file.read_bytes().replace(b"1", b"3"))
        verifier.check_manifest_integrity()
        self.assertTrue(any("stale SHA-256 for results/control.csv" in e for e in verifier.errors))

    def test_unlisted_output(self):
        self.write_csv("results/extended.csv", ["value"], [[3]])
        verifier.check_manifest_integrity()
        self.assertTrue(any("coverage differs" in e and "results/extended.csv" in e
                            for e in verifier.errors))

    def write_summaries(self, count):
        self.write_csv("tables/release_csv_inventory.csv", ["source_csv", "n_rows"],
                       [["results/control.csv", count]])
        self.write_csv("tables/trigger_summary.csv", ["source_csv", "n_rows"],
                       [["results/control.csv", count]])
        self.write_csv("tables/all_results_normalized.csv", ["source_csv"],
                       [["results/control.csv"] for _ in range(count)])

    def test_valid_summaries(self):
        self.write_summaries(2)
        verifier.check_summary_sources()
        self.assertEqual(verifier.errors, [])

    def test_stale_summaries(self):
        self.write_summaries(1)
        verifier.check_summary_sources()
        self.assertEqual(sum("stale source counts" in e for e in verifier.errors), 3)


class SourceOnlyTests(unittest.TestCase):
    def test_preflight_skips_generated_artifacts(self):
        verifier.errors.clear()
        verifier.warnings_out.clear()
        checks = ("check_manifest_trajectory_exclusions", "check_auxiliary_manifests",
                  "check_repair_ledger_schema", "check_manifest_integrity", "check_summary_sources")
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["verify_release_consistency.py", "--source-only"]))
            for name in checks:
                stack.enter_context(patch.object(verifier, name, side_effect=AssertionError(name)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            verifier.main()
        self.assertEqual(verifier.errors, [])


if __name__ == "__main__":
    unittest.main()
