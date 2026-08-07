#!/usr/bin/env python3
"""Check the reconstructed code_15 against code_10's frozen convention."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "code15_sensitivity", ROOT / "scripts" / "code_15_resnet_threshold_sensitivity.py"
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Could not load code_15_resnet_threshold_sensitivity.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    if MODULE.DEFAULT_SEEDS != (20042, 21042, 22042):
        raise AssertionError(f"unexpected ResNet/SGD seed grid: {MODULE.DEFAULT_SEEDS}")
    if MODULE.DEFAULT_PCTS != (90.0, 95.0, 99.0):
        raise AssertionError(f"unexpected percentile grid: {MODULE.DEFAULT_PCTS}")
    if MODULE.DEFAULT_N_NULLS != 50 or MODULE.DEFAULT_K0 != 12:
        raise AssertionError("code_15 does not match the frozen ResNet sensitivity settings")
    expected = 21042 + 9000 + 100 * 12 + 95
    if MODULE.sensitivity_null_seed(21042, 12, 95.0) != expected:
        raise AssertionError("code_15 null seed no longer matches code_10's sensitivity formula")
    filename = MODULE.trajectory_path(Path("exp4_results/trajectories"), 21042).name
    if filename != "resnet18_cifar10__sgd__seed21042.npy":
        raise AssertionError(f"unexpected ResNet trajectory filename: {filename}")
    print("code_15 provenance test passed.")


if __name__ == "__main__":
    main()
