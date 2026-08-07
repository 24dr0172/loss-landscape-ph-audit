#!/usr/bin/env python3
"""Check that code_21 uses the exact code_01 control-generation stream."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "code21_certificate", ROOT / "code_21_path_fidelity_certificate.py"
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Could not load code_21_path_fidelity_certificate.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def expected_controls() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n_samples, subsample = 1200, 600

    t = np.linspace(0.0, 1.0, n_samples)
    theta = 6.0 * np.pi * t
    helix = np.vstack([3.0 * np.cos(theta), 3.0 * np.sin(theta), 12.0 * t]).T
    helix += 0.03 * rng.standard_normal(helix.shape)

    t = np.linspace(-2.2, 2.2, n_samples)
    overpass = np.zeros((n_samples, 50))
    overpass[:, 0] = 500.0 * (t**3 - 3.0 * t)
    overpass[:, 1] = 500.0 * (t**2 - 1.0)
    crossing = np.sqrt(3.0)
    bump = 1000.0 * np.exp(-200.0 * (t + crossing) ** 2)
    bump -= 1000.0 * np.exp(-200.0 * (t - crossing) ** 2)
    overpass[:, 2:] = bump[:, None]
    overpass += 0.5 * rng.standard_normal(overpass.shape)

    index = np.linspace(0, n_samples - 1, subsample, dtype=int)
    return helix[index], overpass[index]


def main() -> None:
    helix_expected, overpass_expected = expected_controls()
    controls = MODULE.control_specs(600, 42)
    helix_actual = controls[0][3]
    overpass_actual = controls[1][3]
    if not np.array_equal(helix_actual, helix_expected):
        raise AssertionError("code_21 helix control does not match code_01 generation order")
    if not np.array_equal(overpass_actual, overpass_expected):
        raise AssertionError("code_21 overpass control does not match code_01 generation order")
    print("code_21 control provenance test passed.")


if __name__ == "__main__":
    main()
