# Graph-Geodesic Neural-Trajectory Audit — Accuracy Rerun Package

This package is the authoritative local rerun source for the neural
optimization audit.  It includes the full 36-cell Experiment 4 main neural
experiment, the 6-cell Experiment 5 stress experiment, the reconstructed ResNet
filtration-percentile sensitivity script, the Phase 1 path-fidelity
certificate, positive controls, and the reproducibility utilities.

It intentionally starts with empty generated-output directories.  The
historical CSVs and figures from the pre-fix release are preserved under
`legacy_release_outputs/` for comparison only.  They must not be mixed with
or cited as results of this accuracy rerun.

## What is corrected

- `audit_common.py` now owns the only production reconstruction from an
  increment Gram matrix to checkpoint distances.  It uses a diagonal-anchored
  computation, validates every increment permutation, and checks the lag-one
  invariant against `sqrt(diag(G))`.
- The main neural audit (`code_10`), stress audit (`code_14`), both positive
  controls (`code_16`, `code_17`), block calibration (`code_19`), graph
  locality diagnostic (`code_20`), Phase 1 certificate (`code_21`), and
  reconstructed threshold sensitivity script (`code_15`) use that shared route.
- The Experiment 5 audit now permits adaptive k up to `n - 1`, matching the
  stated full 200-checkpoint convention.
- `reproduce_all.sh` preserves **both** trajectory archives during `CLEAN=1`
  and refuses to retrain unless `ALLOW_RETRAIN=1` is supplied explicitly.
- The certificate reproduces the `code_01` helix/overpass generation stream,
  computes Theorem E quantities per bar, and labels its numerical invariant
  residual only as an internal coherence diagnostic—not as a PH error bound.

## Add the trajectory archives

Replace the placeholder README files in these directories with the binary
arrays and matching metrics arrays from the same release generation:

- `exp4_results/trajectories/` — 36 Experiment 4 trajectories and 36 matching
  `__metrics.npy` arrays.
- `exp5_results/trajectories/` — 6 Experiment 5 trajectories and 6 matching
  `__metrics.npy` arrays.

The exact file-name requirements are listed in the README inside each
directory.  The runner verifies both counts and every trajectory/metrics pair
before it removes any generated output.  It will not silently substitute a
random-walk stub or start a new training run.

## Install

Use one of the supplied environments:

```bash
conda env create -f environment-cuda.yml
conda activate topological-audit
```

For CPU-only use, create from `environment-cpu.yml`.  `requirements-minimal.txt`
is available for an existing Python environment.

## Verify the source before a costly rerun

Run from the repository root:

```bash
python lint_syntax.py
python verify_release_consistency.py
python tests/test_distance_reconstruction.py
python tests/test_code21_control_provenance.py
python tests/test_code15_provenance.py
```

These checks are required to pass before the main computation.  They compare
the shared Gram reconstruction against direct float64 coordinate distances,
including a permuted matched-step null, and verify the Phase 1 controls against
the `code_01` random-number stream and the reconstructed `code_15` seed/
percentile convention against `code_10`.

## Rerun

With both trajectory archives restored, run:

```bash
CLEAN=1 RUN_HEAVY=1 HASH_TRAJECTORIES=1 bash reproduce_all.sh
```

`CLEAN=1` removes generated outputs but preserves both trajectory directories.
`HASH_TRAJECTORIES=1` records SHA-256 provenance for the 36 Experiment 4
arrays in the Phase 1 certificate; it adds disk I/O but is recommended for
this accuracy rerun.

To regenerate only the lightweight synthetic scripts, use:

```bash
RUN_HEAVY=0 bash reproduce_all.sh
```

Do not set `ALLOW_RETRAIN=1` unless a deliberate retraining experiment is
intended and separately documented.  A trajectory archive from a different
release generation must not be mixed into this rerun.

## Formal inference convention

The only formal neural recurrence rule is:

```python
formal_trigger = pperm < 0.05
```

Matched-step permutations (`b=1`) are the formal null.  Block permutations
with `b>1`, sensitivity sweeps, graph-locality statistics, the path-fidelity
certificate, and positive controls are diagnostics.  They do not redefine
the formal decision.

The package deliberately makes no pre-rerun claim about whether reproducible
H1 is or is not detected.  That conclusion must be reconstructed from the
newly generated `exp4_results/main_paper/table_main_topology.csv` after all
accuracy checks pass.

## Main outputs after a successful rerun

- `exp4_results/main_paper/table_main_topology.csv` — 36-cell formal neural
  matched-step audit.
- `exp4_results/threshold_sensitivity_resnet18_sgd/` — reconstructed diagnostic
  percentile sweep.
- `exp4_results/appendix/path_fidelity/` — Phase 1 certificate and per-bar
  Theorem E calculation.
- `exp5_results/main_paper/table_stress_main.csv` — stress trajectories.
- `results/`, `acf_results/`, `figures/`, `tables/`, and `manifests/` —
  regenerated supporting outputs, summaries, and checksums.

The file `RERUN_CHANGELOG.md` gives a precise audit trail of the source-level
corrections and the intended verification order.
