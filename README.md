# A Graph-Geodesic Audit of Recurrence in Neural Optimization — Release

This repository provides the source and derived results for the
graph-geodesic neural-trajectory audit. It contains the
outputs for the 36-cell Experiment 4 neural audit, the six-cell Experiment 5
stress audit, the reconstructed threshold-sensitivity source, the Phase 1
path-fidelity certificate, positive controls, and release-verification tools.

The included derived outputs represent the corrected analysis generation on
cached trajectory archives. The trajectory arrays were not regenerated; they
are omitted from the public package because of their size. The historical
derived outputs used for comparison are retained under
`legacy_release_outputs/` and are not current results.

## Relationship to the paper

The main audit covers 36 neural training paths across three architectures,
four optimizers, and three seeds. No architecture-optimizer family meets the
two-of-three-seed reproducibility criterion. All 36 runs meet the
null-collapse diagnostic; the single matched-step trigger is under-resolved
and metric-sensitive. These findings do not establish the presence or absence
of recurrence at another resolution, representation, or null model.

Appendix J, **Numerical Implementation and Validation**, presents the final
distance calculation and its checks. The release's Phase 1 path-fidelity
certificate corresponds to the paper's Stage G realized-graph compatibility
check; it is not a recurrence detector or a power guarantee. Source-level
numerical corrections are documented in `RERUN_CHANGELOG.md`. The comparison
ledger and provenance utilities described in the assembled release are not
yet included in this GitHub snapshot.

## Implementation and validation

- `audit_common.py` owns the production reconstruction from an increment Gram
  matrix to checkpoint distances. It uses the diagonal-anchored computation,
  validates increment permutations, and checks the lag-one invariant against
  `sqrt(diag(G))`.
- The main neural audit (`code_10`), stress audit (`code_14`), both positive
  controls (`code_16`, `code_17`), block calibration (`code_19`), graph
  locality diagnostic (`code_20`), Phase 1 certificate (`code_21`), and
  reconstructed threshold-sensitivity script (`code_15`) use that shared
  production route.
- Experiment 5 permits adaptive `k` up to `n - 1`, matching the frozen
  200-checkpoint convention.
- `reproduce_all.sh` preserves both trajectory archives during `CLEAN=1` and
  refuses to retrain unless `ALLOW_RETRAIN=1` is supplied explicitly.
- The certificate uses the `code_01` control-generation stream, computes the
  scale-localised quantities per observed bar, and treats invariant residuals
  as implementation diagnostics rather than PH error bounds.

## Omitted trajectory archives

The public package omits both binary trajectory trees:

- `exp4_results/trajectories/` — 36 Experiment 4 trajectory arrays and 36
  matching metrics arrays, approximately 105 GB in total.
- `exp5_results/trajectories/` — six Experiment 5 trajectory arrays and six
  matching metrics arrays, approximately 1.4 GB in total.

Checksum manifests cover the supplied derived outputs. SHA-256 records are
supplied for the 36 Experiment 4 trajectories in the certificate output; the
Experiment 5 arrays are identified by name and size in their placeholder
README.

Restore the matching arrays from the same release generation before any
heavy rerun. The runner checks both archive counts and every
trajectory/metrics pair before deleting generated outputs.

## Install

Use one of the supplied environments:

```bash
conda env create -f environment-cuda.yml
conda activate topological-audit
```

For CPU-only use, create from `environment-cpu.yml`.
`requirements-minimal.txt` is available for an existing Python environment.

## Verification and optional diagnostic regeneration

From the repository root, run the published source checks:

```bash
python lint_syntax.py
python tests/test_distance_reconstruction.py
python tests/test_code21_control_provenance.py
python tests/test_code15_provenance.py
python verify_release_consistency.py
```

The supplied outputs can be inspected without restoring the raw arrays.
Optional regeneration of the Stage G compatibility diagnostics requires the
matching trajectory archives, which are omitted from the public package:

```bash
python code_21_path_fidelity_certificate.py \
  --traj-dir exp4_results/trajectories \
  --main-results exp4_results/main_paper/table_main_topology.csv \
  --kmax 199 \
  --with-controls \
  --hash-trajectories \
  --out-dir exp4_results/appendix/path_fidelity
```

After an intentional output regeneration, regenerate manifests with
`python make_manifests.py` and run `python verify_release_consistency.py`
again. The separate historical-comparison ledger is not currently published
in this repository.

## Full rerun

A full rerun requires both matching trajectory archives. To deliberately
regenerate the analyses after restoring them, use:

```bash
CLEAN=1 RUN_HEAVY=1 HASH_TRAJECTORIES=1 bash reproduce_all.sh
```

`CLEAN=1` removes generated outputs while preserving both trajectory
directories. Do not set `ALLOW_RETRAIN=1` unless a separately documented
training experiment is intended. A trajectory archive from a different
release generation must not be mixed into this rerun.

For lightweight synthetic analyses only:

```bash
RUN_HEAVY=0 bash reproduce_all.sh
```

## Formal inference convention

The only formal neural recurrence rule is:

```python
formal_trigger = pperm < 0.05
```

Matched-step permutations (`b=1`) are the formal null. The test has
conditional Type-I error control under exchangeable increments; a trigger
does not by itself establish recurrence. Block permutations with `b>1`,
sensitivity sweeps, graph-locality statistics, the path-fidelity certificate,
and positive controls are diagnostics. They do not redefine the formal
decision. Seed reproducibility and resolution diagnostics qualify the
interpretation of each per-cell result.

## Main outputs

- `exp4_results/main_paper/table_main_topology.csv` — 36-cell formal
  matched-step audit.
- `exp4_results/threshold_sensitivity_resnet18_sgd/` — reconstructed
  diagnostic percentile sweep.
- `exp4_results/appendix/path_fidelity/` — Stage G graph-compatibility
  diagnostics and per-bar bounds.
- `exp5_results/main_paper/table_stress_main.csv` — stress trajectories.
- `results/`, `acf_results/`, `figures/`, `tables/`, and `manifests/` —
  supporting outputs, summaries, and checksums.

`RERUN_CHANGELOG.md` records the numerical implementation changes and the
reproduction workflow.
