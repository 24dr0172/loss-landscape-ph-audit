# Accuracy-rerun change log

This file records the source, provenance, and release-state changes that
distinguish this package from the historical release. It is a provenance note,
not a manuscript result.

## Blocking correction: increment-Gram distance recovery

The historical scripts recovered interval squared distances from a double
prefix sum by inclusion--exclusion. For a small interval embedded in a long,
converging trajectory, that subtracts trajectory-scale quantities to recover a
step-scale number and can perturb the nearest-neighbour distances that
determine the OR-kNN graph.

The production implementation is now `audit_common.distance_from_step_gram`.
It accumulates each interval outward from its diagonal anchor and validates

\[
  D(r,r+1)=\sqrt{G_{rr}}.
\]

The following production components use the shared routine:

- `scripts/code_10_neural_trajectory_pipeline.py`
- `scripts/code_14_neural_fullspace_audit.py`
- `scripts/code_15_resnet_threshold_sensitivity.py`
- `scripts/code_16_resnet_positive_control_lambda0.py`
- `scripts/code_17_resnet_positive_control_lambda1.py`
- `scripts/code_19_ar_block_calibration.py`
- `scripts/code_20_graph_locality_diagnostics.py`
- `code_21_path_fidelity_certificate.py`

The retained historical helper in `code_21` documents the reconstruction that
produced the defect. It is not called by any production rerun path. The
release verifier rejects a production script or `reproduce_all.sh` if it
imports or invokes the provenance-only repair utility.

## Completed corrected-output state

The included derived outputs are the corrected analysis generation run on the
cached trajectory archives. The arrays were reused; they were not regenerated.

- The corrected 36-cell formal topology table is byte-identical to the
  historical table. Its SHA-256 is
  `c5437dfc7ea4854258689b3073f7f4e2cf4a74312322ec4b678de522291247db`.
- The preserved current-versus-historical comparison covers 1,302 rows across
  the five released result tables. The stored p-values, observed lifetimes,
  formal/nominal trigger fields, and decisions are unchanged.
- Certificate quantities are repaired separately from the inferential-output
  comparison. The authentic per-row pre/post certificate ledger is generated
  from the matching cached arrays by
  `provenance/generate_repair_ledger.py`.

The historical derived files remain under `legacy_release_outputs/` for
comparison only. They are not current results and are not inputs to
`reproduce_all.sh`.

## Reconstructed and aligned computations

1. `scripts/code_15_resnet_threshold_sensitivity.py` was reconstructed from
   the frozen `code_10` ResNet sensitivity convention.
2. The Experiment 5 full-space audit uses `GEODESIC_MAX_K = SUBSAMPLE_PH - 1`,
   matching the stated 200-checkpoint convention.
3. `code_21` generates the helix and overpass controls with the same random
   stream, 1,200-point generation size, and ordered stride subsampling used by
   `code_01`.
4. Theorem E values are computed per observed H1 bar. The necessary condition
   `required_mu_for_bind = b/d` is exported, but a non-binding ceiling is not
   treated as evidence for or against recurrence.

## Claims deliberately not made

- The internal adjacent-edge residual is a coherence diagnostic. It is not a
  certified error bound relative to an exact metric and does not provide a
  persistent-homology resolution floor.
- The accuracy repair does not assume that the historical trigger, non-trigger,
  or realised-`k` distribution survives automatically; those outputs are
  compared explicitly.
- The path-fidelity certificate is conditional on the realised graph and is
  not a formal detector or a power result.
- The historical-versus-corrected ledger is not back-filled from manuscript
  tables. Without matching cached arrays, the pre-repair certificate constants
  are not independently reproducible from the public derived release.

## Required run order

1. Restore the matching Experiment 4 and Experiment 5 trajectory/metrics
   pairs, if a computation is required.
2. Run the syntax and provenance checks listed in `README.md`.
3. Regenerate the Phase 1 certificate with `code_21` only if the path-fidelity
   outputs are being checked locally.
4. Generate the 36-row repair ledger with the provenance-only utility.
5. Run `tests/test_repair_reconciliation.py`.
6. Regenerate manifests and rerun `verify_release_consistency.py`.
7. Package the release only after the final tests, manifests, and README state
   agree.
