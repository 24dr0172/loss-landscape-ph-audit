# Accuracy-rerun change log

This file records the changes that distinguish this source-and-rerun package
from the historical release.  It is a provenance note, not a manuscript
result.

## Blocking correction: increment-Gram distance recovery

The historical scripts recovered interval squared distances from a double
prefix sum by inclusion--exclusion.  For a small interval embedded in a long,
converging trajectory, that subtracts trajectory-scale quantities to recover a
step-scale number.  It can therefore perturb the nearest-neighbour distances
that determine the OR-kNN graph.

The production implementation is now `audit_common.distance_from_step_gram`.
It accumulates each interval outward from its diagonal anchor and validates the
lag-one identity

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

The private legacy helper in `code_21` is retained only to document the
historical diagnosis.  It is not called by any rerun path; the release
consistency check fails if it is invoked.

## Reconstructed and aligned computations

1. `scripts/code_15_resnet_threshold_sensitivity.py` has been reconstructed
   from the frozen `code_10` ResNet sensitivity convention.  The old release
   referenced this script and its output in manifests but did not contain the
   source.
2. The Experiment 5 full-space audit now uses `GEODESIC_MAX_K =
   SUBSAMPLE_PH - 1`, so its adaptive-connectivity limit agrees with the
   200-checkpoint convention.
3. `code_21` generates the helix and overpass controls with the same random
   stream, 1,200-point generation size, and ordered stride subsampling used by
   `code_01`.
4. Theorem E values are computed per observed H1 bar.  The necessary condition
   `required_mu_for_bind = b/d` is exported, but the calculation does not
   claim that a non-binding ceiling validates or invalidates recurrence.

## Claims deliberately not made

- An internal adjacent-edge residual is a coherence failure if nonzero.  It is
  not, by itself, a certified error bound relative to an exact metric and does
  not provide a bottleneck-stability resolution floor for H1.
- The accuracy rerun does not assume the historical trigger, non-trigger, or
  realised-k distribution survives the corrected distance route.
- The legacy outputs are not current results.  They are preserved separately
  only to make old-versus-rerun comparisons possible.

## Required run order

1. Restore Experiment 4 and Experiment 5 trajectory/metrics pairs in their
   placeholder directories.
2. Run `python lint_syntax.py`, `python verify_release_consistency.py`,
   `python tests/test_distance_reconstruction.py`, and
   `python tests/test_code21_control_provenance.py`, and
   `python tests/test_code15_provenance.py`.
3. Run `CLEAN=1 RUN_HEAVY=1 HASH_TRAJECTORIES=1 bash reproduce_all.sh`.
4. Check the regenerated main table, sensitivity table, path-fidelity
   certificate, and manifests as one release generation.
