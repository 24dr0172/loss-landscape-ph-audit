# Provenance-only repair utilities

This directory contains utilities for documenting the numerical repair. It is
not part of the production analysis path and is never invoked by
`reproduce_all.sh`.

`generate_repair_ledger.py` reads the restored Experiment 4 checkpoint arrays,
computes the certificate once with the historical double-prefix distance
reconstruction and once with the corrected diagonal-anchored reconstruction,
and writes the per-cell comparison ledger documented in
`RERUN_CHANGELOG.md` at the release root. This is a release-provenance record,
not a separate manuscript result. The final computation and validation checks
are described in Appendix J, Numerical Implementation and Validation.

The historical routine is imported from the private helper retained in
`code_21_path_fidelity_certificate.py` solely as a provenance record. The
utility must not be converted into a production or formal-inference mode.

The ledger is authentic only when generated from the matching cached arrays.
It must never be filled from aggregate manuscript tables or from rounded
values quoted in prose.
