# Historical outputs retained for comparison only

These files were moved here from the older release before the accuracy-rerun
source was prepared. They were generated before the increment-Gram distance
reconstruction was centralised and corrected.

They are preserved so that current and historical CSVs, figures, and tables
can be compared. They are **not** active inputs to `reproduce_all.sh`, are not
current results, and must not be mixed with outputs created by this package.

After `make_manifests.py` is run, the preserved files other than the old
documentary CSVs under `legacy_release_outputs/manifests/` are covered by
`manifests/legacy_release_outputs_sha256.csv`. The old manifests remain only as
historical documentation and are intentionally excluded from that new
manifest.

Use the current top-level `results/`, `exp4_results/`, `exp5_results/`,
`tables/`, `figures/`, `acf_results/`, and `manifests/` paths for the corrected
release only.
