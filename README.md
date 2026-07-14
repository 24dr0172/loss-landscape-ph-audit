# Topological Audit Release and Verification Package

This repository contains the analysis code, saved final CSV outputs, manuscript-facing figures and tables, release utilities, and checksum manifests for the loss-landscape persistent-homology reliability audit.

## Layout

```text
loss-landscape-ph-audit/
  audit_common.py
  scripts/
  results/
  figures/
  tables/
  manifests/
  acf_results/
  exp4_results/
  exp5_results/
  logs/
  environment.yml
  environment-cpu.yml
  environment-cuda.yml
  requirements-minimal.txt
  reproduce_all.sh
  make_manuscript_tables.py
  make_manifests.py
  verify_release_consistency.py
  lint_syntax.py
  sitecustomize.py
```

## Core audit convention

Formal neural inference uses only the matched-step permutation test with block size \(b=1\):

```text
pperm = (1 + number of null lifetimes >= observed lifetime) / (N_null + 1)
```

The formal trigger is always

```python
formal_trigger = pperm < 0.05
```

The fields `separation_flag`, `fallback`, `nominal_trigger`, `TSR`, `zrob`, `delta`, percentile ranks, and null-collapse diagnostics are descriptive only. They do not define formal recurrence evidence.

Rows from block-permutation runs with block size greater than one are retained as robustness diagnostics. They are not interpreted as formal recurrence evidence.

The main full-space neural analysis, optimization stress audit, and empirical block-calibration analysis use complete Vietoris–Rips filtration. Percentile-capped filtrations are used only in the sensitivity analysis and do not redefine the formal test.

Graph-geodesic distances are built through `audit_common.geodesic_distance_matrix`. The kNN graph convention is

```python
A = kneighbors_graph(
    X,
    n_neighbors=k,
    mode="distance",
    include_self=False,
)
G = A.maximum(A.T)
D = shortest_path(G, directed=False)
```

Experiment scripts call the shared graph-geodesic implementation rather than rebuilding this graph locally.

## Environment setup

CPU environment:

```bash
conda env create -f environment-cpu.yml
conda activate topological-audit
```

CUDA environment:

```bash
conda env create -f environment-cuda.yml
conda activate topological-audit
```

Minimal pip environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
```

## Release verification

Run the static release checks with:

```bash
python lint_syntax.py
python verify_release_consistency.py
```

`lint_syntax.py` parses and compiles the repository’s Python files without importing heavy libraries.

`verify_release_consistency.py` checks that shared primitives are owned by `audit_common.py`, that experiment scripts do not directly rebuild kNN or shortest-path graphs, and that formal decision assignments do not combine permutation p-values with diagnostic fallback conditions.

These checks verify the released code structure and decision convention. They do not rerun the neural-network experiments.

## Saved outputs and manuscript tables

The reported manuscript results should be checked against the saved final CSV outputs. The primary files include:

```text
table_main_topology.csv
table_appendix_full.csv
block_robustness_full.csv
neural_sensitivity_full.csv
graph_locality_diagnostics_main.csv
positive_control_drift_0.00_corrected.csv
positive_control_drift_1.00_corrected.csv
table_stress_main.csv
table_stress_appendix.csv
block_type1_summary.csv
forced_recurrence_results.csv
```

The following files are release and audit metadata:

```text
release_csv_inventory.csv
all_results_normalized.csv
trigger_summary.csv
```

They should not be used as direct manuscript counts. In particular, duplicate rows in `all_results_normalized.csv` are not independent experiments.

To rebuild the release metadata tables from the saved CSV outputs, run:

```bash
python make_manuscript_tables.py
```

This writes:

```text
tables/release_csv_inventory.csv
tables/all_results_normalized.csv
tables/trigger_summary.csv
```

## Checksum manifests

Build SHA-256 manifests with:

```bash
python make_manifests.py
```

This writes checksum manifests for the included repository files under `manifests/`.

The manifests cover files included in the release. They do not represent checksums for omitted trajectory archives unless those files are present locally when the manifests are created.

## Data included in this release

This repository includes analysis scripts, saved derived CSV outputs, manuscript-facing figures and tables, release-verification utilities, and SHA-256 manifests.

Large binary trajectory arrays and null-lifetime `.npy` files are not included in the GitHub repository.

The omitted Experiment 4 trajectory archive is approximately 105 GB. Its omission is documented in:

```text
exp4_results/trajectories/README.md
```

The omitted Experiment 5 trajectory archive is approximately 1.4 GB. Its omission is documented in:

```text
exp5_results/trajectories/README.md
```

The saved final CSV outputs are sufficient to inspect and verify the numerical results, summary tables, and diagnostics reported in the manuscript without downloading either trajectory archive.

## Interpretation of the released results

The main neural audit contains 36 architecture–optimizer–seed cells. Formal inference is based only on the matched-step \(b=1\) permutation test.

There is one formal trigger: MLP2/MNIST with AdamW, seed 2342. This result is not reproduced across seeds and is sensitive to the graph setting. It is therefore reported as a metric-sensitive boundary case, not as evidence of recurrent neural optimization.

All ResNet-18 main cells have an observed lifetime of zero and a permutation p-value of one.

The block-permutation results for \(b>1\), sensitivity-grid results, separation fields, robust scores, topological signal ratios, and null-collapse measurements are diagnostic only. They do not change the paper’s main conclusion:

> Under the specified audit, the released results do not show seed-reproducible \(H_1\) recurrence in the neural optimization trajectories.
