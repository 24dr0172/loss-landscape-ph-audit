# Topological Audit Reproducibility Package

This repository contains executable code, derived outputs, release utilities, and checksum manifests for the loss-landscape persistent-homology reliability audit.

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

The formal recurrence trigger is always

```python
formal_trigger = pperm < 0.05
```

The fields `separation_flag`, `fallback`, `TSR`, `zrob`, `delta`, and null-collapse diagnostics are descriptive only. They do not define formal recurrence evidence.

Rows from block-permutation runs with block size larger than one are retained as robustness diagnostics. They are not interpreted as formal recurrence evidence.

Graph-geodesic distances are built through `audit_common.geodesic_distance_matrix`. The kNN graph convention is

```python
A = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False)
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

## Verification

Run static checks before launching experiments:

```bash
python lint_syntax.py
python verify_release_consistency.py
```

`lint_syntax.py` parses and compiles repository Python files without importing heavy libraries.

`verify_release_consistency.py` checks that shared primitives are owned by `audit_common.py`, experiment scripts do not directly rebuild kNN/shortest-path graphs, and formal decision assignments do not combine p-values with diagnostic fallback conditions.

## Reproduction

Run a full regeneration:

```bash
CLEAN=1 RUN_HEAVY=1 bash reproduce_all.sh
```

Run a lightweight smoke regeneration that skips heavy neural scripts:

```bash
CLEAN=1 RUN_HEAVY=0 bash reproduce_all.sh
```

Outputs are written to:

```text
results/
figures/
tables/
manifests/
logs/
```

## Tables and manifests

Build release tables from archived CSVs:

```bash
python make_manuscript_tables.py
```

This writes:

```text
tables/release_csv_inventory.csv
tables/all_results_normalized.csv
tables/trigger_summary.csv
```

Build SHA-256 manifests:

```bash
python make_manifests.py
```

This writes root, script, result, figure, and table checksum manifests under `manifests/`.

## Data included in this release

This repository includes executable scripts, derived CSV outputs, manuscript-facing figures and tables, and SHA-256 manifests.

Large binary trajectory arrays and null-lifetime `.npy` files are not included in this GitHub release.

The placeholder `exp4_results/trajectories/README.md` documents the omitted trajectory archive. The derived CSV outputs included here are sufficient to inspect the reported summary tables and diagnostics without downloading the full generated trajectory arrays.

## Release verification

This release centralizes shared audit routines in `audit_common.py`. Experiment scripts call the shared implementation for graph-geodesic distances, permutation nulls, lifetime extraction, summary statistics, and SHA-256 checksums.

Formal recurrence decisions use only the matched-step permutation rule

```python
formal_trigger = pperm < 0.05
```

Rows with block size greater than one are retained only as null-sensitivity diagnostics and are not interpreted as formal recurrence evidence.
