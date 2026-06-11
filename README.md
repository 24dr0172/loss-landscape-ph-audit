# Topological Audit Reproducibility Package

This repository contains the executable code and release utilities for the loss-landscape persistent-homology reliability audit.

## Layout

```text
topological_audit/
  audit_common.py
  scripts/
  results/
  figures/
  tables/
  manifests/
  logs/
  environment.yml
  environment-cpu.yml
  environment-cuda.yml
  requirements-minimal.txt
  reproduce_all.sh
  resume_final.sh
  run_remaining_final.sh
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

`separation_flag`, `fallback`, `TSR`, `zrob`, `delta`, and null-collapse fields are diagnostics only. Block-permutation rows with block size larger than one are robustness diagnostics and should not be promoted to formal recurrence claims.

Graph-geodesic distances are built through `audit_common.geodesic_distance_matrix`. The kNN graph convention is

```python
A = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False)
G = A.maximum(A.T)
D = shortest_path(G, directed=False)
```

Scripts should not rebuild this graph locally.

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

`lint_syntax.py` parses and compiles repository Python files without importing heavy libraries. `verify_release_consistency.py` checks that shared primitives are owned by `audit_common.py`, scripts do not directly rebuild kNN/shortest-path graphs, and formal decision assignments do not combine p-values with diagnostic fallback conditions.

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

Use `resume_final.sh` to restart from a named stage after an interrupted run:

```bash
bash resume_final.sh code_10_main
```

Use `run_remaining_final.sh` only when the early validation scripts have already completed and the remaining scripts need to be regenerated.

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

## Release checklist

Before submission, verify:

```text
[ ] All experiment scripts import shared primitives from audit_common.py.
[ ] No experiment script directly calls kneighbors_graph, NearestNeighbors, or shortest_path for graph-geodesic construction.
[ ] All formal decisions are generated from pperm < 0.05 only.
[ ] Block-size > 1 rows are labelled as diagnostics.
[ ] All result CSVs include seed/config/metric/k/pct/null-count metadata.
[ ] Manuscript tables are generated from archived CSVs.
[ ] Figures referenced by the manuscript exist under figures/.
[ ] SHA-256 manifests are regenerated after the final run.
```
