# Experiment 4 trajectory archive (intentionally omitted)

The full Experiment 4 neural trajectory archive is not distributed with this
source-and-results release. It contains 36 full-space trajectory arrays and 36
matching metrics arrays (approximately 105 GB in total).

The public release retains the derived Experiment 4 outputs under
`exp4_results/`, including the frozen manuscript tables, figures, and
path-fidelity certificate. Those derived files are covered by
`manifests/exp4_results_sha256.csv`. The entire
`exp4_results/trajectories/` directory, including this placeholder, is
intentionally excluded from that manifest because the raw archive is not
shipped.

To run the heavy Experiment 4 analysis, restore the cached arrays from the
same release generation into this directory. There must be 36 matched pairs
with these names:

```text
<architecture>__<optimizer>__seed<seed>.npy
<architecture>__<optimizer>__seed<seed>__metrics.npy
```

The 36 trajectory names are the Cartesian product recorded in
`exp4_results/main_paper/table_main_topology.csv`: three architectures
(`mlp2_mnist`, `smallcnn_cifar10`, `resnet18_cifar10`), four optimizers
(`sgd`, `sgd_momentum`, `adam`, `adamw`), and their three recorded seeds. Each
trajectory array must have its metrics counterpart with the identical stem.

The public archive alone supports inspection and verification of the released
derived outputs; it does not support a fresh heavy rerun until these cached
arrays have been restored.
