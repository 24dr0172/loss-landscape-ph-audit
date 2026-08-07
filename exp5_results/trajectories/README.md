# Experiment 5 trajectory archive (intentionally omitted)

The full Experiment 5 stress-trajectory archive is not distributed with this
source-and-results release. It contains six full-space trajectory arrays and
six matching metrics arrays (approximately 1.4 GB in total).

The public release retains the derived Experiment 5 outputs under
`exp5_results/`, including the frozen stress tables and figures. Those derived
files are covered by `manifests/exp5_results_sha256.csv`. The entire
`exp5_results/trajectories/` directory, including this placeholder, is
intentionally excluded from that manifest because the raw archive is not
shipped.

To run the heavy Experiment 5 analysis, restore the following six matched
pairs from the same release generation:

```text
smallcnn__edge_of_stability__seed42.npy
smallcnn__edge_of_stability__seed42__metrics.npy
smallcnn__edge_of_stability__seed1042.npy
smallcnn__edge_of_stability__seed1042__metrics.npy
smallcnn__edge_of_stability__seed2042.npy
smallcnn__edge_of_stability__seed2042__metrics.npy
smallcnn__extreme_momentum__seed10042.npy
smallcnn__extreme_momentum__seed10042__metrics.npy
smallcnn__extreme_momentum__seed11042.npy
smallcnn__extreme_momentum__seed11042__metrics.npy
smallcnn__extreme_momentum__seed12042.npy
smallcnn__extreme_momentum__seed12042__metrics.npy
```

The public archive alone supports inspection and verification of the released
derived outputs; it does not support a fresh heavy rerun until these cached
arrays have been restored.
