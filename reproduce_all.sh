#!/usr/bin/env bash
# Reproduce the topological-audit release outputs.
# CLEAN=1 removes generated outputs while preserving the large trajectory archive.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CLEAN="${CLEAN:-0}"
RUN_HEAVY="${RUN_HEAVY:-1}"
STRICT_MANIFOLD_BENCHMARK="${STRICT_MANIFOLD_BENCHMARK:-0}"
ALLOW_RETRAIN="${ALLOW_RETRAIN:-0}"

# Preflight the cached main-neural archive before CLEAN removes any derived
# outputs. code_10 requires both each trajectory and its matching metrics file
# to use the cache; a missing metrics file would otherwise cause that run to be
# retrained and its trajectory to be overwritten.
TRAJ_DIR="exp4_results/trajectories"
TRAJ_COUNT=0
METRIC_COUNT=0
MISSING_METRIC_PAIRS=0
if [[ "$RUN_HEAVY" == "1" ]]; then
  if [[ -d "$TRAJ_DIR" ]]; then
    TRAJ_COUNT="$(find "$TRAJ_DIR" -maxdepth 1 -type f -name '*__*__seed*.npy' ! -name '*__metrics.npy' | wc -l | tr -d ' ')"
    METRIC_COUNT="$(find "$TRAJ_DIR" -maxdepth 1 -type f -name '*__*__seed*__metrics.npy' | wc -l | tr -d ' ')"
    while IFS= read -r trajectory_path; do
      metrics_path="${trajectory_path%.npy}__metrics.npy"
      if [[ ! -f "$metrics_path" ]]; then
        echo "[MISSING METRICS] $metrics_path" >&2
        MISSING_METRIC_PAIRS=$((MISSING_METRIC_PAIRS + 1))
      fi
    done < <(
      find "$TRAJ_DIR" -maxdepth 1 -type f \
        -name '*__*__seed*.npy' ! -name '*__metrics.npy' | sort
    )
  fi

  if [[ "$ALLOW_RETRAIN" != "1" ]] && {
    [[ "$TRAJ_COUNT" -lt 36 ]] ||
    [[ "$METRIC_COUNT" -lt 36 ]] ||
    [[ "$MISSING_METRIC_PAIRS" -gt 0 ]]
  }; then
    echo "[FAIL] Cached neural archive is incomplete:" >&2
    echo "       trajectories=$TRAJ_COUNT/36, metrics=$METRIC_COUNT/36, missing_pairs=$MISSING_METRIC_PAIRS" >&2
    echo "       Refusing to start CLEAN reproduction because code_10 could retrain or overwrite a cached run." >&2
    echo "       Restore the missing cache files, or set ALLOW_RETRAIN=1 deliberately." >&2
    exit 1
  fi
fi

if [[ "$CLEAN" == "1" ]]; then
  echo "[CLEAN] Removing generated outputs while preserving exp4_results/trajectories/"
  rm -rf results figures tables manifests logs acf_results exp5_results
  rm -rf \
    exp4_results/main_paper \
    exp4_results/appendix \
    exp4_results/block_robustness \
    exp4_results/neural_sensitivity \
    exp4_results/threshold_sensitivity_resnet18_sgd \
    exp4_results/block_type1_calibration \
    exp4_results/sanity
  rm -f exp4_results/ckpt_*.csv exp4_results/exp4_results_full.csv
fi

mkdir -p results figures tables manifests logs

python -u lint_syntax.py
python -u verify_release_consistency.py

run_logged() {
  local log_name="$1"
  shift
  echo "======================================================================"
  echo "[RUN] $*"
  echo "======================================================================"
  "$@" 2>&1 | tee "logs/${log_name}.log"
}

# Synthetic and lightweight validation scripts.
LIGHT_SCRIPTS_PRE_MANIFOLD=(
  "scripts/code_00_figure1_motivation.py"
  "scripts/code_01_synthetic_validation.py"
  "scripts/code_02_projection_budget_sensitivity.py"
  "scripts/code_03_four_archetype_validation.py"
  "scripts/code_04_curvature_sweep.py"
  "scripts/code_05_k_sensitivity_transition.py"
  "scripts/code_06_transition_focused_robustness.py"
  "scripts/code_07_circle_noise_robustness.py"
)

for script in "${LIGHT_SCRIPTS_PRE_MANIFOLD[@]}"; do
  [[ -f "$script" ]] || { echo "[FAIL] Missing expected script: $script" >&2; exit 1; }
  run_logged "$(basename "$script" .py)" python -u "$script"
done

# The manifold suite is a descriptive stress test. Its known Swiss-roll
# mismatch is written to the CSV but does not terminate normal reproduction.
# Use STRICT_MANIFOLD_BENCHMARK=1 to make mismatches fatal intentionally.
MANIFOLD_SCRIPT="scripts/code_08_manifold_benchmarks.py"
[[ -f "$MANIFOLD_SCRIPT" ]] || { echo "[FAIL] Missing expected script: $MANIFOLD_SCRIPT" >&2; exit 1; }
run_logged "code_08_manifold_benchmarks" \
  env STRICT_MANIFOLD_BENCHMARK="$STRICT_MANIFOLD_BENCHMARK" \
  python -u "$MANIFOLD_SCRIPT"

FORCED_CONTROL_SCRIPT="scripts/code_13_forced_recurrence_positive_control.py"
[[ -f "$FORCED_CONTROL_SCRIPT" ]] || { echo "[FAIL] Missing expected script: $FORCED_CONTROL_SCRIPT" >&2; exit 1; }
run_logged "code_13_forced_recurrence_positive_control" \
  python -u "$FORCED_CONTROL_SCRIPT"

if [[ "$RUN_HEAVY" == "1" ]]; then
  echo "[TRAJECTORIES] Found $TRAJ_COUNT cached trajectories and $METRIC_COUNT matching metrics files (ALLOW_RETRAIN=$ALLOW_RETRAIN)."

  HEAVY_SCRIPTS=(
    "scripts/code_10_neural_trajectory_pipeline.py"
    "scripts/code_14_neural_fullspace_audit.py"
    "scripts/code_15_resnet_threshold_sensitivity.py"
    "scripts/code_16_resnet_positive_control_lambda0.py"
    "scripts/code_17_resnet_positive_control_lambda1.py"
    "scripts/code_18_step_acf_analysis.py"
    "scripts/code_19_ar_block_calibration.py"
    "scripts/code_20_graph_locality_diagnostics.py"
  )
  for script in "${HEAVY_SCRIPTS[@]}"; do
    [[ -f "$script" ]] || { echo "[FAIL] Missing expected script: $script" >&2; exit 1; }
  done

  # Main neural audit, then its two explicitly diagnostic stages.
  run_logged "code_10_neural_trajectory_pipeline_main" \
    env SKIP_MAIN=0 RUN_BLOCK_ROBUSTNESS=0 RUN_NEURAL_SENSITIVITY=0 \
    python -u scripts/code_10_neural_trajectory_pipeline.py
  run_logged "code_10_neural_trajectory_pipeline_block" \
    env SKIP_MAIN=1 RUN_BLOCK_ROBUSTNESS=1 RUN_NEURAL_SENSITIVITY=0 \
    python -u scripts/code_10_neural_trajectory_pipeline.py
  run_logged "code_10_neural_trajectory_pipeline_sensitivity" \
    env SKIP_MAIN=1 RUN_BLOCK_ROBUSTNESS=0 RUN_NEURAL_SENSITIVITY=1 \
    python -u scripts/code_10_neural_trajectory_pipeline.py

  run_logged "code_14_neural_fullspace_audit" \
    python -u scripts/code_14_neural_fullspace_audit.py
  run_logged "code_15_resnet_threshold_sensitivity" \
    python -u scripts/code_15_resnet_threshold_sensitivity.py
  run_logged "code_16_resnet_positive_control_lambda0" \
    python -u scripts/code_16_resnet_positive_control_lambda0.py
  run_logged "code_17_resnet_positive_control_lambda1" \
    python -u scripts/code_17_resnet_positive_control_lambda1.py
  run_logged "code_18_step_acf_analysis" \
    python -u scripts/code_18_step_acf_analysis.py --stages all --force_next
  run_logged "code_19_ar_block_calibration" \
    python -u scripts/code_19_ar_block_calibration.py \
      --trajectory_glob "exp4_results/trajectories/mlp2_mnist__*__seed*.npy"
  run_logged "code_20_graph_locality_diagnostics" \
    python -u scripts/code_20_graph_locality_diagnostics.py
else
  echo "[SKIP heavy] Neural training/audits, ACF, block calibration, and locality diagnostics"
fi

python -u make_manuscript_tables.py
python -u make_manifests.py

echo "Done. Check results/, exp4_results/, exp5_results/, acf_results/, figures/, tables/, manifests/, and logs/."
