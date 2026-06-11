#!/usr/bin/env bash
# Reproduce the full topological-audit release outputs.
# Run from the repository root with CLEAN=1 to regenerate release artifacts.

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

if [[ "$CLEAN" == "1" ]]; then
  echo "[CLEAN] Removing release outputs: results/ figures/ tables/ manifests/ logs/"
  rm -rf results figures tables manifests logs
fi

mkdir -p results figures tables manifests logs

python -u lint_syntax.py
python -u verify_release_consistency.py

SCRIPTS=(
  "scripts/code_00_figure1_motivation.py"
  "scripts/code_01_synthetic_validation.py"
  "scripts/code_02_projection_budget_sensitivity.py"
  "scripts/code_03_four_archetype_validation.py"
  "scripts/code_04_curvature_sweep.py"
  "scripts/code_05_k_sensitivity_transition.py"
  "scripts/code_06_transition_focused_robustness.py"
  "scripts/code_07_circle_noise_robustness.py"
  "scripts/code_08_manifold_benchmarks.py"
  "scripts/code_10_neural_trajectory_pipeline.py"
  "scripts/code_13_forced_recurrence_positive_control.py"
  "scripts/code_14_neural_fullspace_audit.py"
  "scripts/code_15_resnet_threshold_sensitivity.py"
  "scripts/code_16_resnet_positive_control_lambda0.py"
  "scripts/code_17_resnet_positive_control_lambda1.py"
  "scripts/code_18_step_acf_analysis.py"
  "scripts/code_19_ar_block_calibration.py"
)

HEAVY_REGEX='code_10_|code_14_|code_15_|code_16_|code_17_'

for script in "${SCRIPTS[@]}"; do
  if [[ ! -f "$script" ]]; then
    echo "[FAIL] Missing expected script: $script" >&2
    exit 1
  fi
  if [[ "$RUN_HEAVY" == "0" && "$script" =~ $HEAVY_REGEX ]]; then
    echo "[SKIP heavy] $script"
    continue
  fi
  echo "======================================================================"
  echo "[RUN] $script"
  echo "======================================================================"
  python -u "$script" 2>&1 | tee "logs/$(basename "$script" .py).log"
done

python -u make_manuscript_tables.py
python -u make_manifests.py

echo "Done. Check results/, figures/, tables/, manifests/, and logs/."
