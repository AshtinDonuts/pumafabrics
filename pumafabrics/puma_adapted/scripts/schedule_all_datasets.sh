#!/usr/bin/env bash
# Run 15 training jobs sequentially: 5 datasets × 3 boundary loss weights.
# Invoke from anywhere; assumes train.py lives in the parent of this script's
# directory (puma_adapted).

set -euo pipefail

PUMA_ADAPTED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PUMA_ADAPTED"

RESULTS_BASE="${RESULTS_BASE:-./}"

DATASETS=(
  "2nd_order_R3S3_sweeping_16may"
  "2nd_order_R3S3_simple_w_shape"
  "2nd_order_R3S3_simple_line"
  "2nd_order_R3S3_simple_wipe"
  "2nd_order_R3S3_table_wipe_puma0"
)

run_one() {
  local params="$1"
  local blw="$2"
  local tag="$3"
  echo "========== params=${params}  boundary_loss_weight=${blw} -> results/${tag}/ =========="
  python train.py \
    --params "$params" \
    --results-base-directory "$RESULTS_BASE" \
    --results-path "results/${tag}/" \
    --boundary-loss-weight "$blw"
}

for ds in "${DATASETS[@]}"; do
  run_one "${ds}_blw0p01" 0.01  "${ds}_blw0p01"
  run_one "${ds}_blw1p0"  1.0   "${ds}_blw1p0"
  run_one "${ds}_blw10p0" 10.0  "${ds}_blw10p0"
done

echo "All 15 runs finished."
