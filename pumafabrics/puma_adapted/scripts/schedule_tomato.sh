#!/usr/bin/env bash
# Run three training jobs sequentially with different boundary loss weights and
# distinct results subdirectories. Invoke from anywhere; assumes train.py lives
# in the parent of this script's directory (puma_adapted).

set -euo pipefail

PUMA_ADAPTED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PUMA_ADAPTED"

PARAMS="${PARAMS:-2nd_order_R3S3_tomato_31may}"
RESULTS_BASE="${RESULTS_BASE:-./}"

run_one() {
  local blw="$1"
  local tag="$2"
  echo "========== boundary_loss_weight=${blw} -> results/${tag}/ =========="
  python train.py \
    --params "$PARAMS" \
    --results-base-directory "$RESULTS_BASE" \
    --results-path "results/${tag}/" \
    --boundary-loss-weight "$blw"
}

run_one 0.01 "${PARAMS}_blw0p01"
run_one 1.0 "${PARAMS}_blw1p0"
run_one 10.0 "${PARAMS}_blw10p0"

echo "All three runs finished."
