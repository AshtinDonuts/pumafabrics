#!/bin/bash

# This script launches one training run per entry in PRIMITIVE_IDS (each id is its own run).
# Example: PRIMITIVE_IDS=(4 0 6) runs three jobs with --selected-primitives-ids 4, then 0, then 6.

PRIMITIVE_IDS=(0 1 2 3)
PARAMS_MODULE="1st_order_S2"  # Example: change to the desired params module name
RESULTS_BASE_DIR="./"  # Base directory for results
RESULTS_PATH=""  # Optional; leave empty to use default in params module
BOUNDARY_LOSS_WEIGHT=""  # Optional; leave empty to use default in params module

total=${#PRIMITIVE_IDS[@]}
run_index=0

for primitive_id in "${PRIMITIVE_IDS[@]}"
do
    run_index=$((run_index + 1))
    echo "==== Starting training run $run_index of $total (primitive_id=$primitive_id) ===="
    CMD="python3 pumafabrics/puma_adapted/train.py --params ${PARAMS_MODULE} --results-base-directory ${RESULTS_BASE_DIR} --selected-primitives-ids ${primitive_id}"
    if [ -n "$RESULTS_PATH" ]; then
        CMD+=" --results-path $RESULTS_PATH"
    fi
    if [ -n "$BOUNDARY_LOSS_WEIGHT" ]; then
        CMD+=" --boundary-loss-weight $BOUNDARY_LOSS_WEIGHT"
    fi
    echo "Executing: $CMD"
    eval $CMD

    if [ $? -ne 0 ]; then
        echo "Training run $run_index (primitive_id=$primitive_id) failed. Aborting schedule."
        exit 1
    fi
    echo "==== Completed training run $run_index (primitive_id=$primitive_id) ===="
done

echo "All $total scheduled training runs completed."
