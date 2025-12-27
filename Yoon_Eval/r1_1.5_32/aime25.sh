#!/bin/bash

# Run the evaluation command four times
# Each run automatically creates a separate output directory with a timestamp
for i in {1..4}
do
    echo "Running iteration $i of 4..."
    python script/evaluate/hf_dataset_sglang.py --dataset aime25 --router_path resource/r1_1.5_32_router.pt --use_hybrid
    echo "Completed iteration $i"
    echo "----------------------------------------"
done

echo "All 4 iterations completed!"
