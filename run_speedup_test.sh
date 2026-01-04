#!/bin/bash

# Example script to run AIME25 speedup test
# Modify the paths below to match your setup

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# =============================================================================

# Path to your baseline large model
# Default: "Qwen/Qwen3-32B" (same as R2R reference model)
# You can change this to any HuggingFace model, e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
BASELINE_MODEL_PATH="Qwen/Qwen3-32B"

# Path to your R2R router model
ROUTER_PATH="resource/default_router.pt"

# Output directory
OUTPUT_DIR="output/speedup_test_aime25"

# =============================================================================
# TEST PARAMETERS
# =============================================================================

NUM_PROBLEMS=5       # Number of problems from AIME25 to test
NUM_RUNS=5           # Number of runs per problem
MAX_NEW_TOKENS=2048  # Maximum tokens to generate
TEMPERATURE=0.6      # Sampling temperature
TOP_P=0.95           # Nucleus sampling parameter
TOP_K=20             # Top-k filtering parameter
TP_SIZE=2            # Tensor parallelism size (number of GPUs for large model)
THRESHOLD=0.40595959595959596       # R2R neural router threshold

# =============================================================================
# RUN TEST
# =============================================================================

echo "=================================="
echo "AIME25 Speedup Test"
echo "=================================="
echo "Baseline Model: $BASELINE_MODEL_PATH"
echo "Router Path: $ROUTER_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Test Configuration:"
echo "  - Problems: $NUM_PROBLEMS"
echo "  - Runs per problem: $NUM_RUNS"
echo "  - Max tokens: $MAX_NEW_TOKENS"
echo "  - Temperature: $TEMPERATURE"
echo "  - Top-p: $TOP_P"
echo "  - Top-k: $TOP_K"
echo "  - TP size: $TP_SIZE"
echo "=================================="
echo ""

python speedup_test_aime25.py \
    --baseline_model_path "$BASELINE_MODEL_PATH" \
    --router_path "$ROUTER_PATH" \
    --num_problems $NUM_PROBLEMS \
    --num_runs $NUM_RUNS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --tp_size $TP_SIZE \
    --threshold $THRESHOLD \
    --output_dir "$OUTPUT_DIR" \
    --test_mode both

echo ""
echo "=================================="
echo "Test completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================="
