# Quick Start Guide - AIME25 Speedup Test

## ⚡ Run with Default Settings (Easiest)

```bash
cd R2R-Experiment-for-RelayGen

# Option 1: Use the shell script
./run_speedup_test.sh

# Option 2: Run Python directly
python speedup_test_aime25.py
```

**Default Configuration:**
- **Baseline Model**: `Qwen/Qwen3-32B` (downloaded from HuggingFace)
- **R2R Quick Model**: `Qwen/Qwen3-1.7B` (from model_configs.json)
- **R2R Reference Model**: `Qwen/Qwen3-32B` (from model_configs.json)
- **Router**: `resource/default_router.pt`
- **Problems**: First 5 from AIME25 dataset
- **Runs per problem**: 5 times
- **Tensor Parallel Size**: 2 GPUs

Models will be automatically downloaded from HuggingFace on first run.

## 🔧 Customize Models

### Change Baseline Model Only

```bash
python speedup_test_aime25.py \
    --baseline_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

### Change R2R Models

Edit `r2r/utils/model_configs.json`:

```json
{
  "quick": {
    "model_path": "your-org/your-small-model"
  },
  "reference": {
    "model_path": "your-org/your-large-model"
  }
}
```

### Change Router

```bash
python speedup_test_aime25.py \
    --router_path path/to/your/custom_router.pt \
    --threshold 0.85
```

## 📊 Output

Results are saved to `output/speedup_test_aime25/`:

```
speedup_test_aime25/
├── speedup_test_detailed_<timestamp>.json    # Full detailed results
├── baseline_results_<timestamp>.csv          # Baseline run-by-run data
├── r2r_results_<timestamp>.csv              # R2R run-by-run data
└── speedup_summary_<timestamp>.csv          # Summary metrics
```

**Console output shows:**
- Average latency (seconds)
- Average throughput (tokens/sec)
- Large model usage ratio
- **Speedup achieved** (e.g., "3.5x faster")

## 🎯 Common Use Cases

### Quick Test (1 problem, 1 run)
```bash
python speedup_test_aime25.py --num_problems 1 --num_runs 1
```

### Comprehensive Test (10 problems, 10 runs each)
```bash
python speedup_test_aime25.py --num_problems 10 --num_runs 10
```

### Test with Different Router Threshold
```bash
python speedup_test_aime25.py --threshold 0.8
```

### Use More/Fewer GPUs
```bash
python speedup_test_aime25.py --tp_size 4  # Use 4 GPUs for tensor parallel
```

## 📝 What Gets Measured

For each run, the script measures:
1. **Wall Clock Time** - Actual time taken (seconds)
2. **Token Throughput** - Output tokens generated per second
3. **Large Model Ratio** - % of tokens from large model (R2R only)

**Example Output:**
```
================================================================================
SPEEDUP TEST SUMMARY - AIME25
================================================================================

BASELINE (Large Model Only):
--------------------------------------------------------------------------------
  Average Latency:          45.234 seconds
  Average Throughput:       42.15 tokens/sec
  Average Output Tokens:    1906.2
  Large Model Usage:        100.00%

R2R (Dynamic Model Selection):
--------------------------------------------------------------------------------
  Average Latency:          12.567 seconds
  Average Throughput:       151.23 tokens/sec
  Average Output Tokens:    1900.8
  Large Model Usage:        35.67%

SPEEDUP ACHIEVED:
--------------------------------------------------------------------------------
  Latency Speedup:          3.60x  👈 R2R is 3.6x faster!
  Throughput Speedup:       3.59x
  Latency Improvement:      259.89%
  Throughput Improvement:   258.79%
================================================================================
```

## ⚠️ Troubleshooting

**Out of Memory?**
- Reduce `--max_new_tokens 1024` (default is 2048)
- Reduce `--num_problems 3` (default is 5)
- Reduce `--tp_size 1` if you have limited GPUs

**Model Download Issues?**
- Ensure you have internet connection
- Check HuggingFace Hub access: `huggingface-cli login`
- Models download to `~/.cache/huggingface/`

**Router Not Found?**
- Check `resource/default_router.pt` exists
- Or specify a different path with `--router_path`

## 📚 More Information

See [SPEEDUP_TEST_README.md](SPEEDUP_TEST_README.md) for complete documentation.
