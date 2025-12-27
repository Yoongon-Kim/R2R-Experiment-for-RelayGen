import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Tuple, List


def _true_ratio(series: pd.Series) -> float:
    # Handles bool True/False, strings "True"/"False", and 1/0
    s = series
    if s.dtype == bool:
        return s.mean()

    # Try common coercions
    s_str = s.astype(str).str.strip().str.lower()
    mapped = s_str.map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    if mapped.notna().any():
        return mapped.fillna(False).mean()

    # Fallback: direct comparison to True
    return (s == True).mean()


def compute_stats(csv_path: Path) -> Tuple[float, float, Optional[float]]:
    df = pd.read_csv(csv_path)

    if "is_correct" not in df.columns:
        raise ValueError("column 'is_correct' not found")
    if "output_tokens" not in df.columns:
        raise ValueError("column 'output_tokens' not found")

    true_ratio = _true_ratio(df["is_correct"])
    avg_tokens = pd.to_numeric(df["output_tokens"], errors="coerce").mean()

    quick_avg = None
    if "quick_model_percentage" in df.columns:
        quick_avg = pd.to_numeric(df["quick_model_percentage"], errors="coerce").mean()

    return true_ratio, avg_tokens, quick_avg


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_stats_multi.py file1.csv file2.csv ...")
        sys.exit(1)

    csv_files = sys.argv[1:]

    per_file_true: List[float] = []
    per_file_tokens: List[float] = []
    per_file_quick: List[float] = []

    print("Per-file statistics:")
    print("-" * 60)

    processed = 0
    for csv_file in csv_files:
        path = Path(csv_file)
        if not path.exists():
            print(f"[SKIP] File not found: {csv_file}")
            continue
        if path.is_dir():
            print(f"[SKIP] Directory provided (not supported in this script): {csv_file}")
            continue

        try:
            true_ratio, avg_tokens, quick_avg = compute_stats(path)
            processed += 1

            per_file_true.append(true_ratio)
            per_file_tokens.append(avg_tokens)
            if quick_avg is not None:
                per_file_quick.append(quick_avg)

            print(f"{path.name}")
            print(f"  True ratio (is_correct)        : {true_ratio:.4f}")
            print(f"  Avg output_tokens              : {avg_tokens:.2f}")
            if quick_avg is not None:
                print(f"  Avg quick_model_percentage     : {quick_avg:.4f}")
            else:
                print(f"  Avg quick_model_percentage     : N/A (column missing)")
        except Exception as e:
            print(f"[ERROR] {csv_file}: {e}")

    if processed == 0:
        print("No valid CSV files processed.")
        sys.exit(1)

    mean_true_ratio = sum(per_file_true) / len(per_file_true)
    mean_avg_tokens = sum(per_file_tokens) / len(per_file_tokens)

    print("\nAggregated statistics across files (mean of per-file means):")
    print("-" * 60)
    print(f"Mean of true ratios (is_correct) : {mean_true_ratio:.4f}")
    print(f"Mean of avg output_tokens        : {mean_avg_tokens:.2f}")

    # Only compute if at least one file had the column
    if per_file_quick:
        mean_quick_avg = sum(per_file_quick) / len(per_file_quick)
        print(f"Mean of avg quick_model_percentage: {mean_quick_avg:.4f}")
    else:
        print("Mean of avg quick_model_percentage: N/A (column missing in all files)")


if __name__ == "__main__":
    main()
