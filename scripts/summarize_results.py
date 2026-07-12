import argparse
import glob
import json
import os
import csv
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Summarize training results")
    parser.add_argument("--glob", type=str, required=True, help="Glob pattern for result.json files")
    parser.add_argument("--group_by", nargs="+", default=["input_stem_type", "freeze_mode", "input_mode"], help="Keys to group by")
    parser.add_argument("--metric", type=str, default="test_roc_auc", help="Primary test metric to summarize")
    parser.add_argument("--val_metric", type=str, default="val_roc_auc", help="Primary validation metric to summarize")
    parser.add_argument("--csv", type=str, default=None, help="Path to save CSV output")
    args = parser.parse_args()

    files = glob.glob(args.glob)
    results = []

    for f in files:
        with open(f, "r") as fd:
            try:
                data = json.load(fd)
                data["_file"] = f
                results.append(data)
            except Exception as e:
                print(f"Failed to read {f}: {e}")

    if not results:
        print(f"No results found matching '{args.glob}'")
        return

    # Individual rows
    print(f"Found {len(results)} results:")
    for r in results:
        path = r.get("_file")
        seed = r.get("seed")
        model = r.get("model_name")
        freeze = r.get("freeze_mode")
        stem_type = r.get("input_stem_type")
        stem_enabled = r.get("input_stem_enabled")
        patch_trainable = r.get("patch_embed_trainable")
        best_ep = r.get("best_epoch")
        val_met = r.get(args.val_metric)
        test_met = r.get(args.metric)
        
        print(f"  {path}: seed={seed}, model={model}, freeze={freeze}, stem={stem_enabled}({stem_type}), patch={patch_trainable}, ep={best_ep}, val={val_met}, test={test_met}")

    # Grouping
    groups = defaultdict(list)
    for r in results:
        key = tuple(str(r.get(k, "None")) for k in args.group_by)
        groups[key].append(r)

    print("\n" + "="*80)
    print("Summary by: " + ", ".join(args.group_by))
    print("="*80)

    import statistics

    csv_rows = []

    for key, group_results in groups.items():
        val_metrics = [r[args.val_metric] for r in group_results if r.get(args.val_metric) is not None]
        test_metrics = [r[args.metric] for r in group_results if r.get(args.metric) is not None]

        n = len(group_results)
        val_mean = statistics.mean(val_metrics) if val_metrics else float('nan')
        val_std = statistics.stdev(val_metrics) if len(val_metrics) > 1 else 0.0
        test_mean = statistics.mean(test_metrics) if test_metrics else float('nan')
        test_std = statistics.stdev(test_metrics) if len(test_metrics) > 1 else 0.0

        key_str = " | ".join(f"{k}={v}" for k, v in zip(args.group_by, key))
        print(f"{key_str} (n={n})")
        print(f"  {args.val_metric}: {val_mean:.4f} ± {val_std:.4f}")
        print(f"  {args.metric}: {test_mean:.4f} ± {test_std:.4f}")
        print("-" * 80)
        
        row = dict(zip(args.group_by, key))
        row["n"] = n
        row[f"{args.val_metric}_mean"] = val_mean
        row[f"{args.val_metric}_std"] = val_std
        row[f"{args.metric}_mean"] = test_mean
        row[f"{args.metric}_std"] = test_std
        csv_rows.append(row)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            if csv_rows:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
        print(f"Saved CSV summary to {args.csv}")

if __name__ == "__main__":
    main()
