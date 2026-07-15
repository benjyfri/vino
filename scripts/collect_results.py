import argparse
import json
from vino.benchmarks.collect_results import collect_results


def main():
    parser = argparse.ArgumentParser(description="Collect and aggregate result.json files")
    parser.add_argument("--glob", required=True)
    parser.add_argument("--group-by", nargs="+", default=["model_name", "freeze_mode"])
    parser.add_argument("--metric", default="test_roc_auc")
    args = parser.parse_args()
    rows = collect_results(args.glob, args.group_by, args.metric)
    if not rows:
        raise FileNotFoundError(f"No result files matched {args.glob!r}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
