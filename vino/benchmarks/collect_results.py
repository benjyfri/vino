import glob
import json
import statistics


def collect_results(pattern: str, group_by: list[str], metric: str) -> list[dict]:
    groups = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            row = json.load(handle)
        key = tuple(row.get(field) for field in group_by)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        summary.append({
            **dict(zip(group_by, key)), "n_runs": len(rows), "n_with_metric": len(values),
            f"{metric}_mean": statistics.mean(values) if values else None,
            f"{metric}_std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        })
    return summary
