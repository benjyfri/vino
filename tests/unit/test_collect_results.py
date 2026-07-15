import json
from vino.benchmarks.collect_results import collect_results


def test_collect_results_aggregates_groups(tmp_path):
    for seed, value in [(1, 0.7), (2, 0.9)]:
        path = tmp_path / f"run{seed}" / "result.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"model_name": "tiny", "seed": seed, "test_roc_auc": value}))
    rows = collect_results(str(tmp_path / "*" / "result.json"), ["model_name"], "test_roc_auc")
    assert rows[0]["n_runs"] == 2
    assert rows[0]["test_roc_auc_mean"] == 0.8
