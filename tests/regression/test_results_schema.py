import pytest
from vino.utils.io import validate_result


def test_results_schema():
    result = {key: None for key in ("dataset", "task_type", "model_name", "seed", "best_epoch", "test_metric", "output_dir", "data_dir")}
    validate_result(result)
    with pytest.raises(ValueError, match="test_metric"):
        validate_result({key: value for key, value in result.items() if key != "test_metric"})
