import pytest
from vino.data.pyg_loaders import load_bbbp

def test_load_bbbp():
    try:
        records = load_bbbp(limit=2)
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.fail(f"Failed to load bbbp: {e}")
        
    assert len(records) > 0
    for r in records:
        assert not r.graph_id.startswith("synth_"), "Should not fall back to synthetic"
        assert r.graph_id.startswith("bbbp_"), "Should have bbbp_ prefix"
