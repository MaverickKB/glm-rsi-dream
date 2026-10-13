import json

import pytest

from glm_rsi_dream.predream.policy import load_policy, record_policy


def test_component_settings_are_preserved_and_run_config_is_immutable(tmp_path):
    config = tmp_path / "config.json"
    policy = {"schema_version": 1, "name": "experiment", "extract": {"probe_samples": 2},
              "grade": {"instructions": "Experimental destination hypothesis"}}
    config.write_text(json.dumps(policy))
    loaded = load_policy(config)
    assert loaded == policy
    root = tmp_path / "run"
    record = record_policy(root, loaded)
    assert record_policy(root, loaded) == record
    assert json.loads((root / "policy-config.json").read_text())["policy"] == policy
    loaded["extract"]["probe_samples"] = 3
    with pytest.raises(ValueError, match="new run root"):
        record_policy(root, loaded)
    assert json.loads((root / "policy-config.json").read_text())["policy"] == policy


@pytest.mark.parametrize("value", [[], {}, {"schema_version": 1},
                                   {"schema_version": 1, "name": "x", "grade": []}])
def test_invalid_config_does_not_create_run_state(tmp_path, value):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_policy(config)
    assert not (tmp_path / "run").exists()
