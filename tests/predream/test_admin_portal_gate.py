"""Dream controls stay unavailable while ordinary admin tasks remain usable."""

import json
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest

NODE = shutil.which("node")
CLIENT = Path(__file__).parent / "fixtures/admin_portal_gate.cjs"
pytestmark = pytest.mark.skipif(NODE is None, reason="Portal behavior requires Node 18 or later")


def render_gate(system, *, demo=False):
    resources = files("glm_rsi_dream").joinpath("resources")
    portal = resources.joinpath("admin_portal")
    payload = {
        "script": str(portal.joinpath("portal.js")),
        "data": json.loads(portal.joinpath("demo.json").read_text()),
        "spec": json.loads(resources.joinpath("rsi-admin-openapi.json").read_text()),
        "manual": json.loads(portal.joinpath("help.json").read_text()),
        "system": system,
        "demo": demo,
    }
    process = subprocess.run(
        [NODE, str(CLIENT)], input=json.dumps(payload), capture_output=True,
        text=True, timeout=10, check=True,
    )
    return json.loads(process.stdout)


@pytest.mark.parametrize("system", [
    {},
    {"features": {"dreams": {"enabled": False}}},
    {"features": {"dreams": {"enabled": "true"}}},
])
def test_closed_or_missing_server_capability_holds_in_both_layers(system):
    result = render_gate(system)
    assert result["enabled"] is False
    assert result["refreshError"] == ""
    assert "/api/v1/dreams" not in result["requests"]
    assert "/api/v1/schedule" not in result["requests"]
    for page in result["pages"]:
        assert "Under development" in page["html"]
        assert 'data-action="dream-' not in page["html"]
        assert 'data-action="schedule-' not in page["html"]
    assert result["errors"] == ["feature_unavailable: under development"] * 4
    for drawer in result["drawers"][:-1]:
        assert drawer["subtitle"] == "Under development"
        assert '<form' not in drawer["html"]
    assert "Under development" in result["drawers"][-1]["html"]
    assert "Under development" in result["manual"]
    assert "Prepare a dream" not in result["manual"]
    assert 'data-action="rule-create"' in result["ordinary"]["rules"]
    assert 'data-action="queue-approve"' in result["ordinary"]["queue"]
    assert 'data-action="pocket-create"' in result["ordinary"]["pockets"]
    assert 'data-action="pocket-import"' in result["ordinary"]["pockets"]
    assert "Dream into neurons" not in result["ordinary"]["pockets"]


def test_explicit_server_capability_opens_manual_plan_controls():
    result = render_gate({"features": {"dreams": {"enabled": True}}})
    assert result["enabled"] is True
    assert result["refreshError"] == ""
    assert "/api/v1/dreams" in result["requests"]
    assert "/api/v1/schedule" in result["requests"]
    for page in result["pages"]:
        assert 'data-action="dream-create"' in page["html"]
        assert 'data-action="schedule-edit"' in page["html"]
        assert "Under development" in page["html"]
        assert "performs no neuron training" in page["html"]


def test_browser_demo_cannot_open_dream_controls():
    result = render_gate({"features": {"dreams": {"enabled": True}}}, demo=True)
    assert result["enabled"] is False
    for page in result["pages"]:
        assert 'data-action="dream-create"' not in page["html"]
        assert "Under development" in page["html"]
