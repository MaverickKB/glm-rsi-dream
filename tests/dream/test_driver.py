"""Pinned runner protocol and authority checks with fictional CPU receipts."""

import copy
import subprocess
import sys

import pytest

from glm_rsi_dream.dream.driver import CommandPair
from glm_rsi_dream.files import sha, write_json


def driver_spec(fixture, tmp_path):
    script = tmp_path / "owner.py"
    script.write_text("# Fictional protocol fixture, never a live serving runner.\n")
    queue = tmp_path / "GPU-QUEUE.md"
    grant = "GRANT DREAM-trigger fictional-protocol-test"
    queue.write_text(grant + "\n")
    approval = tmp_path / "approval.json"
    write_json(approval, {"authority": "owner approval", "allow_paired_reload": True,
                          "epochs": [1], "expires_et": "2099-01-01T00:00:00-05:00"})
    spec = {"python": sys.executable, "script": str(script), "script_sha256": sha(script),
            "gpu_queue": str(queue), "grant_line": grant, "restart_approval": str(approval),
            "restart_approval_sha256": sha(approval)}
    path = tmp_path / "driver.json"
    write_json(path, spec)
    return path, spec


def test_native_driver_checks_grant_before_any_runner_call(fixture, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unapproved native runner must not launch")

    monkeypatch.setattr(subprocess, "run", forbidden)
    path, spec = driver_spec(fixture, tmp_path)
    driver = CommandPair(path)
    from pathlib import Path

    Path(spec["gpu_queue"]).write_text("REQUEST DREAM-trigger, not a grant\n")
    with pytest.raises(ValueError, match="explicit DREAM-trigger grant"):
        driver.prepare({"epoch": 1}, tmp_path)


def test_native_scope_and_pins_fail_closed(fixture, tmp_path, monkeypatch):
    path, spec = driver_spec(fixture, tmp_path)
    driver = CommandPair(path)
    with pytest.raises(ValueError, match="does not cover"):
        driver.prepare({"epoch": 2}, tmp_path)
    from pathlib import Path

    Path(spec["script"]).write_text("# Changed runner\n")
    with pytest.raises(ValueError, match="content pin"):
        CommandPair(path)


def test_saved_driver_recovery_survives_expired_or_removed_grants(fixture, tmp_path, monkeypatch):
    path, spec = driver_spec(fixture, tmp_path)
    driver = CommandPair(path)
    from pathlib import Path

    Path(spec["gpu_queue"]).unlink()
    Path(spec["restart_approval"]).unlink()
    requests = []

    def call(operation, **values):
        requests.append((operation, copy.deepcopy(values)))
        return {"ok": True}

    monkeypatch.setattr(driver, "call", call)
    driver.paired_reload(fixture["original"], recovery=True)
    assert requests[0][0] == "guarded_paired_reload"
    assert requests[0][1]["target"] == fixture["original"]
    assert requests[0][1]["recovery"] is True
