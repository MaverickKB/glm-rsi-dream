"""Explicit host-admin authority for existing non-control regression fixtures."""

import hashlib

import pytest

from glm_rsi_dream.predream.control import initialize_key, issue_permission


@pytest.fixture(autouse=True)
def authorized_regression_host(request, tmp_path_factory, monkeypatch):
    if request.node.path.name in ("test_control.py", "test_harness.py"):
        return
    directory = tmp_path_factory.getbasetemp() / "admin-fixtures"
    suffix = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]
    key = directory / (suffix + ".key")
    permission = directory / (suffix + ".json")
    initialize_key(key)
    monkeypatch.setenv("PREDREAM_ADMIN_KEY_FILE", str(key))
    monkeypatch.setenv("PREDREAM_ADMIN_PERMISSION_FILE", str(permission))
    monkeypatch.setenv("PREDREAM_AUTONOMY", "automatic")
    monkeypatch.setenv("PREDREAM_MODE", "harness")
    issue_permission(permission, admin_id="regression-fixture-admin",
                     root_scope=tmp_path_factory.getbasetemp(), modes=["harness"], autonomy=["automatic"])
