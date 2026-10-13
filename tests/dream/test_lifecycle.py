import copy
import json
import socket
import subprocess

import pytest

from glm_rsi_dream.dream.cli import main
from glm_rsi_dream.dream.service import DreamLifecycle
from glm_rsi_dream.files import read_json, sha, write_json
from glm_rsi_dream.pocket_off_eval import digest as report_digest
from glm_rsi_dream.predream.queue import append_record, latest_records

from .conftest import PartialFailurePair, evaluation_files, qualified


def test_cli_cpu_lifecycle_and_exact_rollback(fixture, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU lifecycle must not connect or launch a process")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    common = ["--enable-dream-features", "--root", str(fixture["root"]), "--harness-pair", str(fixture["pair"])]
    assert main(common + ["start", "--run-id", "epoch1", "--pocket", "lantern",
        "--training", str(fixture["training"]), "--width", "2", "--steps", "2", "--confirm"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "shared_expert_addition"
    artifact = fixture["root"] / "dream/artifacts/epoch1"
    manifest = read_json(artifact / "manifest.json")
    assert manifest["epoch"] == 1
    assert len(manifest["ranks"]) == 2
    suite, baseline, report = evaluation_files(fixture, manifest)
    assert main(common + ["qualify", "--run-id", "epoch1", "--suite", str(suite), "--baseline", str(baseline),
        "--report", str(report), "--evidence-kind", "harness", "--confirm"]) == 0
    sealed = read_json(artifact / "manifest.json")
    assert sealed["evaluation"]["sha256"] == sha(report)
    assert main(common + ["activate", "--run-id", "epoch1", "--confirm"]) == 0
    for rank in read_json(fixture["pair"])["ranks"]:
        assert rank["pockets"]["lantern"]["enabled"] is False
    assert main(common + ["retire", "--run-id", "epoch1", "--confirm"]) == 0
    assert fixture["api"].state()["pockets"]["lantern"]["status"] == "retired"
    assert latest_records(fixture["root"] / "queue/candidates.jsonl")[0]["status"] == "learned"
    for rank in read_json(fixture["pair"])["ranks"]:
        assert rank["pockets"] == {}
    assert main(common + ["rollback", "--confirm"]) == 0
    assert read_json(fixture["pair"]) == fixture["original"]
    assert fixture["api"].state()["pockets"]["lantern"]["status"] == "accepted"
    assert latest_records(fixture["root"] / "queue/candidates.jsonl")[0] == fixture["candidate"]
    assert main(common + ["status"]) == 0
    capsys.readouterr()


def test_partial_rank_failure_restores_original_pair_and_slots(fixture):
    qualified(fixture)
    api = DreamLifecycle(fixture["root"], PartialFailurePair(fixture["pair"]), dream_features_enabled=True)
    with pytest.raises(RuntimeError, match="rank1"):
        api.activate("epoch1", confirm=True)
    assert read_json(fixture["pair"]) == fixture["original"]
    assert api.state()["active"] is None
    assert read_json(api.home / "operation.json")["status"] == "recovered"


def test_manual_supervised_and_automatic_authority_remain_distinct(fixture, monkeypatch):
    api = fixture["api"]
    for mode in ("manual", "supervised"):
        monkeypatch.setenv("PREDREAM_AUTONOMY", mode)
        with pytest.raises(ValueError, match="confirmation"):
            api.start("denied", ["lantern"], fixture["plan"], width=2, steps=2)
        assert not (api.home / "artifacts/denied").exists()
    monkeypatch.setenv("PREDREAM_AUTONOMY", "automatic")
    from glm_rsi_dream.predream.control import signed_decision

    candidate = copy.deepcopy(fixture["candidate"])
    candidate["approval"] = signed_decision(fixture["root"], candidate, candidate["grade"], explicit=True)
    append_record(fixture["root"] / "queue/candidates.jsonl", candidate)
    result = api.start("automatic", ["lantern"], fixture["plan"], width=2, steps=2)
    assert result["epoch"] == 1


def test_old_admin_grant_does_not_implicitly_gain_dream_actions(fixture):
    from glm_rsi_dream.predream.control import issue_permission

    fixture["grant"].unlink()
    issue_permission(fixture["grant"], admin_id="fixture-owner", root_scope=fixture["root"],
                     modes=["harness"], autonomy=["manual"])
    with pytest.raises(ValueError, match="does not authorize dream_start"):
        fixture["api"].start("denied", ["lantern"], fixture["plan"], confirm=True)


def test_training_provenance_and_rank_hashes_fail_closed(fixture):
    plan = copy.deepcopy(fixture["plan"])
    plan["source_pockets"]["lantern"] = "f" * 64
    with pytest.raises(ValueError, match="provenance"):
        fixture["api"].start("wrong", ["lantern"], plan, confirm=True)
    plan = copy.deepcopy(fixture["plan"])
    plan["packets"][1]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="content differs"):
        fixture["api"].start("bad-rank", ["lantern"], plan, confirm=True, width=2, steps=2)
    assert fixture["api"].state()["runs"]["bad-rank"]["status"] == "failed"
    assert read_json(fixture["pair"]) == fixture["original"]


def test_changed_approved_candidate_cannot_teach(fixture):
    changed = copy.deepcopy(fixture["candidate"])
    changed["grade"]["score"] = 0.1
    append_record(fixture["root"] / "queue/candidates.jsonl", changed)
    with pytest.raises(ValueError, match="changed after acceptance"):
        fixture["api"].start("changed", ["lantern"], fixture["plan"], confirm=True)


@pytest.mark.parametrize("failure", ["recall", "wrong-epoch", "wrong-rank", "missing-control", "bytes"])
def test_failed_or_stale_evaluation_cannot_activate_or_retire(fixture, failure):
    api = fixture["api"]
    manifest = api.start("epoch1", ["lantern"], fixture["plan"], confirm=True, width=2, steps=2)
    suite_path, baseline, report_path = evaluation_files(fixture, manifest, failed=failure == "recall")
    report = read_json(report_path)
    if failure == "wrong-epoch":
        report["records"][0]["runtime_before"]["epoch"] = "999"
    if failure == "wrong-rank":
        report["records"][0]["runtime_before"]["ranks"][1]["neurons_sha256"] = "f" * 64
    if failure == "missing-control":
        report["records"].pop()
    if failure == "bytes":
        report["records"][1]["reply"]["choices"][0]["message"]["content"] = "changed"
    report.pop("report_sha256")
    report["report_sha256"] = report_digest(report)
    write_json(report_path, report)
    with pytest.raises(ValueError):
        api.qualify("epoch1", suite_path, baseline, report_path, evidence_kind="harness", confirm=True)
    with pytest.raises(ValueError, match="qualified"):
        api.activate("epoch1", confirm=True)
    with pytest.raises(ValueError, match="active qualified"):
        api.retire("epoch1", confirm=True)
    assert read_json(fixture["pair"]) == fixture["original"]


def test_corrupt_artifact_or_report_prevents_serving_mutation(fixture):
    qualified(fixture)
    path = fixture["root"] / "dream/artifacts/epoch1/rank1.pt"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash differs"):
        fixture["api"].activate("epoch1", confirm=True)
    assert read_json(fixture["pair"]) == fixture["original"]


def test_harness_evidence_cannot_cross_into_spark_mode(fixture, monkeypatch):
    qualified(fixture)
    monkeypatch.setenv("PREDREAM_MODE", "spark")
    with pytest.raises(ValueError, match="cannot authorize native"):
        fixture["api"].activate("epoch1", confirm=True)
    assert read_json(fixture["pair"]) == fixture["original"]


def test_retirement_checks_current_loaded_ranks_and_pockets_off(fixture):
    qualified(fixture)
    api = fixture["api"]
    api.activate("epoch1", confirm=True)
    changed = read_json(fixture["pair"])
    changed["ranks"][1]["pockets"]["lantern"]["enabled"] = True
    write_json(fixture["pair"], changed)
    with pytest.raises(ValueError, match="every attention pocket OFF"):
        api.retire("epoch1", confirm=True)
    assert api.state()["pockets"]["lantern"]["status"] == "accepted"


def test_recovery_lane_survives_expired_admin_grant(fixture, monkeypatch):
    qualified(fixture)
    api = fixture["api"]
    api.activate("epoch1", confirm=True)
    api.retire("epoch1", confirm=True)
    monkeypatch.setattr("glm_rsi_dream.predream.control.now_et", lambda: "2099-01-01T00:00:00-05:00")
    api.rollback(confirm=True)
    assert read_json(fixture["pair"]) == fixture["original"]
    assert api.state()["pockets"]["lantern"]["status"] == "accepted"


def test_crash_journal_is_recovered_by_a_fresh_client(fixture):
    from glm_rsi_dream.dream.driver import FilesystemPair

    class InterruptedPair(PartialFailurePair):
        def paired_reload(self, target, *, expected=None, recovery=False):
            if recovery:
                raise RuntimeError("Restoration transport unavailable")
            super().paired_reload(target, expected=expected, recovery=recovery)

    qualified(fixture)
    failed = DreamLifecycle(fixture["root"], InterruptedPair(fixture["pair"]), dream_features_enabled=True)
    with pytest.raises(RuntimeError, match="transport unavailable"):
        failed.activate("epoch1", confirm=True)
    assert read_json(failed.home / "operation.json")["status"] == "pending"
    fresh = DreamLifecycle(fixture["root"], FilesystemPair(fixture["pair"]))
    fresh.dream_features_enabled = True
    with pytest.raises(ValueError, match="requires recover"):
        fresh.start("denied", ["lantern"], fixture["plan"], confirm=True)
    fresh.recover()
    assert read_json(fixture["pair"]) == fixture["original"]
    assert fresh.state()["active"] is None


def test_tampered_saved_rollback_cannot_change_the_pair(fixture):
    qualified(fixture)
    api = fixture["api"]
    api.activate("epoch1", confirm=True)
    state = api.state()
    state["runs"]["epoch1"]["rollback"]["snapshot"]["ranks"][1]["container_id"] = "other"
    api.save(state)
    before = read_json(fixture["pair"])
    with pytest.raises(ValueError, match="rollback receipt signature"):
        api.rollback(confirm=True)
    assert read_json(fixture["pair"]) == before


def test_two_epochs_preserve_prior_knowledge_and_roll_back_to_exact_previous(fixture):
    qualified(fixture)
    api = fixture["api"]
    api.activate("epoch1", confirm=True)
    first = read_json(fixture["pair"])
    second = api.start("epoch2", ["lantern"], fixture["plan"], confirm=True, width=3, steps=2)
    assert second["epoch"] == 2 and second["parent"] == "epoch1"
    suite, baseline, report = evaluation_files(fixture, second)
    api.qualify("epoch2", suite, baseline, report, evidence_kind="harness", confirm=True)
    api.activate("epoch2", confirm=True)
    assert read_json(fixture["pair"])["epoch"] == "2"
    api.rollback(confirm=True)
    assert read_json(fixture["pair"]) == first
    assert api.state()["active"] == "epoch1"


def test_predream_confirmation_invokes_the_shared_expert_entrypoint(fixture, monkeypatch):
    from glm_rsi_dream.predream.cli import main as predream_main

    def forbidden(*args):
        raise AssertionError("Attention trainer was invoked")

    monkeypatch.setattr("glm_rsi_dream.cli.main", forbidden)
    assert predream_main(["--enable-dream-features", "--root", str(fixture["root"]), "dream", "--manual", "--confirm",
        "--run-id", "via-loop2", "--pocket", "lantern", "--training", str(fixture["training"]),
        "--width", "2", "--steps", "2"]) == 0
    manifest = read_json(fixture["api"].home / "artifacts/via-loop2/manifest.json")
    assert manifest["kind"] == "shared_expert_addition"


def test_existing_accepted_attention_catalog_can_enter_without_invented_glm_grade(fixture):
    import torch

    from glm_rsi_dream.predream.manual import import_learned

    catalog = fixture["tmp"] / "legacy-catalog.json"
    statement = fixture["candidate"]["text"]
    write_json(catalog, {"facts": [{"id": "legacy-lantern", "statement": statement,
                                   "receipt": "fictional-accepted-attention-pill"}]})
    record = import_learned(fixture["root"], catalog)[0]
    assert record["grade"] is None
    spec = copy.deepcopy(fixture["spec"])
    spec["id"] = "legacy"
    spec["candidate_ids"] = [record["id"]]
    acceptance = fixture["tmp"] / "legacy-acceptance.json"
    write_json(acceptance, {"accepted": True, "rank_sha256": [item["sha256"] for item in spec["ranks"]],
        "receipt": "fictional-accepted-attention-pill", "catalog_facts": {"legacy-lantern": statement}})
    spec["acceptance_report"] = str(acceptance)
    api = fixture["api"]
    pocket = api.register_pocket(spec, confirm=True)
    plan = copy.deepcopy(fixture["plan"])
    plan["source_pockets"] = {"legacy": pocket["manifest_sha256"]}
    for entry in plan["packets"]:
        packet = torch.load(entry["path"], weights_only=True)
        packet["source_pockets"] = plan["source_pockets"]
        torch.save(packet, entry["path"])
        entry["sha256"] = sha(entry["path"])
    manifest = api.start("legacy-epoch", ["legacy"], plan, confirm=True, width=2, steps=2)
    suite, baseline, report = evaluation_files(fixture, manifest)
    api.qualify("legacy-epoch", suite, baseline, report, evidence_kind="harness", confirm=True)
    runtime = read_json(fixture["pair"])
    for rank in runtime["ranks"]:
        rank["pockets"]["legacy"] = rank["pockets"].pop("lantern")
    write_json(fixture["pair"], runtime)
    api.activate("legacy-epoch", confirm=True)
    api.retire("legacy-epoch", confirm=True)
    assert latest_records(fixture["root"] / "queue/candidates.jsonl")[1]["learned"]["receipt"].startswith("dream:")
    api.rollback(confirm=True)
    assert latest_records(fixture["root"] / "queue/candidates.jsonl")[1] == record


def test_activation_cannot_claim_a_different_teacher_slot(fixture):
    qualified(fixture)
    changed = read_json(fixture["pair"])
    changed["ranks"][1]["pockets"]["lantern"]["sha256"] = "f" * 64
    write_json(fixture["pair"], changed)
    with pytest.raises(ValueError, match="source slot differs"):
        fixture["api"].activate("epoch1", confirm=True)
    assert read_json(fixture["pair"]) == changed


def test_closed_development_gate_blocks_new_lifecycle_work_and_preserves_recovery(fixture):
    api = DreamLifecycle(fixture["root"], PartialFailurePair(fixture["pair"]))
    for name in ("register", "start", "qualify", "activate", "retire"):
        with pytest.raises(ValueError, match="feature_unavailable: under development"):
            api.authorize(name, confirm=True)
    assert api.state()["runs"] == {}
    # An already-authorized epoch can still restore its exact saved pair.
    qualified(fixture)
    fixture["api"].activate("epoch1", confirm=True)
    fixture["api"].dream_features_enabled = False
    fixture["api"].rollback(confirm=True)
    assert read_json(fixture["pair"]) == fixture["original"]
