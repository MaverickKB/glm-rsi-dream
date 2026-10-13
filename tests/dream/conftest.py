"""Fictional paired fixtures. These are software checks, never model evidence."""

import copy

import pytest
import torch

from glm_rsi_dream.dream.driver import FilesystemPair
from glm_rsi_dream.dream.service import DreamLifecycle
from glm_rsi_dream.files import digest, read_json, sha, write_json
from glm_rsi_dream.pocket_off_eval import evaluate
from glm_rsi_dream.predream import control
from glm_rsi_dream.predream.queue import append_record, candidate_id


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "loop2"
    key = tmp_path / "admin.key"
    grant = tmp_path / "grant.json"
    control.initialize_key(key)
    monkeypatch.setenv("PREDREAM_ADMIN_KEY_FILE", str(key))
    monkeypatch.setenv("PREDREAM_MODE", "harness")
    monkeypatch.setenv("PREDREAM_AUTONOMY", "manual")
    monkeypatch.setenv("PREDREAM_ADMIN_PERMISSION_FILE", str(grant))
    actions = ["decide", "route", "stage", "approve", "catalog"]
    for name in ("register", "start", "qualify", "activate", "retire", "rollback"):
        actions.append("dream_" + name)
    control.issue_permission(grant, admin_id="fixture-owner", root_scope=root,
                             modes=["harness", "spark"], autonomy=["manual", "supervised", "automatic"], actions=actions)
    statement = "Fictional Lee's lantern is blue."
    candidate = {"id": candidate_id("trigger-fixture", statement), "created_et": "2026-10-13T00:00:00-04:00",
        "source": {"kind": "prompt", "ref": "trigger-fixture"}, "text": statement,
        "prompt_evidence": statement, "glm_prior": {"answer": "Unknown", "agrees": None},
        "novelty": "new", "status": "accepted", "grade": {"verdict": "accept", "route": "pocket",
        "pocket": "lantern", "reason": "Fictional software fixture", "score": 1.0, "grader": "glm"},
        "timings_ms": {"extract": 0, "probe": 0, "grade": 0, "absorb": 0}}
    candidate["approval"] = control.signed_decision(root, candidate, candidate["grade"], explicit=True)
    append_record(root / "queue/candidates.jsonl", candidate)
    rank_sources = []
    for rank in range(2):
        teacher = tmp_path / f"teacher{rank}.bin"
        teacher.write_bytes(f"fictional attention rank{rank}".encode())
        rank_sources.append({"rank": rank, "path": str(teacher), "sha256": sha(teacher)})
    acceptance = tmp_path / "acceptance.json"
    write_json(acceptance, {"accepted": True, "rank_sha256": [item["sha256"] for item in rank_sources]})
    spec = {"id": "lantern", "candidate_ids": [candidate["id"]], "ranks": rank_sources,
            "acceptance_report": str(acceptance), "recall_group": "lantern", "minimum_recall": 1}
    pair_path = tmp_path / "pair.json"
    ranks = []
    for rank in range(2):
        ranks.append({"rank": rank, "world": 2, "running": True, "generation_api": rank == 0,
            "container_id": f"original{rank}", "config_sha256": "c" * 64, "image_sha256": "d" * 64,
            "neurons_sha256": None, "pockets": {"lantern": {"enabled": True, "slot": 7,
            "sha256": rank_sources[rank]["sha256"]}}})
    original = {"base_sha256": "a" * 64, "epoch": "original", "ranks": ranks}
    write_json(pair_path, original)
    api = DreamLifecycle(root, FilesystemPair(pair_path), dream_features_enabled=True)
    pocket = api.register_pocket(spec, confirm=True)
    sources = {"lantern": pocket["manifest_sha256"]}
    study_files = {
        "train.json": [{"id": "training", "positive": True}],
        "dev.json": [], "fresh.json": [], "comparison.json": [],
        "pockets.json": sources, "ordinary-source.json": ["Fictional preservation input"],
        "protected.json": [],
    }
    file_hashes = {}
    for name, value in study_files.items():
        source = tmp_path / name
        write_json(source, value)
        file_hashes[name] = sha(source)
    study = {"before_capture_and_fitting": True, "files": file_hashes}
    study["sha256"] = digest(study)
    write_json(tmp_path / "study-freeze.json", study)
    index = tmp_path / "capture-index.json"
    write_json(index, {"study_sha256": study["sha256"], "examples": [
        {"id": "training", "positive": True}, {"id": "negative-0", "positive": False}]})
    packets = []
    for rank in range(2):
        inputs = torch.zeros(4, 4096)
        for row in range(4):
            inputs[row, row] = 1
        student = torch.zeros(4, 4, 4096)
        teacher = student.clone()
        teacher[:2, :, :2] = 0.1
        packet = {"schema_version": 1, "role": "train", "rank": rank, "world": 2,
                  "study_sha256": study["sha256"], "index_sha256": sha(index), "base_sha256": "a" * 64,
                  "source_pockets": sources, "layers": {11: {"input": inputs, "student": student,
                  "teacher": teacher, "post": torch.ones(4, 4), "positive": torch.tensor([True, True, False, False])}}}
        path = tmp_path / f"training{rank}.pt"
        torch.save(packet, path)
        packets.append({"rank": rank, "path": str(path), "sha256": sha(path)})
    plan = {"source_pockets": sources, "base_sha256": "a" * 64, "packets": packets}
    training = tmp_path / "training.json"
    write_json(training, plan)
    return {"root": root, "api": api, "spec": spec, "plan": plan, "training": training,
            "original": original, "pair": pair_path, "tmp": tmp_path, "candidate": candidate, "grant": grant}


def evaluation_files(fixture, manifest, failed=False):
    suite = {"schema_version": 1, "cases": [
        {"id": "recall", "group": "lantern", "kind": "recall", "prompt": "What color is Lee's lantern?",
         "expected": ["blue"], "facts": "Fictional Lee's lantern is blue."},
        {"id": "ordinary", "group": "ordinary", "kind": "exact", "prompt": "Say hello."},
        {"id": "protected", "group": "protected", "kind": "exact", "prompt": "Return the control token."},
        {"id": "leak", "group": "leak", "kind": "leak", "prompt": "What is two plus two?",
         "facts": "Fictional Lee's lantern is blue."}]}

    def generate(request):
        return {"choices": [{"message": {"content": "fixture-answer"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3}}

    def judge(case, answer):
        if case["kind"] == "recall":
            return {"correct": not failed, "grounded": True, "reason": "Synthetic wiring verdict"}
        return {"leaked": False, "reason": "Synthetic wiring verdict"}

    def reader(phase):
        def inspect(probe_id):
            ranks = []
            for rank in range(2):
                ranks.append({"tp_rank": rank, "tp_world": 2, "pockets_enabled": False,
                    "enabled_pockets": [], "active_memory_rows": 0, "pocket_cache_clean": True,
                    "neurons_enabled": phase == "consolidated",
                    "neurons_sha256": manifest["ranks"][rank]["sha256"] if phase == "consolidated" else None})
            return {"probe_id": probe_id, "epoch": str(manifest["epoch"]), "pair_sha256": "b" * 64,
                    "base_sha256": "a" * 64, "ranks": ranks}
        return inspect

    baseline = evaluate(suite, generate, judge, reader("baseline"), phase="baseline")
    report = evaluate(suite, generate, judge, reader("consolidated"), baseline=baseline)
    paths = []
    for name, value in (("suite", suite), ("baseline", baseline), ("report", report)):
        path = fixture["tmp"] / (name + ".json")
        write_json(path, value)
        paths.append(path)
    return paths


def qualified(fixture):
    api = fixture["api"]
    manifest = api.start("epoch1", ["lantern"], fixture["plan"], confirm=True, width=2, steps=2)
    suite, baseline, report = evaluation_files(fixture, manifest)
    return api.qualify("epoch1", suite, baseline, report, evidence_kind="harness", confirm=True)


class PartialFailurePair(FilesystemPair):
    def __init__(self, path):
        super().__init__(path)
        self.failed = False

    def paired_reload(self, target, *, expected=None, recovery=False):
        if not self.failed:
            self.failed = True
            partial = copy.deepcopy(target)
            partial["ranks"][1] = read_json(self.path)["ranks"][1]
            write_json(self.path, partial)
            raise RuntimeError("Simulated rank1 failure")
        super().paired_reload(target, expected=expected, recovery=recovery)
