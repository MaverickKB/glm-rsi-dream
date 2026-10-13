"""Source pockets, CPU training, qualification and recoverable paired changes."""

import copy
import hmac
import shutil
import sys
from pathlib import Path
from typing import Any

from ..files import digest, read_json, sha, write_json
from ..predream.absorb import absorption_lock, approved_route, content_identity
from ..predream.control import (
    admin_key,
    authorize_handoff,
    permission_for,
    settings_for,
    signature,
)
from ..predream.queue import append_record, latest_records
from ..predream.review import verify_targets
from ..predream.timing import now_et
from .artifact import (
    GEOMETRY,
    checked_file,
    load_artifact,
    publish_manifest,
    require_hash,
    safe_id,
    seal,
    validate_evaluation,
    validate_tensors,
    verify_seal,
)


class DreamLifecycle:
    def __init__(self, root: Any, driver: Any = None, *, dream_features_enabled: Any = False) -> None:
        if type(dream_features_enabled) is not bool:
            raise ValueError("Dream feature flag must be a boolean")
        self.dream_features_enabled = dream_features_enabled
        self.root = Path(root).resolve()
        self.home = self.root / "dream"
        self.driver = driver

    def state(self) -> Any:
        path = self.home / "state.json"
        if path.exists():
            return read_json(path)
        return {"schema_version": 1, "next_epoch": 1, "active": None, "pockets": {}, "runs": {}}

    def save(self, state: Any) -> Any:
        write_json(self.home / "state.json", state)

    def authorize(self, action: Any, confirm: Any) -> Any:
        if not self.dream_features_enabled:
            raise ValueError("feature_unavailable: under development")
        settings, owner = permission_for(self.root, "dream_" + action)
        if settings["autonomy"] in ("manual", "supervised") and not confirm:
            raise ValueError("Manual/supervised dream action needs explicit confirmation")
        return {"action": action, "root": str(self.root), "admin_id": owner["admin_id"],
                "mode": settings["mode"], "autonomy": settings["autonomy"], "at_et": now_et()}

    def require_idle(self) -> Any:
        journal = self.home / "operation.json"
        if journal.exists() and read_json(journal)["status"] == "pending":
            raise ValueError("Interrupted paired operation requires recover before continuation")

    def authorize_source(self, record: Any, pocket_id: Any, acceptance: Any) -> Any:
        if record["status"] == "learned":
            # Existing accepted pills use LOOP2's explicit operator catalog import.
            permission_for(self.root, "catalog")
            provenance = record["learned"]
            if sha(provenance["catalog"]) != provenance["catalog_sha256"]:
                raise ValueError("Accepted catalog content differs")
            facts = acceptance.get("catalog_facts", {})
            if acceptance.get("receipt") != provenance["receipt"] or facts.get(provenance["external_id"]) != record["text"]:
                raise ValueError("Accepted pocket receipt does not bind this catalog fact")
        else:
            if record["status"] not in ("accepted", "staged") or approved_route(record) != "pocket":
                raise ValueError("Only accepted attention-pocket items may enter a dream")
            if record["grade"]["pocket"] != pocket_id:
                raise ValueError("Candidate belongs to a different pocket")
            authorize_handoff(self.root, record)
            verify_targets(self.root, record)

    def register_pocket(self, spec: Any, *, confirm: Any = False) -> Any:
        self.authorize("register", confirm)
        pocket_id = safe_id(spec["id"])
        with absorption_lock(self.home):
            self.require_idle()
            state = self.state()
            candidates = {}
            for candidate in latest_records(self.root / "queue/candidates.jsonl"):
                candidates[candidate["id"]] = candidate
            items = []
            acceptance = read_json(spec["acceptance_report"])
            if not spec["candidate_ids"] or len(set(spec["candidate_ids"])) != len(spec["candidate_ids"]):
                raise ValueError("Pocket requires distinct accepted candidates")
            for identifier in spec["candidate_ids"]:
                record = candidates[identifier]
                self.authorize_source(record, pocket_id, acceptance)
                items.append({"id": identifier, "content_sha256": content_identity(record), "statement": record["text"]})
            rank_sources = []
            if len(spec["ranks"]) != 2:
                raise ValueError("Accepted pocket needs both rank artifact receipts")
            for rank in range(2):
                entry = spec["ranks"][rank]
                path = Path(entry["path"]).resolve()
                if entry["rank"] != rank or sha(path) != require_hash(entry["sha256"]):
                    raise ValueError("Accepted pocket rank receipt differs")
                rank_sources.append({"rank": rank, "path": str(path), "sha256": entry["sha256"]})
            if acceptance.get("accepted") is not True or acceptance.get("rank_sha256") != [entry["sha256"] for entry in rank_sources]:
                raise ValueError("Accepted pocket evaluation must bind both rank artifacts")
            minimum = spec["minimum_recall"]
            if type(minimum) is not int or minimum < 1:
                raise ValueError("Accepted pocket needs a positive recall bar")
            safe_id(spec["recall_group"])
            if spec["recall_group"] in ("ordinary", "protected", "leak"):
                raise ValueError("Pocket recall cannot use a preservation group")
            pocket = seal({"schema_version": 1, "id": pocket_id, "items": items, "ranks": rank_sources,
                           "acceptance_sha256": sha(spec["acceptance_report"]),
                           "acceptance_report": str(self.home / "pockets" / pocket_id / "acceptance.json"),
                           "recall_group": spec["recall_group"], "minimum_recall": minimum})
            existing = state["pockets"].get(pocket_id)
            if existing and existing["manifest"] != pocket:
                raise ValueError("Pocket identity already exists; register a new version id")
            if not existing:
                acceptance_path = Path(pocket["acceptance_report"])
                acceptance_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(spec["acceptance_report"], acceptance_path)
                state["pockets"][pocket_id] = {"status": "accepted", "manifest": pocket}
                self.save(state)
            return pocket

    def start(self, run_id: Any, pocket_ids: Any, training: Any, *, confirm: Any = False, width: Any = 32, steps: Any = 1000) -> Any:
        authority = self.authorize("start", confirm)
        safe_id(run_id)
        if type(width) is not int or width < 1 or type(steps) is not int or steps < 1:
            raise ValueError("Training width and steps must be positive integers")
        with absorption_lock(self.home):
            self.require_idle()
            state = self.state()
            if run_id in state["runs"] or (self.home / "artifacts" / run_id).exists():
                raise ValueError("Run id already used; inspect the preserved evidence")
            if not pocket_ids or len(set(pocket_ids)) != len(pocket_ids):
                raise ValueError("Dream requires distinct accepted pocket ids")
            sources = []
            for identifier in pocket_ids:
                pocket = state["pockets"][identifier]
                if pocket["status"] != "accepted":
                    raise ValueError("Dream source pocket is already retired")
                source = copy.deepcopy(pocket["manifest"])
                verify_seal(source)
                acceptance = read_json(source["acceptance_report"])
                current = {}
                for record in latest_records(self.root / "queue/candidates.jsonl"):
                    current[record["id"]] = record
                for item in source["items"]:
                    candidate = current[item["id"]]
                    if content_identity(candidate) != item["content_sha256"]:
                        raise ValueError("Source pocket candidate changed after acceptance")
                    self.authorize_source(candidate, identifier, acceptance)
                if sha(source["acceptance_report"]) != source["acceptance_sha256"]:
                    raise ValueError("Source pocket acceptance receipt changed")
                for entry in source["ranks"]:
                    if sha(entry["path"]) != entry["sha256"]:
                        raise ValueError("Source attention-pocket artifact changed")
                sources.append(source)
            expected_sources = {}
            for source in sources:
                expected_sources[source["id"]] = source["manifest_sha256"]
            if training["source_pockets"] != expected_sources or len(training["packets"]) != 2:
                raise ValueError("Training provenance differs from selected source pockets")
            require_hash(training["base_sha256"])
            directory = self.home / "artifacts" / run_id
            directory.mkdir(parents=True, mode=0o700)
            epoch = state["next_epoch"]
            state["next_epoch"] += 1
            state["runs"][run_id] = {"status": "training", "epoch": epoch, "authority": authority}
            self.save(state)
            write_json(directory / "training.json", training)
            try:
                import torch

                from ..neurons.fit import fit_packets

                ranks = []
                shapes = None
                for rank in range(2):
                    packet_entry = training["packets"][rank]
                    path = Path(packet_entry["path"]).resolve()
                    if packet_entry["rank"] != rank or sha(path) != require_hash(packet_entry["sha256"]):
                        raise ValueError("Training packet rank or content differs")
                    packet = torch.load(path, weights_only=True, map_location="cpu")
                    if packet.get("rank") != rank or packet.get("source_pockets") != expected_sources or packet.get("base_sha256") != training["base_sha256"]:
                        raise ValueError("Training packet lacks matching native source provenance")
                    output = directory / f"rank{rank}.pt"
                    fit_packets(path, output, rank, width, steps, device="cpu", progress=sys.stderr)
                    current = validate_tensors(output, rank)
                    if shapes is not None and current != shapes:
                        raise ValueError("Paired training placement or width differs")
                    shapes = current
                    ranks.append({"rank": rank, "path": output.name, "sha256": sha(output)})
                parent = state["active"]
                manifest = publish_manifest(directory, {"schema_version": 1, "kind": "shared_expert_addition",
                    "model": "GLM-5.3-Flash-EXL3", "geometry": GEOMETRY, "epoch": epoch,
                    "base_sha256": training["base_sha256"], "source_pockets": sources, "ranks": ranks,
                    "width_per_rank": shapes, "parent": parent, "training_sha256": sha(directory / "training.json"),
                    "evaluation": None, "baseline": None, "suite": None, "evidence_kind": None})
                state["runs"][run_id]["status"] = "trained"
                self.save(state)
                return manifest
            except BaseException as error:
                state["runs"][run_id].update(status="failed", error=str(error))
                self.save(state)
                raise

    def qualify(self, run_id: Any, suite: Any, baseline: Any, report: Any, *, evidence_kind: Any, confirm: Any = False) -> Any:
        authority = self.authorize("qualify", confirm)
        with absorption_lock(self.home):
            self.require_idle()
            state = self.state()
            if state["runs"][run_id]["status"] != "trained":
                raise ValueError("Only a trained artifact may be qualified")
            if evidence_kind not in ("harness", "native"):
                raise ValueError("Evidence must explicitly name harness or native")
            directory = self.home / "artifacts" / safe_id(run_id)
            manifest = load_artifact(directory, qualified=False)
            proposed = copy.deepcopy(manifest)
            for name, path in (("suite", suite), ("baseline", baseline), ("evaluation", report)):
                source = Path(path)
                destination = directory / (name + ".json")
                shutil.copyfile(source, destination)
                proposed[name] = {"path": destination.name, "sha256": sha(destination)}
            proposed["evidence_kind"] = evidence_kind
            validate_evaluation(directory, proposed)
            # Later epochs must recheck every previously consolidated question.
            if manifest["parent"]:
                prior = load_artifact(self.home / "artifacts" / manifest["parent"])
                previous_suite = read_json(checked_file(self.home / "artifacts" / manifest["parent"], prior["suite"]))
                current_suite = read_json(directory / "suite.json")
                current_cases = {}
                for case in current_suite["cases"]:
                    current_cases[case["id"]] = case
                for case in previous_suite["cases"]:
                    if current_cases.get(case["id"]) != case:
                        raise ValueError("New epoch must preserve the previous knowledge/control suite")
            result = publish_manifest(directory, proposed)
            qualification = {"manifest_sha256": result["manifest_sha256"], "authority": authority}
            state["runs"][run_id].update(status="qualified", qualification=qualification,
                                         qualification_signature=signature(qualification, admin_key()))
            self.save(state)
            return result

    def check_driver(self, manifest: Any) -> Any:
        if self.driver is None:
            raise ValueError("Paired runtime driver is unavailable")
        settings = settings_for(self.root)
        expected = "native" if settings["mode"] == "spark" else "harness"
        if self.driver.evidence_kind != expected or manifest["evidence_kind"] != expected:
            raise ValueError("Harness evidence or driver cannot authorize native serving")
        if expected == "native":
            from .driver import CommandPair

            if not isinstance(self.driver, CommandPair):
                raise ValueError("Native serving requires the owner runner and bounded authority receipts")

    def verify_qualification(self, run: Any, manifest: Any) -> Any:
        value = run["qualification"]
        if value["manifest_sha256"] != manifest["manifest_sha256"] or value["authority"]["root"] != str(self.root):
            raise ValueError("Qualification belongs to a different artifact or root")
        if not hmac.compare_digest(run["qualification_signature"], signature(value, admin_key())):
            raise ValueError("Artifact qualification signature differs")

    def verify_source_slots(self, snapshot: Any, manifest: Any) -> Any:
        for source in manifest["source_pockets"]:
            for rank in range(2):
                installed = snapshot["ranks"][rank]["pockets"].get(source["id"])
                if installed is None or installed.get("sha256") != source["ranks"][rank]["sha256"]:
                    raise ValueError("Runtime source slot differs from the accepted teacher artifact")

    def apply_queue(self, originals: Any, replacements: Any, *, restore: Any = False) -> Any:
        if len(originals) != len(replacements):
            raise ValueError("Paired queue transition coverage differs")
        current = {}
        path = self.root / "queue/candidates.jsonl"
        for record in latest_records(path):
            current[record["id"]] = record
        for original, replacement in zip(originals, replacements):
            actual = current[original["id"]]
            if actual not in (original, replacement):
                raise ValueError("Source queue changed during paired lifecycle operation")
            desired = original if restore else replacement
            if actual != desired:
                append_record(path, desired)

    def change(self, state: Any, target: Any, after_state: Any, authority: Any, before: Any, recovery: Any = False, queue_before: Any = None, queue_after: Any = None) -> Any:
        if self.driver.snapshot() != before:
            raise ValueError("Paired runtime changed during preparation")
        journal = {"status": "pending", "before": before, "state_before": state,
                   "target": target, "state_after": after_state, "authority": authority,
                   "queue_before": queue_before or [], "queue_after": queue_after or []}
        journal["signature"] = signature(journal, admin_key())
        write_json(self.home / "operation.json", journal)
        try:
            self.driver.paired_reload(target, expected=before, recovery=recovery)
            if digest(self.driver.snapshot()) != digest(target):
                raise RuntimeError("Paired reload readback differs from exact target")
            self.apply_queue(journal["queue_before"], journal["queue_after"])
            self.save(after_state)
        except BaseException:
            self.recover_locked()
            raise
        journal["status"] = "complete"
        journal.pop("signature")
        journal["signature"] = signature(journal, admin_key())
        write_json(self.home / "operation.json", journal)

    def activate(self, run_id: Any, *, confirm: Any = False) -> Any:
        authority = self.authorize("activate", confirm)
        with absorption_lock(self.root), absorption_lock(self.home):
            self.require_idle()
            state = self.state()
            if state["runs"][run_id]["status"] != "qualified":
                raise ValueError("Activation requires a qualified artifact")
            directory = self.home / "artifacts" / safe_id(run_id)
            manifest = load_artifact(directory)
            self.verify_qualification(state["runs"][run_id], manifest)
            self.check_driver(manifest)
            if manifest["parent"] != state["active"]:
                raise ValueError("Active epoch changed after training; rebase the dream")
            before = self.driver.snapshot()
            self.verify_source_slots(before, manifest)
            target = self.driver.prepare(manifest, directory)
            self.verify_source_slots(target, manifest)
            for rank in range(2):
                current = target["ranks"][rank]
                if target["epoch"] != str(manifest["epoch"]) or current["neurons_sha256"] != manifest["ranks"][rank]["sha256"]:
                    raise ValueError("Prepared runtime does not install the qualified epoch")
                if current["active_memory_rows"] != 0 or current["pocket_cache_clean"] is not True:
                    raise ValueError("Activation requires clean pocket-OFF caches")
                for pocket in current["pockets"].values():
                    if pocket["enabled"] is not False:
                        raise ValueError("Activation requires every attention pocket OFF")
            after = copy.deepcopy(state)
            after["active"] = run_id
            candidates = {}
            for record in latest_records(self.root / "queue/candidates.jsonl"):
                candidates[record["id"]] = record
            rollback_queue = []
            for pocket in manifest["source_pockets"]:
                acceptance = read_json(pocket["acceptance_report"])
                for item in pocket["items"]:
                    if content_identity(candidates[item["id"]]) != item["content_sha256"]:
                        raise ValueError("Source candidate changed before activation")
                    self.authorize_source(candidates[item["id"]], pocket["id"], acceptance)
                    rollback_queue.append(candidates[item["id"]])
            rollback = {"run_id": run_id, "snapshot": before, "state": state, "authority": authority,
                        "queue": rollback_queue}
            after["runs"][run_id].update(status="active", rollback=rollback,
                                         rollback_signature=signature(rollback, admin_key()))
            self.change(state, target, after, authority, before)
            return self.state()

    def retire(self, run_id: Any, *, confirm: Any = False) -> Any:
        authority = self.authorize("retire", confirm)
        with absorption_lock(self.root), absorption_lock(self.home):
            self.require_idle()
            state = self.state()
            if state["active"] != run_id or state["runs"][run_id]["status"] != "active":
                raise ValueError("Retirement requires this active qualified artifact")
            directory = self.home / "artifacts" / safe_id(run_id)
            manifest = load_artifact(directory)
            self.verify_qualification(state["runs"][run_id], manifest)
            self.check_driver(manifest)
            snapshot = self.driver.snapshot()
            self.verify_source_slots(snapshot, manifest)
            for rank in range(2):
                current = snapshot["ranks"][rank]
                if current["neurons_sha256"] != manifest["ranks"][rank]["sha256"] or snapshot["epoch"] != str(manifest["epoch"]):
                    raise ValueError("Runtime no longer owns the evaluated artifact")
                if snapshot["base_sha256"] != manifest["base_sha256"] or current["active_memory_rows"] != 0 or current["pocket_cache_clean"] is not True:
                    raise ValueError("Retirement requires the evaluated base and clean pocket-OFF caches")
                for pocket in current["pockets"].values():
                    if pocket["enabled"] is not False:
                        raise ValueError("Retirement requires every attention pocket OFF")
            target = self.driver.prepare(manifest, directory, retire=True)
            for rank in range(2):
                for pocket in manifest["source_pockets"]:
                    if pocket["id"] in target["ranks"][rank]["pockets"]:
                        raise ValueError("Prepared retirement did not free the source slot")
            after = copy.deepcopy(state)
            candidates = {}
            for record in latest_records(self.root / "queue/candidates.jsonl"):
                candidates[record["id"]] = record
            queue_before = []
            queue_after = []
            for pocket in manifest["source_pockets"]:
                acceptance = read_json(pocket["acceptance_report"])
                for item in pocket["items"]:
                    record = candidates[item["id"]]
                    if content_identity(record) != item["content_sha256"]:
                        raise ValueError("Source candidate changed before retirement")
                    self.authorize_source(record, pocket["id"], acceptance)
                    queue_before.append(record)
                    learned = copy.deepcopy(record)
                    learned["status"] = "learned"
                    learned["learned"] = {"receipt": "dream:" + manifest["manifest_sha256"],
                                           "epoch": manifest["epoch"], "pocket": pocket["id"]}
                    queue_after.append(learned)
                after["pockets"][pocket["id"]]["status"] = "retired"
            after["runs"][run_id]["status"] = "retired"
            self.change(state, target, after, authority, snapshot, queue_before=queue_before, queue_after=queue_after)
            return self.state()

    def recover_locked(self) -> Any:
        journal = read_json(self.home / "operation.json")
        expected = journal.pop("signature")
        if not hmac.compare_digest(expected, signature(journal, admin_key())) or journal["authority"]["root"] != str(self.root):
            raise ValueError("Recovery receipt signature or scope differs")
        if journal["status"] != "pending":
            return self.state()
        self.driver.paired_reload(journal["before"], recovery=True)
        if digest(self.driver.snapshot()) != digest(journal["before"]):
            raise RuntimeError("Exact paired recovery readback failed; evidence preserved")
        self.apply_queue(journal["queue_before"], journal["queue_after"], restore=True)
        self.save(journal["state_before"])
        journal["status"] = "recovered"
        journal["signature"] = signature(journal, admin_key())
        write_json(self.home / "operation.json", journal)
        return self.state()

    def recover(self) -> Any:
        if self.driver is None:
            raise ValueError("Paired runtime driver is unavailable for recovery")
        with absorption_lock(self.root), absorption_lock(self.home):
            return self.recover_locked()

    def rollback(self, *, confirm: Any = False) -> Any:
        if self.driver is None:
            raise ValueError("Paired runtime driver is unavailable for rollback")
        with absorption_lock(self.root), absorption_lock(self.home):
            self.require_idle()
            state = self.state()
            run_id = state["active"]
            if not run_id:
                raise ValueError("No active dream epoch to roll back")
            run = state["runs"][run_id]
            rollback = run["rollback"]
            if not hmac.compare_digest(run["rollback_signature"], signature(rollback, admin_key())):
                raise ValueError("Saved rollback receipt signature differs")
            authority = rollback["authority"]
            if authority["root"] != str(self.root) or rollback["run_id"] != run_id:
                raise ValueError("Saved rollback scope differs")
            if authority["autonomy"] in ("manual", "supervised") and not confirm:
                raise ValueError("Saved manual rollback needs explicit confirmation")
            candidates = {}
            for record in latest_records(self.root / "queue/candidates.jsonl"):
                candidates[record["id"]] = record
            queue_before = []
            for record in rollback["queue"]:
                actual = candidates[record["id"]]
                if actual != record:
                    if content_identity(actual) != content_identity(record) or actual.get("learned", {}).get("receipt") != "dream:" + run["qualification"]["manifest_sha256"]:
                        raise ValueError("Source candidate changed after this dream; preserve the newer decision")
                queue_before.append(actual)
            # The original activation authorizes exact restoration after grant expiry.
            # Exact signed restoration uses the driver's preserved recovery lane.
            self.change(state, rollback["snapshot"], rollback["state"], authority, self.driver.snapshot(), recovery=True,
                        queue_before=queue_before, queue_after=rollback["queue"])
            return self.state()
