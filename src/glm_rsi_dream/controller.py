"""Recoverable two-Spark capture/check/serve windows, run on the head Spark."""
import base64
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

from .files import digest, read_json, sha, write_json
from .guard import body_reports_outage
from .manifest import MANIFEST
from .safety import MemoryGate, SafetyError, require_parity
from .transport import Transport


class Cluster:
    def __init__(self, config, run_dir, transport=None):
        self.config = config
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        if not self.run_id.replace("-", "").replace("_", "").isalnum() or len(self.run_id) > 64:
            raise ValueError("run directory basename must be a safe ID, at most 64 characters")
        self.transport = transport or Transport(config)
        self.state_path = self.run_dir / "state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {"originals": [], "candidates": [], "phase": "new"}
        self.state.setdefault("candidate_history", [])
        self.state.setdefault("trainer_history", [])
        self.gate = MemoryGate(config["safety"])
        for rank, minimum in self.state.get("memory_minima", {}).items():
            self.gate.minima[int(rank)] = minimum
        self.stop = threading.Event()
        self.failure = None
        self.watcher = None
        self.changed = False
        self.restore_lock = threading.RLock()
        self.probe_lock = threading.RLock()
        self.save_lock = threading.RLock()
        self.counter = 0

    def save(self):
        with self.save_lock:
            self.state["memory_minima"] = self.gate.minima
            write_json(self.state_path, self.state)

    def health(self):
        url = self.config["api_url"] + "/health"
        try:
            opened = urllib.request.urlopen(url, timeout=5)
        except urllib.error.HTTPError as error:
            raw = error.read(65536)
            if body_reports_outage(raw):
                raise SafetyError("head health reports rank desynchronization")
            raise SafetyError("head health failed")
        with opened as response:
            raw = response.read(65536)
            if response.status != 200 or not raw:
                raise SafetyError("head health failed")
            if body_reports_outage(raw):
                raise SafetyError("head health reports rank desynchronization")

    def probe(self, serving=True):
        # Sampling and comparison have one owner even with a watcher and caller.
        with self.probe_lock:
            return self._probe(serving)

    def _probe(self, serving=True):
        if (self.run_dir / "STOP").exists() and self.state["phase"] not in ("restored", "restoring"):
            raise InterruptedError("STOP requested")
        for rank in range(2):
            self.gate.observe(rank, self.transport.call(rank, "metrics"))
        if serving:
            self.health()

    def background_probe(self):
        try:
            while not self.stop.wait(self.config["safety"]["probe_seconds"]):
                self.probe()
                monitored = self.state["candidates"] if self.state["phase"] == "serving" else self.state["originals"]
                for rank, candidate in enumerate(monitored):
                    actual = self.transport.call(rank, "inspect", container=candidate["id"])
                    if not actual["State"]["Running"]:
                        raise SafetyError("candidate rank exited")
        except BaseException as error:
            self.failure = error
            self.state['window_abort'] = {'type':type(error).__name__,'reason':str(error)}
            self.save()
            # Restore while a request may still be blocked, rather than waiting for it.
            try:
                self.restore()
            except BaseException as restore_error:
                self.state["restore_error"] = str(restore_error)
                self.save()

    def ensure_safe(self):
        if self.failure is not None:
            raise SafetyError("background serving gate failed: " + str(self.failure))
        self.probe()

    def body(self, prompt, tagged=False, arm="none", teacher=None, completion=False):
        if teacher:
            prompt = "Reference facts stated by the user:\n" + teacher + "\n\nQuestion: " + prompt
        body = {"model": self.config["model"], "temperature": 0, "max_tokens": self.config["max_tokens"],
                "stream": False, "draft": False}
        if completion:
            body["prompt"] = prompt
            body["max_tokens"] = 1
        else:
            body["messages"] = []
            if self.config.get("system_prompt"):
                body["messages"].append({"role": "system", "content": self.config["system_prompt"]})
            body["messages"].append({"role": "user", "content": prompt})
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if tagged:
            request_id = "p-" + uuid.uuid4().hex
            pair = read_json(self.run_dir / "pair/pair.json")
            body["metadata"] = {"dream_eval": {"tag": "glm-eval-v1", "request_id": request_id,
                                              "arm": arm, "pair_sha256": pair["pair_sha256"]}}
        return body

    def infer(self, body, policy=None):
        self.ensure_safe()
        tagged = body.get("metadata", {}).get("dream_eval")
        if policy is not None:
            for rank in range(2):
                rank_policy = dict(policy)
                selected = rank_policy.pop("selected_by_rank", None)
                if selected is not None:
                    rank_policy["selected"] = selected[rank]
                self.transport.write(rank, self.run_id, "pair/requests/" + tagged["request_id"] + ".json",
                                     json.dumps(rank_policy).encode())
        self.counter += 1
        name = tagged["request_id"] if tagged else "untagged-" + str(self.counter) + "-" + uuid.uuid4().hex[:8]
        # An attempted POST is never silently regenerated or retried.
        write_json(self.run_dir / "responses" / (name + "-request.json"), body)
        endpoint = "/v1/completions" if "prompt" in body else "/v1/chat/completions"
        request = urllib.request.Request(self.config["api_url"] + endpoint, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise SafetyError("reply exceeds evidence bound")
        reply = json.loads(raw)
        write_json(self.run_dir / "responses" / (name + "-raw.json"), reply)
        self.ensure_safe()
        receipts = []
        if tagged:
            for rank in range(2):
                path = "receipts/" + name + "-rank" + str(rank) + ".json"
                receipt = json.loads(self.transport.read(rank, self.run_id, path))
                if any(receipt.get(key) != value for key, value in tagged.items()):
                    raise SafetyError("rank receipt selection mismatch")
                if receipt.get("tp_rank") != rank or receipt.get("tp_world") != 2 or receipt.get("hook_failed") or not receipt.get("completed"):
                    raise SafetyError("rank receipt is incomplete or failed")
                receipts.append(receipt)
            write_json(self.run_dir / "responses" / (name + "-receipts.json"), receipts)
        return reply, receipts

    def preflight(self):
        self.probe()
        if self.state["phase"] in ("serving", "cutover", "restoring", "restore_failed"):
            raise SafetyError("an existing serving window requires restore before a new window")
        pins = MANIFEST["engine"]
        originals = []
        for rank, host in enumerate(self.config["hosts"]):
            actual = self.transport.call(rank, "inspect", container=host["original_container"])
            if not actual["State"]["Running"]:
                raise SafetyError("original rank is not running")
            if actual["Image"] != self.config["runtime"]["image_id"]:
                raise SafetyError("original runtime image differs from the pinned baseline")
            command = actual["Config"].get("Cmd") or []
            if "--tp" not in command or command[command.index("--tp") + 1] != "2":
                raise SafetyError("original command does not prove TP=2")
            if "--rank" not in command or command[command.index("--rank") + 1] != str(rank):
                raise SafetyError("original rank command mismatch")
            if "--parallel" not in command or command[command.index("--parallel") + 1] != "4":
                raise SafetyError("pinned native overlay requires parallel=4")
            identity = {"Config": actual["Config"], "HostConfig": actual["HostConfig"], "Image": actual["Image"]}
            previous = self.state["originals"]
            if previous and (previous[rank]["id"] != actual["Id"] or previous[rank]["identity"] != identity):
                raise SafetyError("original identity changed since capture")
            model_dir = command[command.index("serve") + 1]
            code = "import hashlib,json,importlib.util,site,base64,torch; from pathlib import Path\n"
            code += "assert torch.cuda.is_available() and 'GB10' in torch.cuda.get_device_name(0), 'DGX Spark GB10 required'\n"
            code += "root=Path(importlib.util.find_spec('tensorfold').submodule_search_locations[0])\n"
            code += "pins=" + repr(pins["source_files"]) + "\n"
            code += "for name,expected in pins.items():\n assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name\n"
            code += "model=Path(" + repr(model_dir) + ")\n"
            code += "assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()==" + repr(MANIFEST["model_config_sha256"]) + "\n"
            code += "print(json.dumps({'site_dir':site.getsitepackages()[0],'tokenizer':base64.b64encode((model/'tokenizer.json').read_bytes()).decode()}))\n"
            info = json.loads(self.transport.call(rank, "exec", container=actual["Id"], code=code))
            tokenizer = base64.b64decode(info["tokenizer"])
            local = self.run_dir / "tokenizer.json"
            if local.exists() and local.read_bytes() != tokenizer:
                raise SafetyError("tokenizer changed or differs between ranks")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(tokenizer)
            originals.append({"id": actual["Id"], "identity": identity, "site_dir": info["site_dir"]})
        self.state["originals"] = originals
        self.state["phase"] = "prepared"
        self.save()
        return pins

    def manifest(self):
        pair_dir = self.run_dir / "pair"
        binding = {"mechanism": MANIFEST["memory"]["mechanism"], "slots": MANIFEST["memory"]["slots"],
                   "tp_world": 2, "original_identities": [],
                   "tokenizer_sha256": sha(self.run_dir / "tokenizer.json"),
                   "prefix_sha256": sha(pair_dir / "prefix.json")}
        for original in self.state["originals"]:
            binding["original_identities"].append(digest(original["identity"]))
        pair = {"schema_version": 3, "config_sha256": MANIFEST["model_config_sha256"],
                "binding": binding, "pair_sha256": digest(binding)}
        write_json(pair_dir / "pair.json", pair)

    def ready(self):
        deadline = time.monotonic() + self.config["safety"]["ready_timeout_seconds"]
        while time.monotonic() < deadline:
            self.probe(serving=False)
            for rank, candidate in enumerate(self.state["candidates"]):
                actual = self.transport.call(rank, "inspect", container=candidate["id"])
                if not actual["State"]["Running"]:
                    raise SafetyError("serving rank exited during readiness")
            try:
                self.health()
                return
            except (OSError, SafetyError):
                time.sleep(1)
        raise SafetyError("paired serving readiness timed out")

    def restore(self):
        with self.restore_lock:
            return self._restore()

    def _restore(self):
        if len(self.state["originals"]) != 2:
            raise SafetyError("restoration requires two preserved original identities")
        self.stop.set()
        self.state["phase"] = "restoring"
        self.save()
        errors = []
        stopped = set()
        owned = self.state["candidate_history"] + self.state["trainer_history"]
        for candidate in owned:
            if candidate["id"] in stopped:
                continue
            stopped.add(candidate["id"])
            rank = candidate["rank"]
            try:
                self.transport.call(rank, "stop", container=candidate["id"])
            except Exception as error:
                errors.append(str(error))
        for rank in (1, 0):
            try:
                self.transport.call(rank, "start", container=self.state["originals"][rank]["id"])
            except Exception as error:
                errors.append(str(error))
        deadline = time.monotonic() + self.config["safety"]["ready_timeout_seconds"]
        while time.monotonic() < deadline:
            try:
                self.health()
                break
            except (OSError, SafetyError):
                time.sleep(1)
        else:
            errors.append("original head readiness failed after restore")
        for rank, original in enumerate(self.state["originals"]):
            try:
                actual = self.transport.call(rank, "inspect", container=original["id"])
                identity = {"Config": actual["Config"], "HostConfig": actual["HostConfig"], "Image": actual["Image"]}
                if not actual["State"]["Running"] or identity != original["identity"]:
                    errors.append("original exact identity restoration failed")
                if rank < len(self.state["candidates"]):
                    actual = self.transport.call(rank, "inspect", container=self.state["candidates"][rank]["id"])
                    if actual["State"]["Running"]:
                        errors.append("candidate remained running")
            except Exception as error:
                errors.append(str(error))
        for candidate in owned:
            try:
                actual = self.transport.call(candidate["rank"], "inspect", container=candidate["id"])
                if actual["State"]["Running"]:
                    errors.append("recorded prior candidate remained running")
            except Exception as error:
                errors.append(str(error))
        self.state["phase"] = "restore_failed" if errors else "restored"
        self.state["restore_errors"] = errors
        self.save()
        self.changed = False
        if errors:
            raise SafetyError("planned restore verification failed: " + "; ".join(errors))

    @contextmanager
    def window(self, protected):
        pins = self.preflight()
        self.manifest()
        before = []
        for prompt in protected:
            reply, _ = self.infer(self.body(prompt))
            before.append(reply)
        write_json(self.run_dir / "untagged-baseline.json", before)
        session = "w-" + uuid.uuid4().hex[:16]
        self.state["candidates"] = []
        self.changed = True
        try:
            for rank in range(2):
                self.transport.stage(rank, self.run_id, self.run_dir)
                original = self.state["originals"][rank]
                candidate = self.transport.call(rank, "clone", original=original["id"], identity=original["identity"],
                                                  rank=rank, run_id=session, root=self.transport.root(rank, self.run_id),
                                                  site_dir=original["site_dir"], engine_sha256=pins["snapshot_sha256"])
                candidate["rank"] = rank
                self.state["candidates"].append(candidate)
                self.state["candidate_history"].append(candidate)
                self.save()
                if not candidate.get("validated"):
                    raise SafetyError("candidate launch settings failed exact readback")
            self.state["phase"] = "cutover"
            self.save()
            if hasattr(self, "arm_guards"):
                self.arm_guards()
            for rank in range(2):
                self.transport.call(rank, "stop", container=self.state["originals"][rank]["id"])
            for rank in (1, 0):
                self.transport.call(rank, "start", container=self.state["candidates"][rank]["id"])
            self.ready()
            self.state["phase"] = "serving"
            self.save()
            self.stop.clear()
            self.watcher = threading.Thread(target=self.background_probe, daemon=True)
            self.watcher.start()
            after = []
            for prompt in protected:
                reply, _ = self.infer(self.body(prompt))
                after.append(reply)
            require_parity(before, after)
            yield before
        finally:
            self.stop.set()
            if self.watcher is not None and threading.current_thread() != self.watcher:
                self.watcher.join(timeout=190)
            if self.changed:
                self.restore()
            after = []
            for prompt in protected:
                # Clear an older failure so this probe can run. Keep a failure the probe itself records.
                saved = self.failure
                self.failure = None
                try:
                    reply, _ = self.infer(self.body(prompt))
                    after.append(reply)
                finally:
                    if self.failure is None:
                        self.failure = saved
            require_parity(before, after)
            write_json(self.run_dir / "restore-parity.json", {"byte_identical": len(after)})
