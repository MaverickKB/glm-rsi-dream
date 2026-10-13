"""Spark-local, stdlib-only Docker/file agent invoked over SSH by the head.

It uses Docker's Unix socket. Original containers are never reconfigured.
"""

import base64
import copy
import hashlib
import http.client
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.parse
from pathlib import Path
from typing import Any


class DockerConnection(http.client.HTTPConnection):
    def connect(self) -> Any:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(15)
        self.sock.connect("/var/run/docker.sock")


def docker(method: Any, path: Any, body: Any = None) -> Any:
    connection = DockerConnection("localhost", timeout=15)
    data = None if body is None else json.dumps(body)
    connection.request(method, path, data, {"Content-Type": "application/json"})
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    if response.status == 304 and method == "POST" and ("/start" in path or "/stop" in path):
        return None
    if response.status >= 300:
        raise RuntimeError("Docker operation failed: " + str(response.status) + " " + raw.decode()[:300])
    if not raw:
        return None
    return json.loads(raw)


def inspect(container: Any) -> Any:
    return docker("GET", "/containers/" + urllib.parse.quote(container, safe="") + "/json")


def identity(record: Any) -> Any:
    return {"Config": record["Config"], "HostConfig": record["HostConfig"], "Image": record["Image"]}


def metrics() -> Any:
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        memory[fields[0].rstrip(":")] = int(fields[1]) * 1024
    vm = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        name, value = line.split()
        vm[name] = int(value)
    import os
    return {"time": time.monotonic(), "available": memory["MemAvailable"],
            "swap_out_bytes": vm["pswpout"] * os.sysconf("SC_PAGE_SIZE")}


def safe_path(root: Any, relative: Any) -> Any:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("file escaped the scoped run directory")
    return path


def stage(root: Any, encoded: Any) -> Any:
    data = base64.b64decode(encoded, validate=True)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError("only regular staged files are allowed")
            path = safe_path(root, member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError("only regular staged files are allowed")
            content = extracted.read()
            path.write_bytes(content)
            path.chmod(0o600)
    for name in ("capture", "receipts", "pair/requests"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return {"staged_sha256": hashlib.sha256(data).hexdigest()}


def clone(spec: Any) -> Any:
    original = inspect(spec["original"])
    if identity(original) != spec["identity"]:
        raise ValueError("original launch identity changed")
    config = copy.deepcopy(original["Config"])
    host = copy.deepcopy(original["HostConfig"])
    environment = list(config.get("Env") or [])
    for entry in environment:
        name = entry.split("=", 1)[0]
        if name.startswith("TF_DREAM_") or name in ("LW_ADAPTER", "LW_CAPTURE_DIR"):
            raise ValueError("original already has a conflicting intervention")
    environment.extend(["TF_DREAM_CAPTURE=1", "TF_DREAM_CAPTURE_DIR=/capture",
                        "TF_DREAM_CAPTURE_MAX_BYTES=67108864", "TF_DREAM_CAPTURE_COPY_ROWS=32",
                        "TF_DREAM_CAPTURE_ENGINE_SHA256=" + spec["engine_sha256"],
                        "TF_DREAM_EVAL=1", "TF_DREAM_EVAL_MANIFEST=/opt/pocket/pair/pair.json"])
    # Attention capture requires unsplit prompt rows in isolated clones.
    # Original launch settings and the saved original identity stay intact.
    environment = [entry for entry in environment if not entry.startswith(("TF_GLM_HC_SPLIT=","TF_GLM_PREFILL_OVERLAP="))]
    environment.extend(["TF_GLM_HC_SPLIT=0","TF_GLM_PREFILL_OVERLAP=0"])
    config["Env"] = environment
    config["Image"] = original["Image"]
    config["Labels"] = dict(config.get("Labels") or {})
    config["Labels"]["glm-rsi-dream.run"] = spec["run_id"]
    config["Labels"]["glm-rsi-dream.rank"] = str(spec["rank"])
    binds = list(host.get("Binds") or [])
    mounts = {"/opt/pocket", "/capture", "/z-receipts", spec["site_dir"] + "/glm-rsi-dream.pth"}
    for entry in binds:
        if entry.split(":")[1] in mounts:
            raise ValueError("candidate mount conflicts with original")
    root = spec["root"]
    binds.extend([root + ":/opt/pocket:ro", root + "/capture:/capture:rw",
                  root + "/receipts:/z-receipts:rw",
                  root + "/bootstrap.pth:" + spec["site_dir"] + "/glm-rsi-dream.pth:ro"])
    host["Binds"] = binds
    # A restart policy could revive a failed candidate after planned restore.
    host["RestartPolicy"] = {"Name": "no", "MaximumRetryCount": 0}
    config["HostConfig"] = host
    name = "glm-rsi-dream-" + spec["run_id"] + "-rank" + str(spec["rank"])
    result = docker("POST", "/containers/create?name=" + name, config)
    actual = inspect(result["Id"])
    if actual["Image"] != original["Image"]:
        raise ValueError("candidate image readback mismatch")
    expected_config = copy.deepcopy(config)
    expected_config.pop("HostConfig")
    expected_host = copy.deepcopy(host)
    actual_host = copy.deepcopy(actual["HostConfig"])
    # Docker normalizes the nullable boolean on container creation.
    for launch in (expected_host, actual_host):
        if launch.get("OomKillDisable") is None:
            launch["OomKillDisable"] = False
    validated = actual["Config"] == expected_config and actual_host == expected_host
    return {"id": result["Id"], "identity": identity(actual), "validated": validated}


def execute_python(container: Any, code: Any) -> Any:
    record = docker("POST", "/containers/" + container + "/exec",
                    {"AttachStdout": True, "AttachStderr": True,
                     "Cmd": ["python3", "-B", "-c", code]})
    connection = DockerConnection("localhost", timeout=15)
    connection.request("POST", "/exec/" + record["Id"] + "/start",
                       json.dumps({"Detach": False, "Tty": False}), {"Content-Type": "application/json"})
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    if response.status >= 300:
        raise RuntimeError("Docker exec failed")
    chunks = []
    offset = 0
    while offset < len(raw):
        size = int.from_bytes(raw[offset + 4:offset + 8], "big")
        chunks.append(raw[offset + 8:offset + 8 + size])
        offset += 8 + size
    state = docker("GET", "/exec/" + record["Id"] + "/json")
    text = b"".join(chunks).decode()
    if state["ExitCode"] != 0:
        raise RuntimeError("container preflight failed: " + text[:500])
    return text


def expand_root(path: Any) -> Any:
    expanded = os.path.expanduser(path)
    if not expanded.startswith("/"):
        raise ValueError("run directory must be absolute")
    return expanded


def dispatch(spec: Any) -> Any:
    if isinstance(spec.get("root"), str):
        spec["root"] = expand_root(spec["root"])
    action = spec["action"]
    if action == "no_listener":
        port = int(spec["port"])
        if not 1 <= port <= 65535:
            raise ValueError("listener port is invalid")
        for name in ("tcp", "tcp6"):
            path = Path("/proc/net") / name
            for line in path.read_text().splitlines()[1:]:
                fields = line.split()
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
                if local_port == port and fields[3] == "0A":
                    raise RuntimeError("Worker generation listener detected")
        return {"absent": True, "port": port}
    if action == "host_probe":
        result = subprocess.run(["python3", "-B", "-c", spec["code"]], capture_output=True, timeout=15)
        if result.returncode:
            raise RuntimeError("Protected host probe failed")
        return {"passed": True}
    if action == "finish_guard":
        root = Path(spec["root"])
        (root / "guard-done").write_text("done\n")
        pid_path = root / "guard.pid"
        if not pid_path.exists():
            return {"stopped": True}
        pid = int(pid_path.read_text())
        command_path = Path("/proc") / str(pid) / "cmdline"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and command_path.exists():
            try:
                command = command_path.read_bytes()
            except FileNotFoundError:
                break
            if not command:
                break
            if str(root / "src/glm_rsi_dream/guard.py").encode() not in command:
                raise RuntimeError("guard PID identity changed")
            time.sleep(0.2)
        else:
            if command_path.exists():
                import signal
                os.kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        remaining = command_path.read_bytes()
                    except FileNotFoundError:
                        break
                    if not remaining:
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Scoped guard did not stop")
        return {"stopped": True}
    if action == "arm_guard":
        root = Path(spec["root"])
        old_pid = root / "guard.pid"
        if old_pid.exists():
            command_path = Path("/proc") / old_pid.read_text().strip() / "cmdline"
            if command_path.exists() and command_path.read_bytes():
                raise RuntimeError("Previous scoped guard remains active")
        (root / "guard-done").unlink(missing_ok=True)
        value = spec["spec"]
        value["ready_after"] = time.time() + value["safety"]["ready_timeout_seconds"]
        path = root / "guard.json"
        path.write_text(json.dumps(value))
        path.chmod(0o600)
        log = (root / "guard.log").open("ab")
        process = subprocess.Popen(["python3", "-B", str(root / "src/glm_rsi_dream/guard.py"), str(path)],
                                   stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()
        (root / "guard.pid").write_text(str(process.pid))
        return {"pid": process.pid}
    if action == "metrics":
        return metrics()
    if action == "inspect":
        return inspect(spec["container"])
    if action == "stage":
        return stage(spec["root"], spec["archive"])
    if action == "clone":
        return clone(spec)
    if action in ("start", "stop"):
        suffix = "/" + action
        if action == "stop":
            suffix += "?t=5"
        return docker("POST", "/containers/" + spec["container"] + suffix)
    if action == "read":
        raw = safe_path(spec["root"], spec["path"]).read_bytes()
        return {"data": base64.b64encode(raw).decode(), "sha256": hashlib.sha256(raw).hexdigest()}
    if action == "write":
        path = safe_path(spec["root"], spec["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(spec["data"], validate=True))
        path.chmod(0o600)
        return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if action == "exec":
        return execute_python(spec["container"], spec["code"])
    raise ValueError("unknown agent operation")


if __name__ == "__main__":
    print(json.dumps(dispatch(json.load(sys.stdin))))
