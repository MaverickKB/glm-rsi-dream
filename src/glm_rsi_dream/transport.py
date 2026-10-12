"""Head-Spark SSH transport. Constructing it is separate from dry-run planning."""
import base64
import io
import json
import shlex
import subprocess
import tarfile
import threading
from pathlib import Path


class Transport:
    def __init__(self, config):
        self.config = config
        self.agent_code = Path(__file__).with_name("agent.py").read_text()
        self.locks = [threading.RLock(),threading.RLock()]

    def call(self, host_rank, action, **fields):
        with self.locks[host_rank]:
            return self._call(host_rank,action,**fields)

    def _call(self, host_rank, action, **fields):
        rank = host_rank
        host = self.config["hosts"][rank]
        command = list(self.config.get("remote_python", ["python3"]))
        command.extend(["-B", "-c", self.agent_code])
        ssh = list(self.config.get("ssh_command", ["ssh", "-oBatchMode=yes", "-oConnectTimeout=10"]))
        ssh.extend([host["ssh"], shlex.join(command)])
        spec = {"action": action}
        spec.update(fields)
        timeout = 20 if action in ['metrics','inspect','start','stop','write'] else 180
        try:
            result = subprocess.run(ssh, input=json.dumps(spec).encode(), capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Spark rank '+str(rank)+' '+action+' exceeded '+str(timeout)+' seconds') from None
        if result.returncode:
            raise RuntimeError("Spark rank " + str(rank) + " operation " + action + " failed: " + result.stderr.decode()[-500:])
        return json.loads(result.stdout)

    def root(self, rank, run_id):
        return self.config["hosts"][rank]["work_dir"] + "/" + run_id

    def write(self, rank, run_id, relative, raw):
        return self.call(rank, "write", root=self.root(rank, run_id), path=relative,
                         data=base64.b64encode(raw).decode())

    def read(self, rank, run_id, relative):
        record = self.call(rank, "read", root=self.root(rank, run_id), path=relative)
        return base64.b64decode(record["data"], validate=True)

    def stage(self, rank, run_id, run_dir):
        buffer = io.BytesIO()
        source_root = Path(__file__).resolve().parents[1]
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for package in ("glm_rsi_dream", "tensorfold_dream"):
                for path in sorted((source_root / package).rglob("*")):
                    if path.is_file() and path.suffix in (".py", ".json"):
                        archive.add(path, arcname="src/" + str(path.relative_to(source_root)))
            for path in sorted((run_dir / "pair").rglob("*")):
                if path.is_file() and "requests" not in path.parts:
                    archive.add(path, arcname="pair/" + str(path.relative_to(run_dir / "pair")))
            pth = b"import sys; sys.path.insert(0, '/opt/pocket/src'); from tensorfold_dream.bootstrap import install; install()\n"
            info = tarfile.TarInfo("bootstrap.pth")
            info.size = len(pth)
            archive.addfile(info, io.BytesIO(pth))
        data = buffer.getvalue()
        return self.call(rank, "stage", root=self.root(rank, run_id), archive=base64.b64encode(data).decode())
