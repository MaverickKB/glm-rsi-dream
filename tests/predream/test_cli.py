import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from glm_rsi_dream.predream.cli import main


def test_module_help_needs_no_storage_or_network(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "glm_rsi_dream.predream", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    for command in ("extract", "queue", "grade", "absorb", "stage", "dream", "timings"):
        assert command in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_timing_summary_of_missing_run_is_read_only(tmp_path, capsys):
    root = tmp_path / "missing"
    assert main(["--root", str(root), "timings"]) == 0
    assert capsys.readouterr().out.strip() == "{}"
    assert not root.exists()


@pytest.fixture
def glm_protocol_server():
    replies = []
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(size))
            requests.append({"path": self.path, "payload": payload})
            answer = replies.pop(0)
            response = {"choices": [{"finish_reason": "stop", "message": {"content": answer}}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = "http://127.0.0.1:" + str(server.server_port)
    try:
        yield endpoint, replies, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cli_pipeline_routes_replays_and_preserves_blind_probe(tmp_path, capsys, glm_protocol_server):
    from glm_rsi_dream.predream.queue import latest_records

    endpoint, replies, requests = glm_protocol_server
    texts = [
        "In this fictional protocol, cobalt frames use 17 washers.",
        "Mara has an appointment at 09:20 tomorrow.",
        "The fictional dog Tavi wears a purple collar.",
        "Water is H2O.",
    ]
    questions = [
        "How many washers do cobalt frames use in this fictional protocol?",
        "When is Mara's next appointment?",
        "What color is the fictional dog Tavi's collar?",
        "What is the chemical formula for water?",
    ]
    candidates = []
    for index, text in enumerate(texts):
        candidates.append({"text": text, "prompt_evidence": text, "probe_question": questions[index]})
    replies.append(json.dumps({"candidates": candidates}))
    for index in range(4):
        if index == 3:
            replies.append("The chemical formula for water is H2O.")
            replies.append(json.dumps({"agrees": True, "novelty": "known"}))
        else:
            replies.append("I do not know.")
            replies.append(json.dumps({"agrees": None, "novelty": "new"}))
    routes = [("weights", None), ("memory", None), ("pocket", "dogs"), ("weights", None)]
    for index, (route, pocket) in enumerate(routes):
        verdict = "reject" if index == 3 else "accept"
        replies.append(json.dumps({"verdict": verdict, "route": route, "pocket": pocket,
                                   "reason": "Protocol fixture decision", "score": 0.9}))
    source = tmp_path / "source.txt"
    source.write_text("\n".join(texts))
    root = tmp_path / "run"
    common = ["--root", str(root), "--endpoint", endpoint]

    def run(*arguments):
        code = main(common + list(arguments))
        output = capsys.readouterr()
        assert code == 0, output.err
        return output.out

    assert len(json.loads(run("extract", str(source)))) == 4
    table = run("queue", "list", "--status", "pending")
    assert table.count("pending") == 4
    graded = json.loads(run("grade", "--all-pending"))
    assert len(graded["graded"]) == 4 and not graded["errors"]
    absorbed = json.loads(run("absorb"))
    assert len(absorbed["absorbed"]) == 1
    manifest = json.loads(run("stage", "--run-id", "sealed"))
    assert len(manifest["items"]) == 2
    queue_path = root / "queue" / "candidates.jsonl"
    rows = latest_records(queue_path)
    assert [row["status"] for row in rows] == ["staged", "absorbed", "staged", "rejected"]
    before = queue_path.read_bytes()
    assert json.loads(run("grade", "--all-pending")) == {"graded": [], "errors": []}
    assert json.loads(run("absorb"))["absorbed"] == []
    assert json.loads(run("stage", "--run-id", "sealed")) == manifest
    assert queue_path.read_bytes() == before
    assert not replies
    assert len(requests) == 13
    for request in requests:
        assert request["path"] == "/v1/chat/completions"
    for index, request_index in enumerate((1, 3, 5, 7)):
        messages = requests[request_index]["payload"]["messages"]
        assert messages[-1]["content"] == questions[index]
        assert texts[index] not in json.dumps(messages)
    timing_summary = json.loads(run("timings"))
    for stage in ("extract", "probe", "grade", "absorb", "stage", "queue_list"):
        assert timing_summary[stage]["count"] > 0
    assert "dream_preview" not in timing_summary and "dream" not in timing_summary


def test_partial_grade_failure_is_an_unsuccessful_cli_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("glm_rsi_dream.predream.cli.dispatch", lambda args: {"graded": [], "errors": ["unavailable"]})
    assert main(["--root", str(tmp_path), "grade"]) == 1
    assert json.loads(capsys.readouterr().out)["errors"] == ["unavailable"]


def test_dream_confirmation_requires_shared_expert_inputs(tmp_path, monkeypatch, capsys):
    def forbidden(*args):
        raise AssertionError("A dream must never launch the attention trainer")

    monkeypatch.setattr("glm_rsi_dream.cli.main", forbidden)
    assert main(["--enable-dream-features", "--root", str(tmp_path), "dream", "--manual", "--confirm", "--run-id", "epoch1"]) == 1
    assert "shared-expert consolidation" in capsys.readouterr().err
