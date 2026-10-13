"""Extraction boundaries: literal evidence, blind prior calls and replay safety."""

import io
import json

import pytest

from glm_rsi_dream.predream.extract import GLMClient, extract_candidates, extract_file
from glm_rsi_dream.predream.queue import latest_records


class ScriptedGLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def extraction(text="Lumen's inspection code is QUARTZ-418."):
    return json.dumps({"candidates": [{
        "text": text, "prompt_evidence": text,
        "probe_question": "What is Lumen's inspection code?",
    }]})


def run_extract(tmp_path, client, source="Lumen's inspection code is QUARTZ-418."):
    return extract_candidates(source, "fixture:lumen", client=client,
                              queue_path=tmp_path / "queue" / "candidates.jsonl",
                              timings_path=tmp_path / "timings.jsonl")


@pytest.mark.parametrize("novelty,agrees,prior", [
    ("new", None, "I do not know."),
    ("known", True, "The code is QUARTZ-418."),
    ("conflicts", False, "The code is AMBER-92."),
])
def test_prior_is_fresh_and_answer_free(tmp_path, novelty, agrees, prior):
    client = ScriptedGLM([extraction(), prior, json.dumps({"novelty": novelty, "agrees": agrees})])
    records = run_extract(tmp_path, client)
    assert len(records) == 1
    assert records[0]["novelty"] == novelty
    assert records[0]["glm_prior"] == {"answer": prior, "agrees": agrees}
    prior_messages = client.calls[1]
    assert len(prior_messages) == 2
    assert prior_messages[1]["content"] == "What is Lumen's inspection code?"
    assert "QUARTZ-418" not in json.dumps(prior_messages)
    assert "fixture:lumen" not in json.dumps(prior_messages)
    assert latest_records(tmp_path / "queue" / "candidates.jsonl") == records
    timings = []
    for line in (tmp_path / "timings.jsonl").read_text().splitlines():
        timings.append(json.loads(line))
    assert [item["stage"] for item in timings] == ["extract", "probe"]
    assert records[0]["timings_ms"]["probe"] == timings[1]["wall_ms"]


def test_invalid_evidence_repairs_then_stops_after_three(tmp_path):
    invalid = extraction("A fact absent from the source.")
    client = ScriptedGLM([invalid, invalid, invalid])
    with pytest.raises(ValueError, match="three attempts"):
        run_extract(tmp_path, client)
    assert len(client.calls) == 3
    assert not (tmp_path / "queue" / "candidates.jsonl").exists()
    failure = json.loads((tmp_path / "timings.jsonl").read_text())
    assert "error" in failure


def test_json_fence_and_format_repair(tmp_path):
    good = "```json\n" + extraction() + "\n```"
    client = ScriptedGLM(["not json", good, "Unknown.", '{"novelty":"new","agrees":null}'])
    assert run_extract(tmp_path, client)[0]["novelty"] == "new"
    assert "Repair only" in client.calls[1][-1]["content"]


def test_uncertain_prior_cannot_be_classed_as_conflict_without_glm_repair(tmp_path):
    client = ScriptedGLM([extraction(), "Unknown.", '{"novelty":"conflicts","agrees":null}',
                          '{"novelty":"new","agrees":null}'])
    assert run_extract(tmp_path, client)[0]["novelty"] == "new"
    assert len(client.calls) == 4


def test_replay_preserves_queue_and_does_not_repeat_prior(tmp_path):
    client = ScriptedGLM([extraction(), "Unknown.", '{"novelty":"new","agrees":null}'])
    run_extract(tmp_path, client)
    before = (tmp_path / "queue" / "candidates.jsonl").read_bytes()
    replay = ScriptedGLM([extraction()])
    assert run_extract(tmp_path, replay) == []
    assert len(replay.calls) == 1
    assert (tmp_path / "queue" / "candidates.jsonl").read_bytes() == before


def test_partial_progress_survives_failure_and_resume(tmp_path):
    first = json.loads(extraction())["candidates"][0]
    second_text = "Lumen's cabinet is green."
    second = {"text": second_text, "prompt_evidence": second_text,
              "probe_question": "What color is Lumen's cabinet?"}
    batch = json.dumps({"candidates": [first, second]})
    source = first["text"] + " " + second_text
    client = ScriptedGLM([batch, "Unknown.", '{"novelty":"new","agrees":null}', OSError("offline")])
    with pytest.raises(OSError):
        run_extract(tmp_path, client, source)
    assert len(latest_records(tmp_path / "queue" / "candidates.jsonl")) == 1
    resumed = ScriptedGLM([batch, "Unknown.", '{"novelty":"new","agrees":null}'])
    records = run_extract(tmp_path, resumed, source)
    assert len(records) == 1 and records[0]["text"] == second_text
    assert len(latest_records(tmp_path / "queue" / "candidates.jsonl")) == 2


def test_stdin_and_no_fact_source(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("Hello!"))
    client = ScriptedGLM(['{"candidates":[]}'])
    assert extract_file("-", client=client, source_kind="conversation",
                        queue_path=tmp_path / "queue" / "candidates.jsonl",
                        timings_path=tmp_path / "timings.jsonl") == []
    assert len(client.calls) == 1


@pytest.mark.parametrize("endpoint", ["http://glm.invalid:8888", "http://glm.invalid:8888/v1/"])
def test_client_uses_read_only_chat_and_records_trace(tmp_path, monkeypatch, endpoint):
    requests = []

    def respond(request, timeout):
        requests.append(request)
        assert timeout == 180
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "answer"}}]}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    trace = tmp_path / "trace.jsonl"
    client = GLMClient(endpoint, "test-model", trace_path=trace)
    messages = [{"role": "user", "content": "question"}]
    assert client.chat(messages) == "answer"
    assert requests[0].full_url == "http://glm.invalid:8888/v1/chat/completions"
    payload = json.loads(requests[0].data)
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert json.loads(trace.read_text())["messages"] == messages


def test_truncated_model_response_is_not_accepted(monkeypatch):
    response = {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}

    def respond(request, timeout):
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", respond)
    with pytest.raises(ValueError, match="truncated"):
        GLMClient("http://glm.invalid", "test-model").chat([])
