import json

import pytest

from glm_rsi_dream.predream.metrics import class_metrics
from glm_rsi_dream.predream.novelty import judge_novelty, resolve_policy
from glm_rsi_dream.predream.novelty_eval import run_evaluation


class Client:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def chat(self, messages, **options):
        self.requests.append({"messages": messages, "options": options})
        return next(self.replies)


def test_sample_policy_changes_real_requests_without_local_verdicts():
    client = Client(["Unknown.", "Unknown.", json.dumps({"novelty": "new", "agrees": None, "reason": "Both priors unknown"})])
    candidate = {"text": "Fixture Zeta uses amber-41.", "probe_question": "What label does fixture Zeta use?"}
    result = judge_novelty(candidate, client, {"probe_samples": 2, "probe_temperature": 0.7, "agreement_threshold": 0.5})
    assert result["comparison"]["novelty"] == "new"
    assert result["answers"] == ["Unknown.", "Unknown."]
    assert result["claim_visible_during_prior"] is False
    for request in client.requests[:2]:
        assert request["options"] == {"temperature": 0.7}
        assert candidate["text"] not in json.dumps(request["messages"])
    payload = json.loads(client.requests[-1]["messages"][-1]["content"])
    assert payload["agreement_threshold"] == 0.5
    assert payload["prior_answers"] == ["Unknown.", "Unknown."]


def test_direct_assessment_is_an_explicit_alternative():
    client = Client([json.dumps({"novelty": "known", "agrees": True, "answer": "Water is H2O", "reason": "Prior chemical knowledge"})])
    candidate = {"text": "Water is H2O", "probe_question": "What is water's formula?"}
    result = judge_novelty(candidate, client, {"strategy": "claim_assessment"})
    assert result["claim_visible_during_prior"] is True
    assert client.requests[0]["messages"][-1]["content"] == candidate["text"]
    assert len(client.requests) == 1


def test_metrics_include_false_positive_misroute_and_ungraded_false_negative():
    result = class_metrics(["memory", "weights", "reject"], [None, "memory", "memory"])
    assert result["memory"] == {"tp": 0, "fp": 2, "fn": 1, "precision": 0, "recall": 0}
    assert result["weights"]["recall"] == 0 and result["weights"]["precision"] is None
    assert result["reject"]["fn"] == 1


def test_evaluation_hides_labels_preserves_policy_and_reports_class_metrics(tmp_path):
    cases = [{"text": "Water is H2O", "probe_question": "What is water's formula?", "expected": "known"},
             {"text": "Fixture Zeta uses amber-41", "probe_question": "What label does fixture Zeta use?", "expected": "new"}]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    client = Client(["Water is H2O", json.dumps({"novelty": "known", "agrees": True, "reason": "Support"}),
                     "Unknown", json.dumps({"novelty": "new", "agrees": None, "reason": "Unknown"})])
    root = tmp_path / "run"
    report = run_evaluation(path, root, client=client)
    assert report["per_class"]["known"]["precision"] == 1
    assert report["per_class"]["new"]["recall"] == 1
    for request in client.requests:
        assert '"expected"' not in json.dumps(request["messages"])
    with pytest.raises(ValueError, match="completed"):
        run_evaluation(path, root, client=client)


@pytest.mark.parametrize("settings", [{"strategy": "unknown"}, {"probe_samples": True},
                                     {"probe_samples": 0}, {"agreement_threshold": 0},
                                     {"probe_temperature": float("nan")}])
def test_bad_policy_fails_before_model_request(settings):
    with pytest.raises(ValueError):
        resolve_policy(settings)
