"""Pocket-OFF evaluation through caller-owned generation and runtime readers.

This module imports only the standard library. It opens no network connection,
process, GPU or output directory. The caller supplies all live operations.
"""

import copy
import hashlib
import json
import re
import time
import uuid
from typing import Any


def digest(value: Any) -> Any:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_suite(suite: Any) -> Any:
    if (
        not isinstance(suite, dict)
        or type(suite.get("schema_version")) is not int
        or suite.get("schema_version") != 1
    ):
        raise ValueError("A version 1 suite is required")
    if not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("A suite needs a nonempty case list")
    identifiers = set()
    for case in suite["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Each case must be a JSON object")
        if (
            not isinstance(case.get("id"), str)
            or not case["id"]
            or case["id"] in identifiers
        ):
            raise ValueError("Case identifiers must be unique nonempty strings")
        identifiers.add(case["id"])
        if case.get("kind") not in ("recall", "exact", "leak"):
            raise ValueError("Case kind must be recall, exact or leak")
        if not isinstance(case.get("group"), str) or not case["group"]:
            raise ValueError("Every case needs a report group")
        prompt = case.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Every case needs a nonempty bare question")
        if case["kind"] == "recall":
            if (
                not isinstance(case.get("expected"), list)
                or not case["expected"]
                or not case.get("facts")
            ):
                raise ValueError("Recall needs expected values and authorized facts")
            for value in case["expected"]:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Expected values must be nonempty strings")
                pattern = r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)"
                if re.search(pattern, prompt.casefold()):
                    raise ValueError(
                        "Recall question reveals an expected answer: " + case["id"]
                    )
        if case["kind"] == "leak" and not case.get("facts"):
            raise ValueError("Leak grading needs the authorized fact catalog")
    parameters = suite.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("Generation parameters must be a JSON object")
    allowed = {
        "model",
        "max_tokens",
        "temperature",
        "draft",
        "stream",
        "chat_template_kwargs",
    }
    if set(parameters) - allowed:
        raise ValueError(
            "Generation parameters cannot add context, tools or answer hints"
        )
    if (
        parameters.get("temperature", 0) != 0
        or parameters.get("draft", False) is not False
    ):
        raise ValueError("Preservation uses deterministic generation without drafting")
    if parameters.get("stream", False) is not False:
        raise ValueError("Evaluation requires complete nonstreaming replies")
    maximum = parameters.get("max_tokens", 512)
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if parameters.get("chat_template_kwargs", {"enable_thinking": False}) != {
        "enable_thinking": False
    }:
        raise ValueError(
            "Template options must keep thinking disabled without extra context"
        )
    return suite


def request_for(case: Any, parameters: Any, probe_id: Any) -> Any:
    request = copy.deepcopy(parameters)
    request.setdefault("temperature", 0)
    request.setdefault("max_tokens", 512)
    request.setdefault("stream", False)
    request.setdefault("draft", False)
    request.setdefault("chat_template_kwargs", {"enable_thinking": False})
    request["messages"] = [{"role": "user", "content": case["prompt"]}]
    # Existing automatic-memory code recognizes this opaque audit identity.
    # It contains no fact, pocket route, memory tag or teacher context.
    request["metadata"] = {"rsi_probe": probe_id}
    return request


def validate_runtime(snapshot: Any, probe_id: Any, phase: Any) -> Any:
    if not isinstance(snapshot, dict):
        raise ValueError("Runtime reader must return a JSON object")
    if snapshot.get("probe_id") != probe_id or not snapshot.get("epoch"):
        raise ValueError("Runtime readback must bind this probe and a serving epoch")
    for name in ("pair_sha256", "base_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get(name, ""))):
            raise ValueError("Runtime needs a verified " + name)
    ranks = snapshot.get("ranks", [])
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise ValueError("Pocket-OFF proof needs both TP ranks")
    seen = set()
    stable_ranks: dict[Any, Any] = {}
    for rank in ranks:
        if not isinstance(rank, dict):
            raise ValueError("Each rank readback must be a JSON object")
        rank_id = rank.get("tp_rank")
        if type(rank_id) is not int or rank_id not in (0, 1) or rank_id in seen:
            raise ValueError("Runtime needs distinct integer ranks 0 and 1")
        seen.add(rank_id)
        if rank.get("tp_world") != 2 or rank.get("pockets_enabled") is not False:
            raise ValueError("Every attention pocket must be disabled on both ranks")
        rows = rank.get("active_memory_rows")
        if rank.get("enabled_pockets") != [] or type(rows) is not int or rows != 0:
            raise ValueError(
                "An enabled pocket or active memory row invalidates recall"
            )
        if rank.get("pocket_cache_clean") is not True:
            raise ValueError("Clear pocket-conditioned KV caches before evaluation")
        expected_neurons = phase == "consolidated"
        if rank.get("neurons_enabled") is not expected_neurons:
            raise ValueError("Neuron state differs from the requested evaluation phase")
        artifact = rank.get("neurons_sha256")
        if expected_neurons and not re.fullmatch(r"[0-9a-f]{64}", str(artifact)):
            raise ValueError(
                "Consolidated evaluation needs each loaded neuron artifact hash"
            )
        if not expected_neurons and artifact is not None:
            raise ValueError("Baseline must have no neuron artifact loaded")
        stable_ranks[rank_id] = {}
        for name in (
            "tp_rank",
            "tp_world",
            "pockets_enabled",
            "enabled_pockets",
            "active_memory_rows",
            "pocket_cache_clean",
            "neurons_enabled",
            "neurons_sha256",
        ):
            stable_ranks[rank_id][name] = rank[name]
    stable = {
        "epoch": snapshot["epoch"],
        "pair_sha256": snapshot["pair_sha256"],
        "base_sha256": snapshot["base_sha256"],
        "ranks": stable_ranks,
    }
    return digest(stable)


def reply_signature(reply: Any, require_complete: Any = True) -> Any:
    if not isinstance(reply, dict):
        raise ValueError("Generation must return a JSON reply object")
    choices = reply.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("A reply must contain one complete choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Reply choice must be a JSON object")
    message = choice.get("message", {})
    if not isinstance(message, dict):
        raise ValueError("Reply message must be an object")
    content = message.get("content")
    if (
        not isinstance(content, str)
        or not content.strip()
        or choice.get("finish_reason") not in ("stop", "length")
        or (require_complete and choice.get("finish_reason") != "stop")
    ):
        raise ValueError("Reply is empty, malformed or incomplete")
    usage = reply.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Reply usage must be a JSON object")
    tokens = usage.get("completion_tokens")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError(
            "Exact preservation requires the actual completion token count"
        )
    tensorfold = reply.get("tensorfold", {})
    if not isinstance(tensorfold, dict):
        raise ValueError("Reply token identity metadata must be a JSON object")
    return {
        "message": copy.deepcopy(message),
        "finish_reason": choice["finish_reason"],
        "completion_tokens": tokens,
        "token_sha": tensorfold.get("token_sha"),
    }


def semantic_pass(case: Any, judgment: Any) -> Any:
    if (
        not isinstance(judgment, dict)
        or not isinstance(judgment.get("reason"), str)
        or not judgment["reason"].strip()
    ):
        raise ValueError("Semantic grader needs an explicit reason")
    fields = ("correct", "grounded") if case["kind"] == "recall" else ("leaked",)
    for field in fields:
        if type(judgment.get(field)) is not bool:
            raise ValueError("Semantic grader field must be boolean: " + field)
    if case["kind"] == "recall":
        return judgment["correct"] and judgment["grounded"]
    return not judgment["leaked"]


def evaluate(
    suite: Any, generate: Any, judge: Any, inspect_runtime: Any, *, phase: Any = "consolidated", baseline: Any = None
) -> Any:
    """Return raw observations and honest counts, with missing rows in denominators.

    generate(request) returns one OpenAI-shaped reply. judge(case, answer) is a
    separate semantic reviewer. inspect_runtime(probe_id) reads both ranks.
    Run phase=baseline before enabling the new neurons; pass that report here.
    All attention pockets and their conditioned caches stay OFF in both phases.
    """
    suite = copy.deepcopy(validate_suite(suite))
    if phase not in ("baseline", "consolidated"):
        raise ValueError("Phase must be baseline or consolidated")
    suite_hash = digest(suite)
    references = {}
    if phase == "consolidated":
        if (
            not isinstance(baseline, dict)
            or baseline.get("phase") != "baseline"
            or not baseline.get("complete")
        ):
            raise ValueError(
                "Consolidated evaluation needs a complete pocket-OFF baseline"
            )
        if baseline.get("suite_sha256") != suite_hash:
            raise ValueError("Baseline suite and request parameters must match")
        sealed = copy.deepcopy(baseline)
        expected_hash = sealed.pop("report_sha256", None)
        if digest(sealed) != expected_hash:
            raise ValueError("Baseline report content seal changed")
        for record in baseline["records"]:
            if record.get("error") or not record.get("reply_signature"):
                raise ValueError("Baseline has an incomplete observation")
            before_hash = validate_runtime(
                record["runtime_before"], record["probe_id"], "baseline"
            )
            after_hash = validate_runtime(
                record["runtime_after"], record["probe_id"], "baseline"
            )
            if (
                before_hash != after_hash
                or reply_signature(record["reply"], record['kind'] == 'exact') != record["reply_signature"]
            ):
                raise ValueError(
                    "Baseline runtime or raw reply differs from its receipt"
                )
            references[record["id"]] = record
        expected_ids = set()
        for case in suite["cases"]:
            expected_ids.add(case["id"])
            reference = references.get(case["id"])
            if reference is None:
                raise ValueError("Baseline is missing a case")
            expected_request = request_for(
                case, suite.get("parameters", {}), reference["probe_id"]
            )
            if reference["request"] != expected_request:
                raise ValueError(
                    "Baseline request contains different context or parameters"
                )
        if set(references) != expected_ids:
            raise ValueError("Baseline must cover every case once")
    records = []
    stable_runtime = None
    interrupted = None
    for case in suite["cases"]:
        probe_id = uuid.uuid4().hex
        request = request_for(case, suite.get("parameters", {}), probe_id)
        record = {
            "id": case["id"],
            "group": case["group"],
            "kind": case["kind"],
            "probe_id": probe_id,
            "request": copy.deepcopy(request),
            "passed": False,
            "runtime_verified": False,
        }
        started = time.monotonic()
        try:
            if interrupted is not None:
                raise ValueError("Not run after runtime failure: " + interrupted)
            before = copy.deepcopy(inspect_runtime(probe_id))
            record["runtime_before"] = before
            before_hash = validate_runtime(before, probe_id, phase)
            if stable_runtime is None:
                stable_runtime = before_hash
            if stable_runtime != before_hash:
                raise ValueError("Runtime changed within the evaluation pass")
            if baseline and before["base_sha256"] != baseline["base_sha256"]:
                raise ValueError("Base model differs from the frozen baseline")
            reply = copy.deepcopy(generate(copy.deepcopy(request)))
            record["reply"] = reply
            after = copy.deepcopy(inspect_runtime(probe_id))
            record["runtime_after"] = after
            if validate_runtime(after, probe_id, phase) != before_hash:
                raise ValueError("Runtime changed during generation")
            record["runtime_verified"] = True
            signature = reply_signature(reply, case['kind'] == 'exact')
            record["reply_signature"] = signature
            if case["kind"] == "exact":
                if phase == "baseline":
                    record["passed"] = None
                else:
                    record["passed"] = (
                        signature == references[case["id"]]["reply_signature"]
                    )
            elif signature['finish_reason'] != 'stop':
                # A valid runtime can produce a failed, truncated factual answer.
                # Count that failure and keep measuring the remaining questions.
                # Exact controls still require complete baseline/candidate replies.
                record['passed'] = False
                record['answer_incomplete'] = True
            else:
                judgment = copy.deepcopy(
                    judge(copy.deepcopy(case), signature["message"]["content"])
                )
                record["judgment"] = judgment
                record["passed"] = semantic_pass(case, judgment)
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            OSError,
            RuntimeError,
        ) as error:
            record["error"] = str(error)
            # Lost runtime evidence closes generation; answer/grade errors stay measurable.
            if not record['runtime_verified']:
                interrupted = str(error)
        record["wall_ms"] = (time.monotonic() - started) * 1000
        records.append(record)
    groups: dict[Any, Any] = {}
    complete = True
    runtime_verified = True
    passed = phase == "consolidated"
    for record in records:
        group = groups.setdefault(
            record["group"], {"total": 0, "passed": 0, "errors": 0}
        )
        group["total"] += 1
        group["passed"] += record["passed"] is True
        group["errors"] += "error" in record
        if "error" in record:
            complete = False
        if record["passed"] is not True:
            passed = False
        if not record["runtime_verified"]:
            runtime_verified = False
    report = {
        "schema_version": 1,
        "phase": phase,
        "suite_sha256": suite_hash,
        "groups": groups,
        "complete": complete,
        "passed": passed,
        "runtime_verified": runtime_verified,
        "records": records,
        "baseline_comparison": phase == "consolidated",
        "semantic_grading": "Caller-supplied reviewer; separate from exact byte/token checks",
    }
    if records and isinstance(records[0].get("runtime_before"), dict):
        report["base_sha256"] = records[0]["runtime_before"].get("base_sha256")
    report["report_sha256"] = digest(report)
    return report
