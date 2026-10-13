"""Configurable GLM novelty experiments, with judgments made by GLM."""

import json
import math
from typing import Any

from .extract import (
    COMPARE_INSTRUCTIONS,
    PRIOR_INSTRUCTIONS,
    ask_json,
    validate_comparison,
)

ASSESSMENT_INSTRUCTIONS = (
    "Assess whether you already knew the candidate before this request. "
    "This experiment deliberately shows the claim to you. Do not treat merely "
    "seeing it as prior knowledge. Return JSON with answer (your prior knowledge), "
    "agrees (true/false/null), novelty (known/conflicts/new), and reason. "
    "Unknown or uncertain prior knowledge means new/null. "
    "Existing support means known/true; incompatible prior means conflicts/false."
)


def resolve_policy(policy: Any = None) -> Any:
    settings: dict[str, Any] = {
        "strategy": "closed_book", "probe_samples": 1, "probe_temperature": None,
        "agreement_threshold": 1.0, "prior_instructions": PRIOR_INSTRUCTIONS,
        "comparison_instructions": COMPARE_INSTRUCTIONS,
        "assessment_instructions": ASSESSMENT_INSTRUCTIONS,
    }
    if policy is not None:
        for name, value in policy.items():
            if name == "extraction_instructions":
                continue
            if name not in settings:
                raise ValueError("Unknown novelty setting: " + name)
            settings[name] = value
    if settings["strategy"] not in ("closed_book", "claim_assessment"):
        raise ValueError("Novelty strategy must be closed_book or claim_assessment")
    samples = settings["probe_samples"]
    if type(samples) is not int or samples < 1:
        raise ValueError("probe_samples must be a positive integer")
    for name in ("prior_instructions", "comparison_instructions", "assessment_instructions"):
        if not isinstance(settings[name], str) or not settings[name].strip():
            raise ValueError(name + " must be nonempty text")
    threshold = settings["agreement_threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("agreement_threshold must be between zero and one")
    temperature = settings["probe_temperature"]
    if temperature is not None:
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("probe_temperature must be finite and nonnegative")
    if settings["strategy"] == "claim_assessment":
        if samples != 1 or temperature is not None or threshold != 1:
            raise ValueError("Sampling controls apply to closed_book; claim_assessment uses one direct judgment")
    return settings


def judge_novelty(candidate: Any, client: Any, policy: Any = None) -> Any:
    settings = resolve_policy(policy)
    answers = []
    if settings["strategy"] == "closed_book":
        for sample in range(settings["probe_samples"]):
            messages = [{"role": "system", "content": settings["prior_instructions"]},
                        {"role": "user", "content": candidate["probe_question"]}]
            if settings["probe_temperature"] is None:
                answer = client.chat(messages)
            else:
                answer = client.chat(messages, temperature=settings["probe_temperature"])
            answers.append(answer)
        payload = {"candidate": candidate["text"], "prior_answer": answers[0]}
        instructions = settings["comparison_instructions"]
        instructions += "\nInclude a brief reason in the JSON response."
        if len(answers) > 1:
            payload = {"candidate": candidate["text"], "prior_answers": answers,
                       "agreement_threshold": settings["agreement_threshold"]}
            instructions += (
                "\nAssess every sampled prior answer. Mark known only when the fraction "
                "explicitly supporting the claim meets agreement_threshold. Mark conflicts "
                "only when the fraction explicitly contradicting it meets the same threshold. "
                "Otherwise mark new. Give a reason for your judgment."
            )
        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(payload)}]
        comparison = ask_json(client, messages, validate_comparison)
    else:
        instructions = settings["assessment_instructions"]

        def validate_assessment(value: Any) -> Any:
            validate_comparison(value)
            if not isinstance(value.get("answer"), str):
                raise ValueError("Assessment requires a prior answer string")

        messages = [{"role": "system", "content": instructions},
                    {"role": "user", "content": candidate["text"]}]
        comparison = ask_json(client, messages, validate_assessment)
        answers.append(comparison["answer"])
    return {"answer": answers[0], "answers": answers, "comparison": comparison,
            "policy": settings, "claim_visible_during_prior": settings["strategy"] == "claim_assessment"}
