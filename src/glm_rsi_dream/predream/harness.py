"""Run the admin-controlled funnel on an external host using generation only."""

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import control
from .absorb import absorb, stage
from .extract import GLMClient, extract_candidates
from .grade import grade_pending
from .policy import load_policy, record_policy
from .queue import format_table, latest_records
from .timing import StageTimer, now_et, summarize_timings


def prepare(root: Any, policy_path: Any) -> Any:
    policy = load_policy(policy_path)
    record_policy(root, policy)
    settings = control.settings_for(root)
    if settings["mode"] != "harness":
        raise ValueError("The harness entrypoint requires control.mode=harness")
    return policy


def handoff(root: Any, run_id: Any) -> Any:
    """The shared controller and handoff modules own all retention authority."""
    if control.settings_for(root)["mode"] != "harness":
        raise ValueError("The harness entrypoint requires control.mode=harness")
    accepted = []
    for record in latest_records(Path(root) / "queue/candidates.jsonl"):
        if record["status"] == "accepted":
            control.authorize_handoff(root, record)
            accepted.append(record["id"])
    if not accepted:
        return {"accepted_ready": [], "memory": None, "staging": None}
    return {"accepted_ready": accepted, "memory": absorb(root), "staging": stage(root, run_id)}


def receipt(root: Any, result: Any) -> Any:
    root = Path(root)
    counts: dict[Any, Any] = {}
    proposals = []
    for record in latest_records(root / "queue/candidates.jsonl"):
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1
        if "proposal" in record:
            proposals.append(record["id"])
    result.update({
        "observed_et": now_et(), "host": socket.gethostname(), "pid": os.getpid(),
        "mode": "harness", "status_counts": counts, "awaiting_admin": proposals,
        "timings": summarize_timings(root / "timings.jsonl"),
    })
    root.mkdir(parents=True, exist_ok=True)
    (root / "harness-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_funnel(source_path: Any, root: Any, *, policy_path: Any, endpoint: Any, run_id: Any,
               model: Any = "GLM-5.3-Flash-EXL3", timeout: Any = 180, source_ref: Any = None, client: Any = None) -> Any:
    """Collect, propose, list and hand off only controller-approved candidates."""
    address = urlsplit(endpoint)
    if address.scheme not in ("http", "https") or address.port != 8888:
        raise ValueError("Harness generation endpoint must use HTTP(S) port8888")
    if address.username is not None or address.password is not None:
        raise ValueError("Generation endpoint must not contain credential material")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    root = Path(root)
    source_path = Path(source_path)
    source_text = source_path.read_text(encoding="utf-8")
    policy = prepare(root, policy_path)
    if client is None:
        client = GLMClient(endpoint, model, timeout=timeout, trace_path=root / "traces/glm.jsonl")
    collected = extract_candidates(
        source_text, source_ref or str(source_path.resolve()), client=client,
        queue_path=root / "queue/candidates.jsonl", timings_path=root / "timings.jsonl",
        policy=policy["extract"],
    )
    grades = grade_pending(root, client, policy=policy["grade"])
    with StageTimer(root, "queue_list", "run"):
        table = format_table(latest_records(root / "queue/candidates.jsonl"))
    result = {"collected": len(collected), "grading": grades, "queue_table": table,
              "endpoint": endpoint, "errors": grades["errors"]}
    if not grades["errors"]:
        result["handoff"] = handoff(root, run_id)
    return receipt(root, result)


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Private local run directory")
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("GLM_ENDPOINT"))
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--timeout", type=float, default=180)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Collect and propose; retain only controller-approved items")
    run.add_argument("input", type=Path)
    run.add_argument("--source-ref")
    run.add_argument("--run-id", required=True)
    final = commands.add_parser("handoff", help="Persist approved memory and raw staging after admin review")
    final.add_argument("--run-id", required=True)
    commands.add_parser("list", help="List local queue rows without generation")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            if args.endpoint is None:
                raise ValueError("Supply --endpoint or GLM_ENDPOINT")
            result = run_funnel(
                args.input, args.root, policy_path=args.policy_config, endpoint=args.endpoint,
                model=args.model, timeout=args.timeout, run_id=args.run_id, source_ref=args.source_ref,
            )
        else:
            prepare(args.root, args.policy_config)
            if args.command == "handoff":
                result = receipt(args.root, {"handoff": handoff(args.root, args.run_id), "errors": []})
            else:
                with StageTimer(args.root, "queue_list", "run"):
                    print(format_table(latest_records(args.root / "queue/candidates.jsonl")))
                return 0
    except (ValueError, RuntimeError, OSError) as error:
        print("harness: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 1 if result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
