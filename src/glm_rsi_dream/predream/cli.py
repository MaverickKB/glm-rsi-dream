"""Inspect and prepare knowledge with an installed pre-dream pipeline."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .timing import StageTimer, summarize_timings


def build_parser() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-dream-features", action="store_true", help="Explicit host development opening after qualification")
    parser.add_argument("--root", type=Path, required=True, help="Run storage directory")
    parser.add_argument("--endpoint", default=os.environ.get("GLM_ENDPOINT", "http://127.0.0.1:8888"), help="GLM service URL before /v1")
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--policy-config", type=Path, help="JSON config for novelty and grading hypotheses")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("admin-init", help="Host-admin setup; never expose as a model tool")
    initialize.add_argument("--key-file", type=Path, required=True)
    grant = commands.add_parser("admin-grant", help="Issue bounded admin authority using the host signing key")
    grant.add_argument("--output", type=Path, required=True)
    grant.add_argument("--admin-id", required=True)
    grant.add_argument("--root-scope", type=Path, required=True)
    grant.add_argument("--mode", action="append", choices=("spark", "harness"), required=True)
    grant.add_argument("--autonomy", action="append", choices=("manual", "supervised", "automatic"), required=True)
    grant.add_argument("--hours", type=float, default=16)
    grant.add_argument("--action", action="append", dest="actions", help="Explicit bounded actions, including dream_start, dream_activate and dream_retire")
    approve = commands.add_parser("approve", help="Admin approval of explicitly selected pending proposals")
    approve.add_argument("--id", action="append", required=True, dest="identifiers")
    commands.add_parser("control-status", help="Show configured mode, autonomy and permission status")
    extract = commands.add_parser("extract", help="Extract and probe candidate knowledge")
    extract.add_argument("input", help="Source file or - for standard input")
    extract.add_argument("--source-ref")
    extract.add_argument("--source-kind", choices=("prompt", "conversation"), default="prompt")
    learn = commands.add_parser("learn", help="Queue deliberately prepared knowledge without novelty filtering")
    learn.add_argument("input", help="UTF-8 text or JSON facts pack")
    learn.add_argument("--source-ref")
    learn.add_argument("--learned-facts", type=Path, help="Existing learned facts with explicit provenance receipts")
    queue = commands.add_parser("queue", help="Inspect the append-only queue")
    queue_commands = queue.add_subparsers(dest="queue_command", required=True)
    learned = queue_commands.add_parser("import-learned", help="Register an explicit existing learned-fact catalog")
    learned.add_argument("input", type=Path)
    listing = queue_commands.add_parser("list")
    listing.add_argument("--status", choices=("pending", "accepted", "rejected", "absorbed", "staged", "learned"))
    grade = commands.add_parser("grade", help="Have GLM grade pending candidates")
    selection = grade.add_mutually_exclusive_group()
    selection.add_argument("--all-pending", action="store_true")
    selection.add_argument("--id", dest="candidate_id")
    commands.add_parser("absorb", help="Absorb accepted personal memory")
    stage = commands.add_parser("stage", help="Prepare accepted weights and pockets")
    stage.add_argument("--run-id", required=True)
    dream = commands.add_parser("dream", help="Preview staging or start shared-expert consolidation")
    dream.add_argument("--run-id", help="Preview this staged run, or all runs when omitted")
    dream.add_argument("--manual", action="store_true", required=True)
    dream.add_argument("--confirm", action="store_true", help="Explicit human authorization for training")
    dream.add_argument("--pocket", action="append", dest="pockets", help="Registered accepted teacher pocket")
    dream.add_argument("--training", type=Path, help="Sealed paired shared-expert training plan")
    dream.add_argument("--width", type=int, default=32)
    dream.add_argument("--steps", type=int, default=1000)
    commands.add_parser("timings", help="Summarize recorded wall times")
    evaluate = commands.add_parser("evaluate", help="Evaluate a labeled grading set with GLM")
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--component", choices=("novelty", "grade", "funnel"), default="grade")
    evaluate.add_argument("--queue-from", type=Path, help="Preserved original pending snapshots for a funnel transfer experiment")
    evaluate.add_argument("--learned-facts", type=Path, help="Admin-attested catalog for duplicate/update evaluation")
    return parser


def stage_module(name: Any) -> Any:
    try:
        return importlib.import_module("." + name, __package__)
    except ModuleNotFoundError as error:
        if error.name != __package__ + "." + name:
            raise
        raise RuntimeError(name + " lane has not installed its module yet") from error


def dispatch(args: Any) -> Any:
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.command in ("admin-init", "admin-grant", "approve", "control-status"):
        from . import control

        if args.command == "admin-init":
            return control.initialize_key(args.key_file)
        if args.command == "admin-grant":
            return control.issue_permission(args.output, admin_id=args.admin_id, root_scope=args.root_scope,
                                            modes=args.mode, autonomy=args.autonomy, hours=args.hours, actions=args.actions)
        if args.command == "approve":
            return control.approve(args.root, args.identifiers)
        result = {"settings": control.settings_for(args.root), "authorized": False}
        try:
            _, admin = control.permission_for(args.root, "decide")
            result["authorized"] = True
            result["admin_id"] = admin["admin_id"]
        except (ValueError, OSError, KeyError) as error:
            result["blocker"] = str(error)
        return result
    if args.command == "timings":
        return summarize_timings(args.root / "timings.jsonl")
    settings = {}
    if args.command in ("extract", "grade", "evaluate") and args.policy_config is not None:
        from .policy import load_policy, record_policy

        policy = load_policy(args.policy_config)
        record_policy(args.root, policy)
        section = "grade"
        if args.command == "extract" or (args.command == "evaluate" and args.component == "novelty"):
            section = "extract"
        settings["policy"] = policy[section]
        if args.command == "evaluate" and args.component == "funnel":
            settings["policy"] = policy
    if args.command == "evaluate":
        if args.queue_from is not None and args.component != "funnel":
            raise ValueError("--queue-from requires --component funnel")
        if args.component in ("novelty", "funnel"):
            module_name = "novelty_eval" if args.component == "novelty" else "funnel_eval"
            module = stage_module(module_name)
            extract_module = stage_module("extract")
            client = extract_module.GLMClient(
                args.endpoint, args.model, timeout=args.timeout,
                trace_path=args.root / "traces" / "glm.jsonl",
            )
            if args.component == "funnel":
                settings["queue_from"] = args.queue_from
                settings["learned_facts"] = args.learned_facts
            return module.run_evaluation(args.cases, args.root, client=client, **settings)
        module = stage_module("grade_sanity")
        return module.run_sanity(
            args.cases, args.root, endpoint=args.endpoint, model=args.model,
            timeout=args.timeout, **settings,
        )
    if args.command == "extract":
        if args.input == "-":
            text = sys.stdin.read()
            source_ref = args.source_ref or "stdin"
        else:
            source = Path(args.input)
            text = source.read_text(encoding="utf-8")
            source_ref = args.source_ref or str(source)
        module = stage_module("extract")
        client = module.GLMClient(
            args.endpoint, args.model, timeout=args.timeout,
            trace_path=args.root / "traces" / "glm.jsonl",
        )
        return module.extract_candidates(
            text, source_ref, source_kind=args.source_kind, client=client,
            queue_path=args.root / "queue" / "candidates.jsonl",
            timings_path=args.root / "timings.jsonl",
            **settings,
        )
    if args.command == "queue":
        if args.queue_command == "import-learned":
            return stage_module("manual").import_learned(args.root, args.input)
        module = stage_module("queue")
        with StageTimer(args.root, "queue_list", "run"):
            records = module.list_candidates(args.root / "queue" / "candidates.jsonl", status=args.status)
            print(module.format_table(records))
        return None
    if args.command == "learn":
        module = stage_module("manual")
        extract_module = stage_module("extract")
        if args.learned_facts is not None:
            module.import_learned(args.root, args.learned_facts)
        client = extract_module.GLMClient(
            args.endpoint, args.model, timeout=args.timeout,
            trace_path=args.root / "traces" / "glm.jsonl",
        )
        return module.learn_file(args.root, args.input, client=client, source_ref=args.source_ref)
    if args.command == "grade":
        module = stage_module("grade")
        extract_module = stage_module("extract")
        client = extract_module.GLMClient(
            args.endpoint, args.model, timeout=args.timeout,
            trace_path=args.root / "traces" / "glm.jsonl",
        )

        def grade_request(prompt: Any) -> Any:
            return client.chat([{"role": "user", "content": prompt}])

        return module.grade_pending(
            args.root, grade_request, all_pending=args.candidate_id is None,
            candidate_id=args.candidate_id, **settings,
        )
    module = stage_module("absorb")
    if args.command == "absorb":
        return module.absorb(args.root)
    if args.command == "stage":
        return module.stage(args.root, args.run_id)
    if args.command == "dream":
        if not args.enable_dream_features:
            raise ValueError("feature_unavailable: under development")
        if args.confirm:
            if not args.run_id or not args.pockets or args.training is None:
                raise ValueError("Dream requires --run-id, --pocket and --training for shared-expert consolidation")
            from ..dream.service import DreamLifecycle
            from ..files import read_json

            return DreamLifecycle(args.root, dream_features_enabled=args.enable_dream_features).start(args.run_id, args.pockets, read_json(args.training),
                                                   confirm=True, width=args.width, steps=args.steps)
        plan = module.dream_plan(args.root, args.run_id)
        plan["confirmation_requirements"] = [
            "Stage attention pockets and independently accept their real paired artifacts first",
            "Register accepted teacher pockets with glm-dream-trigger register-pocket",
            "Supply --confirm, --run-id, --pocket and a sealed paired --training plan",
            "Dream uses the shared-expert trainer on CPU; native evaluation and paired reload have separate authority",
        ]
        return plan
    raise ValueError("unknown command")


def main(argv: Any = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, RuntimeError, OSError) as error:
        print("predream: " + str(error), file=sys.stderr)
        return 1
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if isinstance(result, dict) and result.get("errors"):
        return 1
    return 0
