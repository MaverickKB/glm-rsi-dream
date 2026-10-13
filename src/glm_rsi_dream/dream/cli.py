"""CPU dream training and explicit paired artifact lifecycle commands."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..files import read_json
from .driver import CommandPair, FilesystemPair
from .service import DreamLifecycle


def build_parser() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-dream-features", action="store_true", help="Explicit host development opening after qualification")
    parser.add_argument("--root", type=Path, required=True, help="LOOP2 storage root")
    drivers = parser.add_mutually_exclusive_group()
    drivers.add_argument("--harness-pair", type=Path, help="Offline paired state, never native evidence")
    drivers.add_argument("--native-driver", type=Path, help="Owner-pinned runner and bounded authority receipts")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    register = commands.add_parser("register-pocket")
    register.add_argument("spec", type=Path)
    start = commands.add_parser("start")
    start.add_argument("--run-id", required=True)
    start.add_argument("--pocket", action="append", required=True, dest="pockets")
    start.add_argument("--training", type=Path, required=True)
    start.add_argument("--width", type=int, default=32)
    start.add_argument("--steps", type=int, default=1000)
    qualify = commands.add_parser("qualify")
    qualify.add_argument("--run-id", required=True)
    qualify.add_argument("--suite", type=Path, required=True)
    qualify.add_argument("--baseline", type=Path, required=True)
    qualify.add_argument("--report", type=Path, required=True)
    qualify.add_argument("--evidence-kind", choices=("harness", "native"), required=True)
    for name in ("activate", "retire"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
    commands.add_parser("rollback")
    commands.add_parser("recover")
    for name, command in commands.choices.items():
        if name not in ("status", "recover"):
            command.add_argument("--confirm", action="store_true")
    return parser


def dispatch(args: Any) -> Any:
    driver: Any = None
    if args.harness_pair:
        driver = FilesystemPair(args.harness_pair)
    if args.native_driver:
        driver = CommandPair(args.native_driver)
    lifecycle = DreamLifecycle(args.root, driver, dream_features_enabled=args.enable_dream_features)
    if args.command == "status":
        result = lifecycle.state()
        result["features"] = {"dream": {"enabled": args.enable_dream_features, "label": "Under development"}}
        path = lifecycle.home / "operation.json"
        result["operation_status"] = read_json(path)["status"] if path.exists() else "absent"
        return result
    if args.command == "register-pocket":
        return lifecycle.register_pocket(read_json(args.spec), confirm=args.confirm)
    if args.command == "start":
        return lifecycle.start(args.run_id, args.pockets, read_json(args.training),
                               confirm=args.confirm, width=args.width, steps=args.steps)
    if args.command == "qualify":
        return lifecycle.qualify(args.run_id, args.suite, args.baseline, args.report,
                                 evidence_kind=args.evidence_kind, confirm=args.confirm)
    if args.command == "activate":
        return lifecycle.activate(args.run_id, confirm=args.confirm)
    if args.command == "retire":
        return lifecycle.retire(args.run_id, confirm=args.confirm)
    if args.command == "recover":
        return lifecycle.recover()
    return lifecycle.rollback(confirm=args.confirm)


def main(argv: Any = None) -> Any:
    args = build_parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print("dream: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
