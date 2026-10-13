"""Run offline replay or a separately enabled collector, with no service install."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .tap import TapGLMClient, drain
from .tap_policy import TapPolicy
from .tap_replay import replay
from .tap_store import inspect_spool, retry_failed, revoke


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    recorded = commands.add_parser("replay")
    recorded.add_argument("trace", type=Path)
    recorded.add_argument("--principal", required=True)
    recorded.add_argument("--opt-in", action="store_true")
    collect = commands.add_parser("collect")
    collect.add_argument("--endpoint", required=True)
    collect.add_argument("--watch", action="store_true")
    collect.add_argument("--interval", type=float, default=5)
    collect.add_argument("--max-events", type=int, default=4)
    removal = commands.add_parser("revoke")
    removal.add_argument("--principal", required=True)
    commands.add_parser("inspect")
    commands.add_parser("retry", help="Retry failed events after correcting their input/policy/model blocker")
    args = parser.parse_args(argv)
    try:
        policy = TapPolicy.load(args.policy)
        if args.command == "inspect":
            result = inspect_spool(args.root)
        elif args.command == "retry":
            result = retry_failed(args.root)
        elif args.command == "revoke":
            result = revoke(args.root, args.principal)
        elif not policy.enabled:
            result = {"status": "disabled"}
        elif args.command == "replay":
            result = replay(args.trace, args.root, policy=policy, principal=args.principal, opt_in=args.opt_in)
        else:
            if args.endpoint.rstrip("/") != policy.endpoint.rstrip("/"):
                raise ValueError("Collector must use the configured head endpoint")
            address = urlsplit(args.endpoint)
            if address.scheme not in {"http", "https"} or address.port != 8888 or not address.hostname:
                raise ValueError("Collector requires the configured head HTTP(S)8888 endpoint")
            if address.username is not None or address.password is not None:
                raise ValueError("Collector endpoint must not include credentials")
            if args.interval < 1:
                raise ValueError("Collector interval must be at least one second")
            while True:
                # Reload the switch for every pass so a running collector can stop.
                policy = TapPolicy.load(args.policy)
                if not policy.enabled:
                    result = {"status": "disabled"}
                    break
                client = TapGLMClient(args.endpoint, "GLM-5.3-Flash-EXL3", timeout=30, trace_path=None)
                result = drain(args.root, policy=policy, client=client, max_events=args.max_events)
                if not args.watch:
                    break
                print(json.dumps(result), flush=True)
                time.sleep(args.interval)
        print(json.dumps(result, indent=2))
        return 1 if result.get("status") in {"error", "event_call_limit"} else 0
    except (OSError, ValueError, KeyError, TypeError):
        print("tap: input or storage validation failed", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
