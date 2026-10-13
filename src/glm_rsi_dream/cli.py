"""Offline planning and explicit Spark-head execution."""
import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_RUN_DIR, expand_home, load_config, require_spark
from .files import read_json
from .manifest import MANIFEST


def build_parser():
    parser = argparse.ArgumentParser(description=MANIFEST['name'] + ', ' + MANIFEST['status'])
    parser.add_argument('--config', default='config.example.json',
                        help='config file (default: config.example.json)')
    parser.add_argument('--run', default=DEFAULT_RUN_DIR,
                        help='private run directory (default: ~/.cache/glm-rsi-dream/demo-1)')
    parser.add_argument('--lesson', default='examples/synthetic/lesson.json')
    parser.add_argument('--eval', default='examples/synthetic/eval.json')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--question')
    parser.add_argument('command', choices=['run', 'ask', 'restore', 'status'])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.run = expand_home(args.run)
    try:
        if args.command == 'status':
            path = Path(args.run) / 'state.json'
            state = read_json(path) if path.exists() else {'phase': 'absent'}
            result = {'phase': state['phase'], 'release': MANIFEST['status']}
            print(json.dumps(result, indent=2))
            return 0
        config = load_config(args.config, live=args.live and not args.dry_run)
        if not args.live or args.dry_run:
            from .dataset import validate
            if args.command == 'run':
                validate(read_json(args.lesson), read_json(args.eval))
            print(json.dumps({'mode': 'dry-run', 'release': MANIFEST['status'],
                              'command': args.command, 'target': 'head Spark and worker Spark',
                              'steps': ['capture GLM teacher attention', 'train 96 rows at 11 DSA layers',
                                        'GLM grade free answers', 'keep/reject', 'restore exact originals']}, indent=2))
            return 0
        require_spark()
        from .loop import ask, run
        from .native_cluster import AttentionCluster
        cluster = AttentionCluster(config, args.run)
        if args.command == 'run':
            result = run(cluster, args.lesson, args.eval)
        elif args.command == 'ask':
            if not args.question:
                raise ValueError('ask requires --question')
            result = ask(cluster, args.question)
        else:
            cluster.restore()
            result = {'restored': cluster.state['phase'] == 'restored'}
        public = {key: value for key, value in result.items() if key != 'selected'}
        print(json.dumps(public, indent=2))
        if result.get('verdict') == 'reject':
            return 2
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1
