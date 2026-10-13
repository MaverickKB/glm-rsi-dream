"""Package existing rank additions for inference. No optimizer or fitting path."""

import argparse
import json
from pathlib import Path

from .artifact import MODEL, digest, geometry, identity, load_pair, read_rank


def bundle(rank0, rank1, config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new bundle directory")
    raw_config = Path(config).read_bytes()
    shape = geometry(json.loads(raw_config))
    payloads = []
    layers = []
    ranks = []
    mechanisms = []
    for rank, source in enumerate((rank0, rank1)):
        raw = Path(source).read_bytes()
        mechanism, sites = read_rank(raw, rank, shape["hidden_size"])
        layers.append(sites)
        mechanisms.append(mechanism)
        payloads.append(raw)
        ranks.append({"rank": rank, "path": f"rank{rank}.pt", "sha256": digest(raw)})
    if mechanisms[0] != mechanisms[1]:
        raise ValueError("Both ranks must use the same shared mechanism")
    if set(layers[0]) != set(layers[1]):
        raise ValueError("Both ranks must carry identical layer placements")
    for index in layers[0]:
        if layers[0][index]["gate"].shape != layers[1][index]["gate"].shape:
            raise ValueError("Paired addition widths differ")
    binding = {"mechanism": mechanisms[0], "model": MODEL, "world": 2,
               "model_config_sha256": digest(raw_config), "geometry": shape, "ranks": ranks}
    manifest = {"schema_version": 1, "binding": binding, "pair_sha256": identity(binding)}
    output.mkdir(parents=True)
    for rank, raw in enumerate(payloads):
        (output / f"rank{rank}.pt").write_bytes(raw)
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    load_pair(path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("bundle")
    create.add_argument("--rank0", type=Path, required=True)
    create.add_argument("--rank1", type=Path, required=True)
    create.add_argument("--model-config", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "bundle":
        manifest = bundle(args.rank0, args.rank1, args.model_config, args.output)
    else:
        manifest, _ = load_pair(args.manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
