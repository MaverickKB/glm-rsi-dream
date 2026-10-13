"""Relocate a bundle and read it with the current attention and shared consumers."""

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from glm_rsi_dream.export import FOUR_STREAMS, sha256
from glm_rsi_dream.files import read_json, write_json
from glm_rsi_dream.pocket_bundle import import_bundle, verify_bundle


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PairedCPUExchange:
    """A deterministic communication double, with exact local-partial checks."""

    def __init__(self, rank, partials):
        self.rank = rank
        self.partials = partials

    def all_gather(self, local, output):
        expected = self.partials[self.rank]
        if not torch.equal(local.reshape_as(expected), expected):
            raise ValueError("Actual consumer produced a different local four-stream correction")
        output.copy_(torch.stack(self.partials).reshape_as(output))


def frozen_mix(before, combine, post, base, delta):
    # Tensor dimensions express the stream contract independently of the hook.
    streams = before.float().unsqueeze(2)
    coefficients = combine.unsqueeze(3)
    mixed = (streams * coefficients).sum(dim=1)
    branch_sum = base.unsqueeze(1) + delta
    branch = branch_sum.to(before.dtype).float()
    output = mixed + post.unsqueeze(2) * branch
    return output.to(before.dtype)


def prove_four_streams(native, loader, worlds):
    native_cases = 0
    loader_cases = 0
    independent_cases = 0
    zero_effect_cases = 0
    original_records = native.records
    native.records = lambda w, b: iter(b.proof_records)
    hidden = worlds[0].cfg.hidden
    torch.manual_seed(1835)
    try:
        for index in worlds[0].meta[loader.KEY]["layers"]:
            for dtype in (torch.bfloat16, torch.float32):
                for scale in (0.0, 0.1, 1.0, 10.0):
                    inputs = (torch.randn(10, hidden) * scale).to(dtype)
                    before = torch.randn(10, 4, hidden).to(dtype)
                    combine = torch.rand(10, 4, 4)
                    for _ in range(20):
                        combine /= combine.sum(dim=2, keepdim=True)
                        combine /= combine.sum(dim=1, keepdim=True)
                    post = torch.rand(10, 4)
                    gathered = torch.randn(2, 10, hidden)  # Native branch partials are FP32.
                    base = gathered[0, :8].float() + gathered[1, :8].float()
                    for partial_rows in (False, True):
                        partials = []
                        for w in worlds:
                            state = w.meta[native.KEY]
                            selected = read_json(state["root"] / "neurons-current.json")["by_rank"][w.rank]
                            site = native.artifact(w, selected)["layers"][index]
                            gate = torch.nn.functional.linear(inputs[:8].float(), site["gate"]).clamp(max=10)
                            up = torch.nn.functional.linear(inputs[:8].float(), site["up"]).clamp(-10, 10)
                            activation = torch.nn.functional.silu(gate) * up
                            delta = torch.nn.functional.linear(activation, site["down"]).reshape(8, 4, hidden)
                            if partial_rows:
                                delta[0].zero_()
                                delta[7].zero_()
                            partials.append(delta)
                        total = partials[0] + partials[1]
                        expected = frozen_mix(before[:8], combine[:8], post[:8], base, total)
                        averaged = total.mean(dim=1, keepdim=True).expand_as(total)
                        lossy = frozen_mix(before[:8], combine[:8], post[:8], base, averaged)
                        if scale == 0:
                            if torch.count_nonzero(total):
                                raise ValueError("Zero activation changed the base branch")
                            zero_effect_cases += 1
                        else:
                            if torch.equal(expected, lossy):
                                raise ValueError("Proof artifact fails to distinguish independent stream writes")
                            independent_cases += 1
                        records = [(None, None, 0, 8)]
                        if partial_rows:
                            records = [({"policy": {"neurons_off": True}}, None, 0, 1),
                                       (None, None, 1, 6),
                                       ({"policy": {"neurons_off": True}}, None, 7, 1)]
                        paired_outputs = []
                        for w in worlds:
                            w.comm = PairedCPUExchange(w.rank, partials)
                            original_x = before.reshape(10, 4 * hidden)
                            buffer = SimpleNamespace(x=original_x.clone(), normed=inputs,
                                                     sy=torch.zeros(10, hidden), comb=combine, post=post,
                                                     prefill=scale in (0.0, 1.0), proof_records=records)
                            layer = SimpleNamespace(index=index)
                            native.before_ffn(layer, w, buffer, 8)
                            native.shared(layer, w, buffer, 0, 3)
                            native.shared(layer, w, buffer, 3, 8)
                            if torch.count_nonzero(buffer.sy):
                                raise ValueError("Four-stream writes changed the base shared vector")
                            native.after_ffn(layer, w, buffer, 8, gathered)
                            actual = buffer.x[:8].reshape(8, 4, hidden)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            if (not torch.equal(buffer.x[8:], original_x[8:])
                                    or buffer._dream_delta is not None or buffer._dream_before is not None):
                                raise ValueError("Owner hook changed inactive rows or retained temporary state")
                            paired_outputs.append(actual.clone())
                            native_cases += 1
                            if partial_rows:
                                continue
                            other = SimpleNamespace(x=original_x.clone(), normed=inputs,
                                                    sy=torch.zeros(10, hidden), comb=combine, post=post)
                            loader.before_ffn(layer, w, other, 8)
                            loader.shared(layer, w, other, 0, 3)
                            loader.shared(layer, w, other, 3, 8)
                            loader.after_ffn(layer, w, other, 8, gathered)
                            if not torch.equal(other.x, buffer.x):
                                raise ValueError("Relocated inference loader differs from the owner writer")
                            loader_cases += 1
                        if not torch.equal(paired_outputs[0], paired_outputs[1]):
                            raise ValueError("Paired residual outputs disagree")
    finally:
        native.records = original_records
    return {"native_four_stream_cases_exact": native_cases, "loader_four_stream_cases_exact": loader_cases,
            "independent_stream_counterexamples": independent_cases, "hidden": hidden,
            "zero_effect_cases_exact": zero_effect_cases,
            "dtypes": ["bfloat16", "float32"], "partial_rows_and_off_policy_verified": True,
            "scheduling_branches": ["prefill", "decode"],
            "paired_exchange": "CPU communication double", "maximum_absolute_error": 0.0}


def prove(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh proof directory")
    pocket_path = args.attention_source / "tensorfold_dream/pocket_native.py"
    automatic_path = args.attention_source / "tensorfold_dream/automatic_native.py"
    sources = {}
    for path in (pocket_path, automatic_path, args.neurons_source,
                 args.loader_source / "dream_loader/artifact.py", args.loader_source / "dream_loader/runtime.py"):
        sources[str(path.resolve())] = sha256(path)
    verified = verify_bundle(args.bundle, args.base, args.base_identity)
    args.output.mkdir(parents=True)
    relocated = args.output / "moved-server"
    relocated.mkdir()
    shutil.copytree(args.base, relocated / "base")
    shutil.copyfile(args.base_identity, relocated / "base-identity.json")
    shutil.copytree(args.bundle, relocated / "transfer-bundle")
    boot = import_bundle(relocated / "transfer-bundle", relocated / "personalized-v1",
                         relocated / "base", relocated / "base-identity.json")
    sys.path.insert(0, str(args.attention_source))
    pocket = source_module("proof_pocket_native", pocket_path)
    automatic = source_module("proof_automatic_native", automatic_path)
    native = source_module("proof_neurons_native", args.neurons_source)
    sys.path.insert(0, str(args.loader_source))
    from dream_loader import runtime as loader_runtime
    from dream_loader.artifact import load_pair
    from dream_loader.runtime import KEY, load_local, shared

    loaded = relocated / "personalized-v1"
    text = read_json(relocated / "base/config.json")
    text = text.get("text_config", text)
    torch.manual_seed(3251)
    states_read = 0
    attention_tensor_pairs = 0
    shared_cases = 0
    max_error = 0.0
    worlds = []
    mechanism = None
    for item in boot["ranks"]:
        rank = item["rank"]
        cfg = SimpleNamespace(hidden=text["hidden_size"], streams=4, limit=10, quant="exl3", dense_limit=8192)
        layers = []
        for _ in range(text["num_hidden_layers"]):
            layers.append(SimpleNamespace(moe=SimpleNamespace(shared=object())))
        w = SimpleNamespace(world=2, rank=rank, device="cpu", cfg=cfg, layers=layers, meta={})
        state = pocket.load(loaded / item["attention_manifest"], w)
        state["prefix"] = read_json(state["root"] / "prefix.json")
        w.meta["tensorfold_dream.eval"] = state
        catalog = automatic.configuration(state)
        if catalog is None:
            raise ValueError("Proof requires an enabled source catalog")
        for entry in catalog["pockets"]:
            router = entry.get("routers", [entry.get("router")])[0]
            question = "What " + router["subjects"][0] + " does " + router["owner"] + " have?"
            prepared = automatic.prepare(SimpleNamespace(w=w), {"prompt": question})
            selected = prepared.get("metadata", {}).get("dream_eval", {}).get("request_id", "")
            if not selected.startswith("auto-" + entry["key"] + "-" + catalog["revision"] + "-"):
                raise ValueError("Native ordinary admission did not select the supplied pocket")
            request = "auto-" + entry["key"] + "-" + catalog["revision"] + "-" + "0" * 32
            policy = automatic.policy(w, {"spec": {"request_id": request}}, [1, 2, 3])
            if policy["selected"] != entry["selected"][rank] or policy["full_snapshot"] != entry["full"][rank]:
                raise ValueError("Native automatic policy selected the wrong rank")
            for kind in ("selected", "full"):
                artifact = entry[kind][rank]
                restored = torch.load(state["root"] / artifact["path"], weights_only=True, map_location="cpu")
                original_root = args.bundle / f"attention/rank{rank}"
                original = torch.load(original_root / artifact["path"], weights_only=True, map_location="cpu")
                for before, after in zip(original["rows"], restored["rows"]):
                    if not torch.equal(before, after):
                        raise ValueError("Relocation changed an attention tensor")
                    attention_tensor_pairs += 1
                states_read += 1
        if boot["shared_manifest"]:
            path = loaded / boot["shared_manifest"]
            manifest, _ = load_pair(path)
            os.environ["TF_GLM_DREAM_MANIFEST"] = str(path)
            runtime = load_local(SimpleNamespace(w=w), relocated / "base")
            w.meta[KEY] = runtime
            current = read_json(state["root"] / "neurons-current.json")
            original = native.artifact(w, current["by_rank"][rank])
            mechanism = original["mechanism"]
            if mechanism == FOUR_STREAMS:
                worlds.append(w)
            for index, site in original["layers"].items():
                if mechanism == FOUR_STREAMS:
                    # Its parity probe includes the entire paired HC update below.
                    continue
                for scale in (0.1, 1.0, 10.0):
                    inputs = torch.randn(8, cfg.hidden) * scale
                    gate = torch.nn.functional.linear(inputs, site["gate"]).clamp(max=10)
                    up = torch.nn.functional.linear(inputs, site["up"]).clamp(-10, 10)
                    expected = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, site["down"])
                    buffer = SimpleNamespace(normed=inputs, sy=torch.zeros_like(expected))
                    shared(SimpleNamespace(index=index), w, buffer, 0, len(inputs))
                    error = float((buffer.sy - expected).abs().max())
                    max_error = max(max_error, error)
                    if not torch.equal(buffer.sy, expected):
                        raise ValueError("Imported stock-loader correction changed")
                    shared_cases += 1
            if manifest["binding"]["model_config_sha256"] != verified["base"]["config_sha256"]:
                raise ValueError("Stock loader consumed the wrong base config")
    result = {"bundle_sha256": verified["bundle_sha256"], "world": 2,
              "attention_states_read": states_read, "attention_tensor_pairs_exact": attention_tensor_pairs,
              "shared_cases_exact": shared_cases, "shared_max_absolute_error": max_error,
              "native_source_sha256": sources, "cpu_consumer_round_trip": True,
              "proof_tool_sha256": sha256(__file__),
              "cuda_adapter_exercised": False, "native_server_boot_verified": False,
              "semantic_recall_verified": False}
    if mechanism == FOUR_STREAMS:
        result["four_streams"] = prove_four_streams(native, loader_runtime, worlds)
    for path, expected_hash in sources.items():
        if sha256(path) != expected_hash:
            raise ValueError("Consumer source changed during the proof: " + path)
    write_json(args.output / "proof.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "base", "base-identity", "attention-source", "neurons-source", "loader-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prove(args), indent=2))


if __name__ == "__main__":
    main()
