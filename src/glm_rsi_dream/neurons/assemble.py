"""Align sealed native teacher/student rows into a training-only packet."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .scope import verify_examples, verify_scope


def load_capture(root: Any, entry: Any, rank: Any, memory: Any) -> Any:
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Capture escaped the sealed training root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Native capture hash differs from its rank receipt")
    packet = torch.load(path, weights_only=True, map_location="cpu")
    if packet["rank"] != rank or packet["world"] != 2:
        raise ValueError("Native capture rank identity differs")
    if packet["attention_memory"] != memory:
        raise ValueError("Teacher/student attention state differs from declared arm")
    return packet


def rows(chunks: Any, shift: Any) -> Any:
    positions = []
    inputs = []
    post = []
    residual = []
    for chunk in chunks:
        for position in chunk["positions"]:
            positions.append(position - shift)
        inputs.append(chunk["input"])
        post.append(chunk["post"])
        residual.append(chunk["residual"])
    if positions != sorted(set(positions)):
        raise ValueError("Native capture positions are duplicated or unordered")
    return positions, torch.cat(inputs), torch.cat(post), torch.cat(residual)


def assemble(root: Any, index: Any, output: Any, rank: Any) -> Any:
    root = Path(root).resolve()
    index_path = root / index
    sealed = json.loads(index_path.read_text())
    if sealed.get("role") != "train" or sealed.get("sealed_before_fitting") is not True:
        raise ValueError("Only a sealed training capture index may enter fitting")
    study_sha, expected = verify_scope(root)
    if sealed.get('study_sha256') != study_sha:
        raise ValueError('Capture index differs from the frozen study')
    verify_examples(sealed['examples'], expected)
    original = root / 'capture-index-original.json'
    if original.exists():
        if json.loads(original.read_text())['examples'] != sealed['examples']:
            raise ValueError('Reused capture membership or tensors changed')
    result = {"schema_version": 1, "role": "train", "rank": rank, "world": 2,
              "study_sha256": study_sha,
              "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(), "layers": {}}
    buffers: dict[Any, Any] = {}
    seen = set()
    for example in sealed["examples"]:
        if example["id"] in seen:
            raise ValueError("Duplicate training example")
        seen.add(example["id"])
        positive = example["positive"]
        if not isinstance(positive, bool):
            raise ValueError("Training scope label must be boolean")
        student = load_capture(root, example["student"][rank], rank, False)
        teacher = load_capture(root, example["teacher"][rank], rank, positive)
        if student["layers"].keys() != teacher["layers"].keys():
            raise ValueError("Teacher/student layer coverage differs")
        for layer, chunks in student["layers"].items():
            positions, inputs, post, residual = rows(chunks, example["student_shift"])
            teacher_positions, _, _, desired = rows(teacher["layers"][layer], example["teacher_shift"])
            if positions != teacher_positions:
                raise ValueError("Teacher/student token positions differ")
            buffer = buffers.setdefault(layer, {"input": [], "post": [], "student": [],
                                               "teacher": [], "positive": [], 'first_question': []})
            buffer["input"].append(inputs)
            buffer["post"].append(post)
            buffer["student"].append(residual)
            buffer["teacher"].append(desired)
            buffer["positive"].append(torch.full((len(positions),), positive, dtype=torch.bool))
            first = torch.zeros(len(positions), dtype=torch.bool)
            first[0] = positive
            buffer['first_question'].append(first)
    for layer, buffer in buffers.items():
        result["layers"][layer] = {}
        for name, values in buffer.items():
            result["layers"][layer][name] = torch.cat(values)
    if not result["layers"]:
        raise ValueError("Training index contains no native layer rows")
    torch.save(result, output)


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("index")
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    assemble(args.root, args.index, args.output, args.rank)


if __name__ == "__main__":
    main()
