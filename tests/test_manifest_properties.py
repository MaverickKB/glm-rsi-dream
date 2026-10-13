"""Property checks for manifest identity, digests, and candidate files."""
import hashlib
import json
import uuid
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from glm_rsi_dream.files import digest, read_json, sha, write_json
from tensorfold_dream.eval_native import (
    MAGIC,
    checked_file,
    http_metadata,
    requested,
    selection,
    wire_extension,
)
from tensorfold_dream.pocket_native import load

PAIR = "ab" * 32
NAME = st.text(alphabet="abcdefghij", min_size=1, max_size=8)


def write_pair(root, binding, schema=3, pair_sha=None):
    manifest = {
        "schema_version": schema,
        "binding": binding,
        "pair_sha256": digest(binding) if pair_sha is None else pair_sha,
    }
    write_json(root / "pair.json", manifest)
    return manifest


def binding_for(prefix_bytes, slots=96, mechanism="attention_state_v1"):
    return {
        "mechanism": mechanism,
        "slots": slots,
        "prefix_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
    }


@given(st.dictionaries(NAME, st.integers(min_value=-1000, max_value=1000), max_size=6))
def test_digest_is_stable_when_keys_move(payload):
    again = dict(reversed(list(payload.items())))
    assert digest(payload) == digest(again)
    if payload:
        changed = dict(payload)
        key = next(iter(changed))
        changed[key] = changed[key] + 1
        assert digest(payload) != digest(changed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), {"slot": float("nan")}])
def test_nonfinite_json_cannot_be_hashed_or_written(tmp_path, value):
    with pytest.raises(ValueError):
        digest(value)
    with pytest.raises(ValueError):
        write_json(tmp_path / "numbers.json", value)


def test_read_json_rejects_nonfinite_tokens(tmp_path):
    path = tmp_path / "nan.json"
    path.write_text('{"slots": NaN}\n')
    with pytest.raises(ValueError, match="nonfinite"):
        read_json(path)


@given(st.binary(min_size=1, max_size=128), st.integers(min_value=0, max_value=127))
def test_sha_changes_when_one_byte_flips(tmp_path, raw, index):
    path = tmp_path / "blob.bin"
    path.write_bytes(raw)
    original = sha(path)
    mutable = bytearray(raw)
    mutable[index % len(mutable)] ^= 0x1
    path.write_bytes(bytes(mutable))
    assert sha(path) != original
    assert len(original) == 64


@given(NAME, st.binary(min_size=1, max_size=64))
def test_checked_file_accepts_only_pinned_relative_files(tmp_path, name, raw):
    root = tmp_path / uuid.uuid4().hex
    root.mkdir()
    relative = "mem/" + name + ".bin"
    path = root / relative
    path.parent.mkdir()
    path.write_bytes(raw)
    digest_hex = hashlib.sha256(raw).hexdigest()
    found = checked_file(root, {"path": relative, "sha256": digest_hex})
    assert found == path.resolve()
    with pytest.raises(ValueError, match="digest mismatch"):
        checked_file(root, {"path": relative, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="digest mismatch"):
        checked_file(root, {"path": relative, "sha256": "A" * 64})
    with pytest.raises(ValueError, match="manifest directory"):
        checked_file(root, {"path": "../" + name + ".bin", "sha256": digest_hex})
    with pytest.raises(ValueError, match="manifest directory"):
        checked_file(root, {"path": str(path), "sha256": digest_hex})


def test_checked_file_rejects_directories_and_non_objects(tmp_path):
    root = tmp_path / "manifest"
    nested = root / "nested"
    nested.mkdir(parents=True)
    with pytest.raises(ValueError, match="manifest directory"):
        checked_file(root, {"path": "nested", "sha256": "a" * 64})
    with pytest.raises(ValueError, match="manifest directory"):
        checked_file(root, ["nested", "a" * 64])


def test_load_accepts_a_paired_manifest(tmp_path):
    prefix = b'{"teacher_ids": [1, 2]}\n'
    (tmp_path / "prefix.json").write_bytes(prefix)
    binding = binding_for(prefix)
    write_pair(tmp_path, binding)
    loaded = load(tmp_path / "pair.json", SimpleWorld(2))
    assert loaded["pair_sha256"] == digest(binding)
    assert loaded["active"] == {}
    assert loaded["root"] == tmp_path


class SimpleWorld:
    def __init__(self, world):
        self.world = world


@pytest.mark.parametrize("mutate", ["schema", "mechanism", "slots", "pair", "world", "prefix", "nan"])
def test_load_rejects_a_broken_manifest(tmp_path, mutate):
    prefix = b'{"teacher_ids": [1]}\n'
    (tmp_path / "prefix.json").write_bytes(prefix)
    binding = binding_for(prefix)
    schema = 3
    pair_sha = None
    world = 2
    if mutate == "schema":
        schema = 2
    elif mutate == "mechanism":
        binding["mechanism"] = "other"
    elif mutate == "slots":
        binding["slots"] = 95
    elif mutate == "pair":
        pair_sha = "c" * 64
    elif mutate == "world":
        world = 1
    elif mutate == "prefix":
        (tmp_path / "prefix.json").write_bytes(prefix + b" ")
    if mutate == "nan":
        (tmp_path / "pair.json").write_text('{"schema_version": NaN}\n')
    else:
        write_pair(tmp_path, binding, schema=schema, pair_sha=pair_sha)
    with pytest.raises(ValueError):
        load(tmp_path / "pair.json", SimpleWorld(world))


@given(st.text(alphabet="abcXYZ0123456789_-", min_size=0, max_size=120))
def test_selection_accepts_only_exact_request_ids(request_id):
    spec = {"tag": "glm-eval-v1", "request_id": request_id, "arm": "main", "pair_sha256": PAIR}
    metadata = {"dream_eval": spec}
    allowed = 1 <= len(request_id) <= 96
    if allowed:
        assert selection(metadata, PAIR) == spec
    else:
        with pytest.raises(ValueError):
            selection(metadata, PAIR)


def test_eval_wire_is_absent_until_a_request_is_tagged():
    assert requested(None) is False
    assert requested({"dream_eval": {"arm": "none"}}) is True
    assert http_metadata({"ignored": 1, "dream_eval": {"arm": "none"}}) == {"dream_eval": {"arm": "none"}}
    assert http_metadata(None) == {}
    assert wire_extension(None, None) == []
    encoded = wire_extension(None, {"arm": "none"})
    assert encoded[0] == MAGIC
    assert encoded[1] == len(encoded) - 2
    with pytest.raises(ValueError, match="bound"):
        wire_extension({"capture": "x" * 70000}, None)


def test_selection_rejects_capture_mixed_with_eval():
    metadata = {"dream_eval": {"tag": "glm-eval-v1"}, "dream_capture": {"tag": "x"}}
    with pytest.raises(ValueError, match="mutually exclusive"):
        selection(metadata, PAIR)


def test_round_trip_json_preserves_digest(tmp_path):
    payload = {"slots": 96, "nested": {"arm": "main", "n": 0}}
    write_json(tmp_path / "value.json", payload)
    assert digest(read_json(tmp_path / "value.json")) == digest(payload)
    text = Path(tmp_path / "value.json").read_text()
    assert json.loads(text) == payload
