"""Pinned native source compilation and inspectable source identity, CPU-only."""
import gzip
import inspect
import json
import linecache
import types
from pathlib import Path

import pytest

from tensorfold_dream import bootstrap, capture_native, eval_native

SOURCES = json.loads(gzip.decompress((Path(__file__).resolve().parent/'upstream_sources.json.gz').read_bytes()))


def test_startup_source_check_uses_packaged_manifest(tmp_path, monkeypatch):
    import hashlib

    from glm_rsi_dream.manifest import MANIFEST

    source = "# fictional upstream source\n"
    expected = hashlib.sha256(source.encode()).hexdigest()
    path = tmp_path / "cli.py"
    path.write_text(source)
    monkeypatch.setitem(MANIFEST, "engine", {"source_files": {"cli.py": expected}})
    monkeypatch.setattr(bootstrap, "HANDLER_HASHES", {"cli.py": expected})
    bootstrap.verify_source_directory(tmp_path)
    path.write_text(source + "# changed\n")
    with pytest.raises(ValueError, match="startup source mismatch"):
        bootstrap.verify_source_directory(tmp_path)


@pytest.mark.parametrize('fullname,name', list(bootstrap.OVERLAY_MODULES.items()))
def test_overlay_compiles_and_preserves_inspect_source(fullname, name, tmp_path):
    relative = fullname.replace('tensorfold.', '', 1).replace('.', '/') + '.py'
    original = SOURCES[relative]
    path = tmp_path/Path(relative).name
    path.write_text(original)
    transformed = bootstrap.transform(name, original)
    loader = bootstrap.OverlayLoader(fullname, str(path))
    before = dict(capture_native.PROCESS_CAPTURE)
    capture_native.PROCESS_CAPTURE.update(failures=0, disabled=False)
    try:
        code = loader.get_code(fullname)
        assert loader.get_source(fullname) == transformed
        assert path.read_text() == original
        assert ''.join(linecache.getlines(code.co_filename)) == transformed
        functions = []
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType) and constant.co_name != '<module>':
                functions.append(constant)
        assert functions
        for child in functions:
            if child.co_name.startswith('<') or child.co_freevars:
                continue
            function = types.FunctionType(child, {})
            lines, start = inspect.getsourcelines(function)
            assert lines == transformed.splitlines(True)[start-1:start-1+len(lines)]
        with pytest.raises(ValueError, match='source mismatch'):
            bootstrap.transform(name, original+'\n')
    finally:
        capture_native.PROCESS_CAPTURE.clear()
        capture_native.PROCESS_CAPTURE.update(before)


def test_tagged_wire_leaves_untagged_bytes_absent_and_round_trips():
    assert eval_native.wire_extension(None, None) == []
    spec = {'tag':'glm-eval-v1', 'request_id':'cpu-test', 'arm':'main', 'pair_sha256':'a'*64}
    payload = eval_native.wire_extension(None, spec)
    record = eval_native.parse_admit({}, payload, 0)
    assert record['eval'] == spec
    assert eval_native.selection({'dream_eval':spec}, 'a'*64) == spec
    for field,value in [('pair_sha256','b'*64),('request_id','../escape'),('arm','control')]:
        altered = dict(spec)
        altered[field] = value
        with pytest.raises(ValueError):
            eval_native.selection({'dream_eval':altered}, 'a'*64)
