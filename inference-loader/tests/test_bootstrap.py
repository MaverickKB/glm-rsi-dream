import gzip
import inspect
import json
import linecache
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dream_loader import bootstrap, runtime, serve
from dream_loader.artifact import FOUR_STREAMS, MECHANISM

SOURCES = json.loads(gzip.decompress(Path(__file__).with_name("upstream_sources.json.gz").read_bytes()))


@pytest.mark.parametrize("fullname", list(bootstrap.MODULES))
def test_pinned_source_compiles_and_remains_inspectable(tmp_path, fullname):
    original = SOURCES[bootstrap.MODULES[fullname]]
    path = tmp_path / "native.py"
    path.write_text(original)
    loader = bootstrap.Loader(fullname, str(path))
    code = loader.get_code(fullname)
    transformed = bootstrap.transform(fullname, original)
    assert path.read_text() == original
    assert loader.get_source(fullname) == transformed
    assert "tensorfold_dream" not in transformed
    assert "glm_rsi_dream" not in transformed
    assert "_capture" not in transformed
    assert "_eval" not in transformed
    if fullname.endswith("engine"):
        assert transformed.index("_dream.attach(self, model_dir)") < transformed.index("self.e = Engine(w")
    assert "".join(linecache.getlines(code.co_filename)) == transformed
    for child in code.co_consts:
        if not isinstance(child, types.CodeType) or child.co_freevars or child.co_name.startswith("<"):
            continue
        lines, start = inspect.getsourcelines(types.FunctionType(child, {}))
        assert lines == transformed.splitlines(True)[start - 1:start - 1 + len(lines)]
    with pytest.raises(ValueError, match="pin mismatch"):
        bootstrap.transform(fullname, original + "\n")


def test_source_directory_pin_validation(tmp_path, monkeypatch):
    pins = {}
    for relative, source in SOURCES.items():
        if relative in bootstrap.PINS["engine_source_files"]:
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
            pins[relative] = bootstrap.PINS["engine_source_files"][relative]
    monkeypatch.setitem(bootstrap.PINS, "engine_source_files", pins)
    bootstrap.verify_sources(tmp_path)
    (tmp_path / next(iter(pins))).write_text("changed")
    with pytest.raises(ValueError, match="pin mismatch"):
        bootstrap.verify_sources(tmp_path)


def test_late_install_rejected(monkeypatch):
    monkeypatch.setitem(sys.modules, bootstrap.PREFIX + "engine", SimpleNamespace())
    with pytest.raises(RuntimeError, match="before"):
        bootstrap.install()


def test_serve_delegates_to_original_with_identical_arguments(monkeypatch):
    calls = []
    original = SimpleNamespace(load=lambda: lambda: calls.append(list(sys.argv)))
    monkeypatch.setattr(serve, "entry_points", lambda **kwargs: [original])
    monkeypatch.setattr(serve, "install", lambda: calls.append("installed"))
    monkeypatch.setenv(runtime.ENV, "/opt/dream/addition/manifest.json")
    args = ["tensorfold", "serve", "/model", "--tp", "2", "--rank", "1", "--master", "head"]
    monkeypatch.setattr(sys, "argv", args)
    serve.main()
    assert calls == ["installed", args]
    assert "--port" not in calls[1]


def test_serve_missing_manifest_never_invokes_engine(monkeypatch):
    original = SimpleNamespace(load=lambda: pytest.fail("original must not run"))
    monkeypatch.setattr(serve, "entry_points", lambda **kwargs: [original])
    monkeypatch.delenv(runtime.ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["tensorfold", "serve", "/model"])
    with pytest.raises(ValueError, match="required"):
        serve.main()


@pytest.mark.parametrize("mechanism", [MECHANISM, FOUR_STREAMS])
@pytest.mark.parametrize("name", ["_compute", "compute_steps"])
def test_four_stream_prompts_use_stock_unsplit_path(mechanism, name):
    import ast

    source = bootstrap.transform(bootstrap.PREFIX + "forward",
                                 SOURCES[bootstrap.MODULES[bootstrap.PREFIX + "forward"]])
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            function = node
            break
    calls = []

    def split_steps(*args):
        calls.append("split")
        yield 35

    namespace = {"Weights": object, "State": object, "Buffers": object, "torch": torch,
                 "_dream": runtime, "layer_forward": lambda *args: calls.append("unsplit"),
                 "split_layers": lambda *args: calls.append("split"), "split_layer_steps": split_steps,
                 "glue": SimpleNamespace(embed=lambda *args: None, stream_mean=lambda *args: None,
                                         rmsnorm=lambda *args: None)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "stock-prompt-dispatch", "exec"), namespace)
    tensor = torch.zeros(3, 4)
    split = SimpleNamespace(applies=lambda rows: True, finish=lambda: None, stream=None)
    b = SimpleNamespace(prefill=True, split=split, ids=tensor, x=tensor, hidden=tensor,
                        fnormed=tensor, fxs=tensor, tap_at={})
    w = SimpleNamespace(meta={runtime.KEY: {"mechanism": mechanism}},
                        cfg=SimpleNamespace(hidden=4, streams=4, eps=1e-5),
                        layers=[SimpleNamespace(index=35)], embed=None, norm=None)
    if name == "_compute":
        namespace[name](w, None, b, 3, logits=False)
    else:
        assert list(namespace[name](w, None, b, 3, head=False)) == [35]
    assert calls == ["unsplit" if mechanism == FOUR_STREAMS else "split"]
