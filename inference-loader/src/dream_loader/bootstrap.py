"""Two source-pinned insertion points; installed stock source stays intact."""

import hashlib
import importlib.abc
import importlib.machinery
import json
import linecache
import sys
from pathlib import Path

PINS = json.loads(Path(__file__).with_name("pins.json").read_text())
PREFIX = "tensorfold.families.glm5_next.cuda."
MODULES = {PREFIX + "engine": "families/glm5_next/cuda/engine.py",
           PREFIX + "forward": "families/glm5_next/cuda/forward.py"}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Stock loader anchor absent or ambiguous: " + old.strip())
    return source.replace(old, new, 1)


def transform(fullname, source):
    relative = MODULES[fullname]
    expected = PINS["engine_source_files"][relative]
    if hashlib.sha256(source.encode()).hexdigest() != expected:
        raise ValueError("Stock source pin mismatch: " + relative)
    source = replace_once(source, "from __future__ import annotations\n",
                          "from __future__ import annotations\nfrom dream_loader import runtime as _dream\n")
    if fullname.endswith(".engine"):
        return replace_once(source, "        self.w = w\n",
                            "        self.w = w\n        _dream.attach(self, model_dir)\n")
    source = replace_once(source, '            with prof.timed("moe: combine"):\n',
                        '            _dream.shared(layer, w, b, lo, hi)\n'
                        '            with prof.timed("moe: combine"):\n')
    source = replace_once(source,
                          '    g = _ffn(layer, w, b, R)\n'
                          '    glue.hc_post(x, x, g, b.post[:R], b.comb[:R])\n',
                          '    _dream.before_ffn(layer, w, b, R)\n'
                          '    g = _ffn(layer, w, b, R)\n'
                          '    glue.hc_post(x, x, g, b.post[:R], b.comb[:R])\n'
                          '    _dream.after_ffn(layer, w, b, R, g)\n')
    # Stock HC row splitting performs hc_post inside its exchange and bypasses
    # layer_forward. Select the stock unsplit path for four-stream additions.
    source = replace_once(source, '    split = b.split is not None and b.split.applies(R)\n',
                          '    split = not _dream.four_streams(w) and b.split is not None and b.split.applies(R)\n')
    return replace_once(source,
                        '    if b.prefill and b.split is not None and sparse_np is None and b.split.applies(R):\n',
                        '    if not _dream.four_streams(w) and b.prefill and b.split is not None and sparse_np is None and b.split.applies(R):\n')


def verify_sources(root):
    for relative, expected in PINS["engine_source_files"].items():
        path = Path(root) / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Stock source pin mismatch: " + relative)


class Loader(importlib.machinery.SourceFileLoader):
    def get_source(self, fullname):
        if hasattr(self, "compiled_source"):
            return self.compiled_source
        return super().get_source(fullname)

    def get_code(self, fullname):
        original = self.get_data(self.path).decode()
        source = transform(fullname, original)
        filename = self.path + ".glm-dream-" + hashlib.sha256(source.encode()).hexdigest()
        code = compile(source, filename, "exec")
        self.compiled_source = source
        linecache.cache[filename] = (len(source.encode()), None, source.splitlines(True), filename)
        return code

    def exec_module(self, module):
        code = self.get_code(module.__name__)
        module.__file__ = code.co_filename
        exec(code, module.__dict__)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ImportError("Stock TensorFold must expose source-backed modules")
        spec.loader = Loader(fullname, spec.origin)
        return spec


def install():
    for fullname in MODULES:
        if fullname in sys.modules:
            raise RuntimeError("Install the loader before TensorFold engine imports")
    for finder in sys.meta_path:
        if isinstance(finder, Finder):
            return
    spec = importlib.machinery.PathFinder.find_spec("tensorfold")
    if spec is None or not spec.submodule_search_locations or len(spec.submodule_search_locations) != 1:
        raise ImportError("One stock TensorFold source directory is required")
    verify_sources(Path(spec.submodule_search_locations[0]))
    # Load dependencies before modifying the import path; failures propagate.
    from . import runtime  # noqa: F401

    sys.meta_path.insert(0, Finder())
