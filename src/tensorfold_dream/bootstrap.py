# SPDX-License-Identifier: Apache-2.0
"""Read-only, hash-pinned import overlay. Installed by the package .pth on BOTH ranks.

Only TF_DREAM_CAPTURE=1 or TF_DREAM_EVAL=1 installs the overlay. Off leaves engine source,
startup settings and ADMIT bytes unchanged on both ranks.
No engine files are overwritten. Enabled source is compiled with a few
explicit observer/apply calls. Evaluation startup and tagged admission agree on both ranks.
The .pth format requires one physical import line. Its expanded try block
logs a capture-only failure and lets native serving start.
"""

import hashlib
import importlib.abc
import importlib.machinery
import linecache
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = "tensorfold.families.glm5_next.cuda."
HANDLER_HASHES = {
    "server/app.py": "66b41e8c769abf10ca79bebc180165c3bcc9b4dcb31679f7eb4667100fdf05e5",
    "server/http.py": "4ecfccb9fa1d9925903ddecfcd556c06ebfae09170b8562f58ed33c9f4a34652",
    "families/glm5_next/cuda/app.py": "fe97ccb18a3a2c63ab905f5ade0ddd778a2a623d15a237d740bbfaca82b1396a",
    "cuda/server.py": "3d870bf08152668d312324c63e2d8ef0bbaa1db951425c12476aa23500a3b991",
    "cuda/http.py": "c1c2eb0db9a20926fb39200dd5e126363b63391153217a4ada0f4b9ac242f2e1",
}
OVERLAY_MODULES = {
    PACKAGE + "engine": "engine",
    PACKAGE + "multi": "multi",
    PACKAGE + "forward": "forward",
    PACKAGE + "app": "glm_app",
    PACKAGE + "decode": "decode",
    PACKAGE + "verify": "verify",
    PACKAGE + "multi_prefill": "multi_prefill",
}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("capture overlay anchor is absent or ambiguous: " + old[:80])
    return source.replace(old, new, 1)


def transform(name, source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    # Full hashes are retained in the packaged source-pin resource.

    from glm_rsi_dream.manifest import MANIFEST
    pinned = MANIFEST["engine"]
    if digest != pinned["files"][name]:
        raise ValueError(f"capture overlay source mismatch: {name}")
    source = replace_once(source, "from __future__ import annotations\n",
                          "from __future__ import annotations\nfrom tensorfold_dream import capture_native as _capture\n")
    if name == "glm_app":
        call = "        result = super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)\n"
        source = replace_once(source, call,
            '        metadata = body.get("metadata")\n'
            '        if _capture.requested(metadata):\n'
            '            capture_metadata = _capture.http_metadata(metadata)\n'
            '            with _capture.request_metadata(self.engine, capture_metadata):\n'
            '                result = super().run(body, chat, emit, prepared=prepared, cancelled=cancelled)\n'
            '        else:\n' + "    " + call)
    elif name == "engine":
        source = replace_once(source, "        self.w = w\n", "        self.w = w\n        _capture.attach(self, model_dir)\n")
        source = replace_once(source, "constraint=None, *, vision=None, background: bool = False) -> dict[str, Any]:",
                              "constraint=None, *, vision=None, background: bool = False, metadata=None) -> dict[str, Any]:")
        source = replace_once(source, 'background=background, glm={"code": code, "spec": spec})',
                              'background=background, glm={"code": code, "spec": spec,\n                                              "capture": _capture.capture_request(self, prompt, metadata, vision)})')
    elif name == "multi":
        source = replace_once(source, "        self._emit(ADMIT, payload)\n",
                              '        payload += _capture.wire_extension(info.get("capture"))\n        self._emit(ADMIT, payload)\n')
        source = replace_once(source, "vrows, vkeys, vpos, _ = unpack_keys(p, i + 2 + n)",
                              "vrows, vkeys, vpos, end = unpack_keys(p, i + 2 + n)")
        source = replace_once(source, "        return dict(sid=sid, slot=slot, base=base, size=size,",
                              "        return _capture.parse_admit_capture(dict(sid=sid, slot=slot, base=base, size=size,")
        source = replace_once(source, "positions=positions, images=images, keyed=keyed)",
                              "positions=positions, images=images, keyed=keyed), p, end)")
        source = replace_once(source, "        self.lanes[lane.sid] = lane\n",
                              "        self.lanes[lane.sid] = lane\n        _capture.admitted(self, a, s)\n")
        source = replace_once(source, "        prev = e.use(lane.st)\n",
                              "        _capture.select_pieces(self, [(lane, stop)])\n        prev = e.use(lane.st)\n")
        source = replace_once(source,
                              "                    if not complete:\n                        return None\n            if stop < n:\n",
                              "                    if not complete:\n                        return None\n"
                              "            if not replay:\n                _capture.completed(self.w, e.pbuf, stop - start)\n"
                              "            if stop < n:\n")
        source = replace_once(source, "        finally:\n            e.use(prev)\n",
                              "        except BaseException:\n"
                              "            _capture.failed(self.w, e.pbuf, 'native prefill failed')\n            raise\n"
                              "        finally:\n            e.use(prev)\n")
        source = replace_once(source, "            e.use(prev)\n            s.prefill_s",
                              "            e.use(prev)\n            _capture.capture_call(setattr, e.pbuf, '_capture_pieces', [], force=True)\n            s.prefill_s")
        group_start = source.index("        if layers is None:\n            heads = multi_prefill.prefill_pieces(e, pieces)")
        group_stop = source.index("        parts, who = [], []\n", group_start)
        native_group = source[group_start:group_stop]
        guarded_group = "        _capture.select_pieces(self, group)\n        try:\n"
        for line in native_group.splitlines(True):
            guarded_group += "    " + line
        guarded_group += ("        except BaseException:\n"
                          "            _capture.failed(self.w, e.pbuf, 'native grouped prefill failed')\n"
                          "            _capture.capture_call(setattr, e.pbuf, '_capture_pieces', [], force=True)\n            raise\n"
                          "        capture_rows = 0\n        for piece in pieces:\n"
                          "            capture_rows += piece.rows\n"
                          "        _capture.completed(self.w, e.pbuf, capture_rows)\n")
        source = replace_once(source, native_group, guarded_group)
        source = replace_once(source, "        parts, who = [], []\n",
                              "        _capture.capture_call(setattr, e.pbuf, '_capture_pieces', [], force=True)\n        parts, who = [], []\n")
        source = replace_once(source, "        lane = self.lanes.pop(sid)\n",
                              "        _capture.finished(self, sid)\n        lane = self.lanes.pop(sid)\n")
    elif name == "forward":
        source = replace_once(source, '        return partials(w, b, R, fill, "moe: all-gather", site=(layer.index, "f"))',
                              '        result = partials(w, b, R, fill, "moe: all-gather", site=(layer.index, "f"))\n'
                              '        if layer is w.layers[-1]:\n            _capture.terminal(w, b, R)\n        return result')
        source = replace_once(source,
            "    if head:\n        return mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])\n    return None\n",
            "    if head:\n        result = mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])\n"
            "        _capture.completed(w, b, R)\n        return result\n"
            "    return None\n")
        source = replace_once(source,
            "    if not head:                         # the normed rows (MTP, last_hidden) without the head: None\n"
            "        return None\n"
            "    if b.prefill:                        # the head reads the last row only (fnormed keeps every row for the MTP)\n"
            "        return mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])\n"
            "    return mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])\n",
            "    if not head:\n        return None\n"
            "    if b.prefill:\n"
            "        result = mm(b, b.fnormed[R - 1:R], w.head, b.fxs[R - 1:R], b.logits[:1])\n"
            "    else:\n        result = mm(b, b.fnormed[:R], w.head, b.fxs[:R], b.logits[:R])\n"
            "    _capture.completed(w, b, R)\n    return result\n")
    from .overlay import extend
    return extend(name, source, replace_once)


def verify_source_directory(package_root):
    from glm_rsi_dream.manifest import MANIFEST

    pins = MANIFEST["engine"]
    for name, expected in HANDLER_HASHES.items():
        if pins["source_files"].get(name) != expected:
            raise ValueError("capture startup handler pin mismatch: " + name)
    for name, expected in pins["source_files"].items():
        path = package_root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("capture startup source mismatch: " + name)


class OverlayLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        code = self.get_code(module.__name__)
        # Classes are inspected through module.__file__; functions use their
        # code filename. Both must identify the exact compiled source.
        module.__file__ = code.co_filename
        exec(code, module.__dict__)

    def get_source(self, fullname):
        # After compilation, inspect must recover the same source even when
        # linecache is cleared. The installed engine file stays unchanged.
        if hasattr(self, "_compiled_source"):
            return self._compiled_source
        return super().get_source(fullname)

    def get_code(self, fullname):
        source = self.get_data(self.path).decode()
        name = OVERLAY_MODULES[fullname]
        from tensorfold_dream.capture_native import capture_call

        def overlay_code():
            transformed = transform(name, source)
            digest = hashlib.sha256(transformed.encode()).hexdigest()
            filename = self.path + ".capture-" + digest
            code = compile(transformed, filename, "exec")
            # Triton inspects kernels while exec_module executes decorators.
            # Publish before execution. A None mtime prevents checkcache from
            # replacing these lines with the different source still on disk.
            self._compiled_source = transformed
            linecache.cache[filename] = (
                len(transformed.encode()), None,
                transformed.splitlines(keepends=True), filename,
            )
            return code

        code = capture_call(overlay_code, startup=True)
        if code is None:
            self._compiled_source = source
            return compile(source, self.path, "exec")
        return code


class OverlayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in OVERLAY_MODULES:
            return None
        from tensorfold_dream.capture_native import (
            PROCESS_CAPTURE,
            _raise_capture,
            capture_call,
        )

        if PROCESS_CAPTURE["disabled"]:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            capture_call(_raise_capture, ValueError("capture overlay requires a source-backed TensorFold module"),
                         startup=True)
            return spec
        spec.loader = OverlayLoader(fullname, spec.origin)
        return spec


def _install():
    if os.environ.get("TF_DREAM_CAPTURE") != "1" and os.environ.get("TF_DREAM_EVAL") != "1":
        return
    for finder in sys.meta_path:
        if isinstance(finder, OverlayFinder):
            return
    for fullname in OVERLAY_MODULES:
        if fullname in sys.modules:
            raise RuntimeError("capture bootstrap must run before TensorFold imports")
    spec = importlib.machinery.PathFinder.find_spec("tensorfold")
    if spec is None or not spec.submodule_search_locations or len(spec.submodule_search_locations) != 1:
        raise ImportError("capture bootstrap requires one installed TensorFold source directory")
    verify_source_directory(Path(spec.submodule_search_locations[0]))
    # Load the capture implementation before installing any overlay. Import or
    # dependency failure therefore leaves native module loading intact.
    importlib.import_module("tensorfold_dream.capture_native")
    importlib.import_module("tensorfold_dream.eval_native")

    sys.meta_path.insert(0, OverlayFinder())


_STARTUP_NOTICE = False


def install():
    global _STARTUP_NOTICE
    if (os.environ.get("TF_DREAM_CAPTURE") != "1" and os.environ.get("TF_DREAM_EVAL") != "1") or _STARTUP_NOTICE:
        return
    try:
        _install()
    except Exception as exc:
        if not _STARTUP_NOTICE:
            _STARTUP_NOTICE = True
            try:
                print("[dream-capture] startup: " + str(exc) + "; capture/eval disabled for process", file=sys.stderr, flush=True)
            except Exception:
                pass
