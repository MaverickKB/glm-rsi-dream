"""Pinned stock MoE CPU doubles shared by tests and the offline parity tool."""

import ast
import gzip
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from dream_loader import bootstrap

SOURCES = json.loads(gzip.decompress(Path(__file__).with_name("upstream_sources.json.gz").read_bytes()))
FORWARD = bootstrap.PREFIX + "forward"


def stock_moe(shared_function, values, layer, weights, buffers, count, *, chunked):
    source = bootstrap.transform(FORWARD, SOURCES[bootstrap.MODULES[FORWARD]])
    tree = ast.parse(source)
    function = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "moe_block":
            function = node
            break
    assert function is not None
    # Execute the whole stock function. Its CUDA imports resolve to explicit
    # CPU doubles; the source's branch and call order remain intact.
    calls = []

    def front(layer, w, b, lo, hi):
        b.sy[lo:hi].copy_(values[lo:hi])

    def partials(w, b, rows, fill, label, site):
        if chunked:
            middle = rows // 2
            fill(0, middle)
            fill(middle, rows)
        else:
            fill(0, rows)
        calls.append(site)
        return b.part[:rows].clone()

    def combine_shared(experts, shared, coefficients, out):
        out.copy_(shared + (experts[:, :2] * coefficients.unsqueeze(-1)).sum(1))

    def combine(experts, coefficients, out):
        routed = (experts[:, :-1] * coefficients.unsqueeze(-1)).sum(1)
        out.copy_(routed + experts[:, -1])

    namespace = {"__name__": "cpu_stock_forward", "__package__": "cpu_stock_glm", "LayerW": object, "Weights": object,
                 "Buffers": object, "torch": torch, "_dream": SimpleNamespace(shared=shared_function),
                 "prof": SimpleNamespace(timed=lambda _: nullcontext()),
                 "glue": SimpleNamespace(router=lambda *args: None, select=lambda *args: None,
                                         combine_shared=combine_shared, combine=combine),
                 "grouped": SimpleNamespace(route=lambda *args: None),
                 "exl3_mm": SimpleNamespace(routed=lambda *args: None),
                 "l2pf": SimpleNamespace(ACTIVE=None), "shared_front": front,
                 "undone": lambda stop, done: [(0, stop)] if done is None else [],
                 "partials": partials}
    code = ast.Module(body=[function], type_ignores=[])
    package = types.ModuleType("cpu_stock_glm")
    package.__path__ = []
    package.exl3_mm = namespace["exl3_mm"]
    with patch.dict(sys.modules, {"cpu_stock_glm": package}):
        exec(compile(code, "cpu-stock-moe", "exec"), namespace)
        output = namespace["moe_block"](layer, weights, buffers, count)
    assert calls == [(layer.index, "f")]
    return output
