"""Small source-pinned hooks for a native middle-layer FFN experiment."""

from typing import Any


def extend(name: Any, source: Any, replace: Any) -> Any:
    if name != "forward":
        return source
    source = replace(source,
        '            with prof.timed("moe: combine"):\n',
        '            from glm_rsi_dream.neurons.native import shared\n'
        '            shared(layer, w, b, lo, hi)\n'
        '            with prof.timed("moe: combine"):\n')
    source = replace(source,
        '    g = _ffn(layer, w, b, R)\n'
        '    glue.hc_post(x, x, g, b.post[:R], b.comb[:R])\n',
        '    from glm_rsi_dream.neurons.native import before_ffn, after_ffn\n'
        '    before_ffn(layer, w, b, R)\n'
        '    g = _ffn(layer, w, b, R)\n'
        '    glue.hc_post(x, x, g, b.post[:R], b.comb[:R])\n'
        '    after_ffn(layer, w, b, R, g)\n')
    return source
