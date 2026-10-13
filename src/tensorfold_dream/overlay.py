# SPDX-License-Identifier: Apache-2.0
"""Compose request-owned attention memory with the pinned capture loader."""


from typing import Any


def extend(name: Any, source: Any, replace: Any) -> Any:
    source = replace(source, "from __future__ import annotations\n",
                     "from __future__ import annotations\nfrom tensorfold_dream import eval_native as _eval\nfrom tensorfold_dream import request_memory as _memory\n")
    if name == "glm_app":
        source = replace(source, "if _capture.requested(metadata):",
                         "if _capture.requested(metadata) or _eval.requested(metadata):")
        source = replace(source, "capture_metadata = _capture.http_metadata(metadata)",
                         "capture_metadata = _eval.http_metadata(metadata)")
    elif name == "cuda_http":
        source = replace(source, "                self._json(200, health.of(app).snapshot(app))\n",
                         "                body = health.of(app).snapshot(app)\n"
                         "                self._json(200 if body.get('ok') else 503, body)\n")
    elif name == "engine":
        source = replace(source, "        _capture.attach(self, model_dir)\n",
                         "        _capture.attach(self, model_dir)\n        _eval.attach(self, model_dir)\n")
        source = replace(source, '"capture": _capture.capture_request(self, prompt, metadata, vision)})',
                         '"capture": _capture.capture_request(self, prompt, metadata, vision),\n'
                         '                                              "eval": _eval.request(self, metadata)})')
    elif name == "multi":
        source = replace(source, "        self._check()\n        g = self.g\n",
                         "        self._check()\n        _memory.prepare_request(self, s)\n        g = self.g\n")
        source = replace(source, "        need = self._need(len(s.prompt))\n",
                         "        try:\n            need = _memory.extent_need(self, s)\n"
                         "        except Exception:\n            return True  # admit reports a preflight error to this request\n")
        if source.count("self._need(len(prompt))") != 3:
            raise ValueError("three native prompt extent sites required")
        source = source.replace("self._need(len(prompt))", "_memory.extent_need(self, s)")
        source = replace(source, "        self.lanes[sid].key = key\n",
                         "        self.lanes[sid].key = key\n"
                         "        if info.get('eval') is not None:\n"
                         "            try:\n                self._flush()\n"
                         "            except Exception as error:\n                self.broken = error\n                raise\n"
                         "            _memory.complete_admission(self, sid)\n")
        source = replace(source, "                self._admitted(a, s, feed, constraint)\n",
                         "                if a.get('eval') is not None:\n"
                         "                    self._eval_admit = a['sid']\n"
                         "                self._admitted(a, s, feed, constraint)\n"
                         "                if a.get('eval') is not None:\n"
                         "                    _memory.complete_admission(self, a['sid'])\n"
                         "                    self._eval_admit = None\n")
        source = replace(source,
                         "            self.apply(unseal(self.g._share(None), self.received, self.rank))\n",
                         "            try:\n"
                         "                self.apply(unseal(self.g._share(None), self.received, self.rank))\n"
                         "            except Exception as error:\n"
                         "                self.broken = error\n"
                         "                _memory.unblock_peer(self)\n")
        source = replace(source, '        return {"streams":',
                         '        return {"ok": self.broken is None, "rank_desynchronized": self.broken is not None, "streams":')
        source = replace(source, '        payload += _capture.wire_extension(info.get("capture"))\n',
                         '        payload += _eval.wire_extension(info.get("capture"), info.get("eval"))\n')
        source = replace(source, "return _capture.parse_admit_capture(dict(", "return _eval.parse_admit(dict(")
        source = replace(source, "        hit = self._resume(key) if draft and key is not None else None\n",
                         '        hit = self._resume(key) if draft and key is not None and info.get("eval") is None else None\n')
        # Native head admission runs before _flush; eval.admitted must be local.
        source = replace(source, "        _capture.admitted(self, a, s)\n",
                         "        _capture.admitted(self, a, s)\n        _eval.admitted(self, a)\n")
        source = replace(source, "        prev = self.e.use(lane.st)\n",
                         "        if _eval.tagged(self.w, lane.st):\n            return\n        prev = self.e.use(lane.st)\n")
        source = replace(source, "        _capture.select_pieces(self, [(lane, stop)])\n",
                         "        _capture.select_pieces(self, [(lane, stop)])\n        _eval.select_prefill(self, [(lane, stop)])\n")
        source = replace(source, "        _capture.select_pieces(self, group)\n",
                         "        _capture.select_pieces(self, group)\n        _eval.select_prefill(self, group)\n")
        source = replace(source, "        lane.st, lane.slot = st, slot\n",
                         "        if _eval.tagged(self.w, lane.st):\n            st._eval_sid = lane.sid\n        lane.st, lane.slot = st, slot\n")
        source = replace(source, "        _capture.finished(self, sid)\n",
                         "        _capture.finished(self, sid)\n        _eval.finished(self, sid)\n")
    elif name == "decode":
        source = replace(source, "        R = stage(self.w, self.st, self.buf, tokens)\n",
                         "        self.buf._eval_pieces = [(self.st, 0, len(tokens))]\n"
                         "        R = stage(self.w, self.st, self.buf, tokens)\n")
        source = replace(source, '        if g is not None:\n            self.replays[kind] += 1\n            g.replay()\n            return self.buf.logits[:R]\n',
                         '        if _eval.has_main(self.w, self.buf):\n            g = None\n'
                         '        if g is not None:\n            self.replays[kind] += 1\n            g.replay()\n            return self.buf.logits[:R]\n')
    elif name == "verify":
        source = replace(source, "        g = self.graphs.get(R)\n",
                         "        _eval.select_decode(w, b, segments)\n"
                         "        g = None if _eval.has_main(w, b) else self.graphs.get(R)\n")
    from .attention_overlay import extend_attention
    source = extend_attention(name, source, replace)
    from glm_rsi_dream.neurons.overlay import extend as extend_neurons
    return extend_neurons(name, source, replace)
