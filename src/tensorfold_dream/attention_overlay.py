# SPDX-License-Identifier: Apache-2.0
"""Small explicit extension of the release's pinned source overlay."""

def extend_attention(name,source,replace):
    if name=='engine':
        source=replace(source,'            self.multi.model_dir = self.model_dir\n',
            '            self.multi.model_dir = self.model_dir\n'
            '            from tensorfold_dream.attention_native import initialize\n'
            '            initialize(self)\n')
    elif name=='forward':
        source=replace(source,'    l2pf.site(layer, "o")\n',
            '    from tensorfold_dream.attention_native import capture_read\n'
            '    capture_read(layer, w, b, qa, lc, pos_dev, lo, R)\n'
            '    l2pf.site(layer, "o")\n')
        anchor='    with prof.timed("dsa: expand"):\n        return latent.expand_v(ol, a.absorb, b.vn[rs], prompt=b.prefill and latent.PROMPT_MMA).view(R, HL * c.v_dim)\n'
        source=replace(source,anchor,
            '    from tensorfold_dream.attention_native import capture_output\n'
            '    capture_output(layer, w, b, ol, lo, R)\n'+anchor)
    return source
