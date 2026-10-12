# SPDX-License-Identifier: Apache-2.0
"""Request-owned hybrid attention memory. Backbone tensors remain untouched."""
import hashlib
import json
from pathlib import Path

import torch

KEY='tensorfold_dream.eval'


def initialize(engine):
    state=engine.w.meta.get(KEY)
    if state is None:
        raise RuntimeError('Attention manifest unavailable on one rank')
    from tensorfold.families.glm5_next.cuda import decode
    prefix=json.loads((state['root']/'prefix.json').read_text())
    ids=prefix['teacher_ids']
    if len(ids)<96:
        raise ValueError("Teacher prefix must cover the 96 memory slots")
    if engine.e.grid and len(ids)%engine.e.grid:
        raise ValueError('Numerical prefix must align with the native prompt grid')
    decode.prefill(engine.e, ids, None, mtp=False, sample=False)
    snap=decode.take_snapshot(engine.e, ids, None, mtp=False)
    decode.save_rows(engine.e,snap)
    snap.ids=[0]*len(ids)
    state['snapshot']=snap
    state['prefix']=prefix
    value={'schema_version':3,'mechanism':'attention_state_v1','length':len(ids),
           'rec':snap.rec.cpu(),'conv':snap.conv.cpu(),'rows':[], 'tail':[]}
    for row in snap.rows:
        value['rows'].append(row.cpu())
    for slots,rows in snap.tail:
        value['tail'].append((slots.cpu(),rows.cpu()))
    torch.save(value,Path('/capture')/('fullstate-rank'+str(engine.w.rank)+'.pt'))
    engine.e.reset()


def install(decoder, record):
    """Admission is local on both ranks; initial state was agreed at startup."""
    if record['spec']['arm']!='main' or not record['policy'].get('use_memory'):
        return
    from tensorfold.families.glm5_next.cuda import decode
    state=decoder.w.meta[KEY]
    lane=decoder.lanes[record['sid']]
    prefix=state['prefix']
    ordinary=list(lane.s.prompt)
    normal=prefix['normal_ids']
    if ordinary[:len(normal)]!=normal:
        raise ValueError('Ordinary system prefix differs from frozen native template')
    suffix=ordinary[len(normal):]
    snap=state['snapshot']
    selected=record['policy'].get('selected')
    if selected:
        path=(state['root']/selected['path']).resolve()
        if not path.is_relative_to(state['root'].resolve()):
            raise ValueError('Memory path escaped manifest')
        raw=path.read_bytes()
        actual=hashlib.sha256(raw).hexdigest()
        if actual!=selected['sha256']:
            raise ValueError('Memory artifact identity mismatch')
        if state.get('compressed_sha')!=actual:
            value=torch.load(path,map_location=decoder.w.device,weights_only=True)
            if value['schema_version']!=3 or value['mechanism']!='attention_state_v1' or value['rank']!=decoder.w.rank or not value['training_only'] or not value['dense_only']:
                raise ValueError('Memory tensor owner mismatch')
            n=value['length']
            from tensorfold.families.glm5_next.cuda import kda
            # This path restores a numerical state into multi-prefill directly.
            # Native chunked KDA uses32-row sub-chunks; kept-prompt grid64 is
            # an engine cache-retention policy, not this snapshot's geometry.
            alignment=32 if kda.CHUNKED else 1
            aligned=n%alignment==0
            if n!=96 or n>len(snap.ids) or not aligned or len(value['rows'])!=len(snap.rows):
                raise ValueError('Compressed geometry mismatch')
            compressed=decode.Snapshot([0]*n,snap.rec,snap.conv,None,-1,-1,tail=value['tail'])
            compressed.rows=value['rows']
            state['compressed']=compressed
            state['compressed_sha']=actual
        snap=state['compressed']
        record['artifact_sha256']=actual
    n=len(snap.ids)
    if selected and n+len(suffix)+lane.s.count>decoder.w.cfg.dense_limit:
        raise ValueError('Compressed cartridge admits dense contexts only')
    if n+len(suffix)+lane.s.count>lane.st.capacity:
        raise ValueError('Request extent too small for memory')
    lane.s.prompt=[0]*n+suffix
    lane.stops=[]
    lane.point=None
    previous=decoder.e.use(lane.st)
    try:
        decode.load_rows(decoder.e,snap)
        decode.restore(decoder.e,snap)
    finally:
        decoder.e.use(previous)
    lane.st._eval_sid=record['sid']
    record['main_calls']=1
    record['memory_activated']=True
    record['memory_length']=n
    record['ordinary_ids']=ordinary
    record['suffix_start']=n
    record['attention_capture']=record['policy'].get('attention_capture',False)


def capture_read(layer, w, b, qa, lc, pos_dev, lo, count):
    from .eval_native import spec_for
    for st,offset,size in getattr(b,'_eval_pieces',()):
        record=spec_for(w,st)
        if record is None or not record.get('attention_capture'):
            continue
        if offset!=lo or size!=count:
            continue
        n=record['suffix_start']
        position=int(pos_dev.item())
        if position<n or position+count>w.cfg.dense_limit:
            raise ValueError('Capture requires short native memory trajectory')
        # Packed FP8 rows are decoded by the actual native format below.
        rows=lc[:position+count].clone()
        record['captures'].setdefault(layer,[]).append({'queries':qa.detach().float().cpu(),
             'cache':rows.cpu(),'position':position,'prefix_length':n})


def capture_output(layer,w,b,ol,lo,count):
    from .eval_native import spec_for
    for st,offset,size in getattr(b,'_eval_pieces',()):
        record=spec_for(w,st)
        if record is None or not record.get('attention_capture') or offset!=lo or size!=count:
            continue
        packet=record['captures'].get(layer)
        if packet:
            packet[-1]['native_output']=ol.detach().float().cpu()
