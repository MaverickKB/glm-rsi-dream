#!/usr/bin/env python3
"""offline_gate.py -- REBUILT 2026-10-17 (box, CPU only).

REBUILT: the original offline_gate.py (sha256 112afdd20ff9...) was lost in the 14:39-14:48 ET box re-lay. This file is a
re-implementation from the written spec. It is NOT the original and its sha256 is NOT 112afdd20ff9.

What it does (PREDICTION, calibration-format caches):
  For each gate, splice per row: the arm's L output (pocket on) where the gate fires, B output (pocket off) where it
  does not. Then score the spliced arm with the FROZEN P1 scorers, run as-is (score_facts.py, score_controls74.py),
  and the same line logic as p1_runner.report (strict F6 leaks, pocket values on the 22, S-vs-B diffs on 96 rows).
  Dev/rewording grades are spliced from the arm's existing frozen-judge grade files (no judge rerun).

Gates (all parameters come from the env file named by OG_ENV, default eng/p1b/private/og.env):
  K  key match: fact keys = subject entity of each cartridge F record (leading proper noun, possessive stripped;
     a record led by a pronoun/determiner inherits the previous record's key). Fires if the question names a key.
  P  G1 probe (OG_PROBE, threshold read from the probe file, never recalibrated) on cached layer-35 answer-start
     activations (OG_CACHE_BUNDLE, OG_CACHE_MANIFEST), matched to rows by exact prompt text. Fires if score > thresh.
  E  owner-entity check: owner = most frequent subject entity across the records of OG_POCKETS (config, never
     hardcoded). Fires if the question names the owner.
  A  mechanism 3, attribute-key check (stack on P+E): attribute keys per F fact, derived from the cartridge F train
     records only (see attr_keys). Fires if the question shares an attribute key with some F fact.
Combinations are ANDs of single gates, named with '+'.
  offline_gate.py run --arm p1|p1b1 [--gates K,P,E,P+E,...] [--out file.json]
"""
import argparse
import collections
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

VERSION = 'offline_gate REBUILT r1'
norm = lambda s: re.sub(r'[^a-z0-9@]+', ' ', (s or '').lower()).strip()          # p1_common.norm, verbatim
ABSTAIN = re.compile(r"\b(?:don't|do not|doesn't|does not|didn't|isn't|not (?:specified|stated|mentioned|given|"
                     r"included|provided)|no (?:information|record|details?|mention)|unknown|"
                     r"can't (?:say|tell|determine)|cannot (?:say|tell|determine))\b", re.I)  # thin_score2 verbatim
COMPARISON = ('controls74', 'nearmiss10', 'otherperson12')
PRONOUNS = {'both', 'they', 'he', 'she', 'it', 'its', 'their', 'his', 'her', 'the', 'a', 'an', 'all', 'each', 'fictional'}
STOP = set('''a an the of to in on at by for from with about as into over under and or but is are was were be been being
do does did done have has had having what which who whom whose when where why how that this these those it its his her
their there here up down out off one any some all each both such own same so than too very can will just should would
could may might must shall according you your yours me my mine we our us they them i please tell give describe kind
item list listed thing'''.split())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_env(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            env[k.strip()] = os.path.expandvars(v.strip().strip('"'))
    for k, v in list(env.items()):                                        # second pass resolves ${OTHER} refs
        env[k] = re.sub(r'\$\{(\w+)\}', lambda m: env.get(m.group(1), os.environ.get(m.group(1), '')), v)
    return env


def words(text):
    return re.findall(r"[A-Za-z0-9@_\-]+(?:'s)?", text or '')


def stem(w):
    w = w.lower()
    w = w[:-2] if w.endswith("'s") else w
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
        w = w[:-1]
    return w


def subject_entity(sentence):
    toks = words(sentence)
    if not toks:
        return None
    t = toks[0][:-2] if toks[0].endswith("'s") else toks[0]
    if t.lower() in PRONOUNS or not t[:1].isupper():
        return None
    return t


def fact_keys(records):
    keys, last = [], None
    for s in records:
        k = subject_entity(s) or last
        if k:
            keys.append(k); last = k
    return sorted(set(keys))


def owner_from_pockets(pockets):
    c = collections.Counter()
    for p in pockets:
        last = None
        for s in re.split(r'(?<=[.!?])\s+', p.get('facts', '')):
            k = subject_entity(s) or last
            if k:
                c[k] += 1; last = k
    return c.most_common(1)[0][0], dict(c)


def names_in(text, names):
    toks = {t[:-2] if t.endswith("'s") else t for t in words(text)}
    return sorted(n for n in names if n in toks)


def attr_keys(train, entity_names, min_frac):
    """Mechanism 3 keys, from the cartridge F train records only (prompt text per fact id).
    key(f) = content stems present in >= min_frac of f's record prompts, minus stopwords, minus entity names
    (capitalised non-initial tokens in any record), minus the requested-value stems of f, minus the 'carrier' stems
    that reach min_frac in EVERY fact of the same pocket group (the object, not the attribute)."""
    by = collections.defaultdict(list)
    for r in train:
        by[r['fact']].append(r)
    ents = {e.lower() for e in entity_names}
    for r in train:
        for i, t in enumerate(words(r['prompt'])):
            if i and len(t) > 1 and t[:1].isupper() and t[1:2].islower():      # Titlecase names, not X / AI
                ents.add(stem(t))
    keys = {}
    for f, rs in by.items():
        vals = set.intersection(*({stem(w) for w in words(r.get('requested_value', ''))} for r in rs))
        cnt = collections.Counter()
        for r in rs:
            cnt.update({stem(w) for w in words(r['prompt']) if w.lower() not in STOP})
        keys[f] = {w for w, n in cnt.items() if n >= math.ceil(min_frac * len(rs))
                   and w not in STOP and w not in ents and w not in vals and len(w) >= 1}
    groups = collections.defaultdict(list)
    for f in keys:
        groups[f.split('-')[0]].append(f)
    for g, fs in groups.items():
        if len(fs) > 1:
            carrier = set.intersection(*(keys[f] for f in fs))
            for f in fs:
                keys[f] -= carrier
    return {f: sorted(v) for f, v in sorted(keys.items())}


def attr_match(prompt, keys):
    q = {stem(w) for w in words(prompt)}
    return sorted(f for f, ks in keys.items() if q & set(ks))


def probe_scores(env, prompts):
    import torch
    b = torch.load(env['OG_CACHE_BUNDLE'], map_location='cpu', weights_only=False)
    p = torch.load(env['OG_PROBE'], map_location='cpu', weights_only=False)
    man = {json.loads(l)['id']: json.loads(l) for l in open(env['OG_CACHE_MANIFEST']) if l.strip()}
    X = b['x'].float()
    s = ((X - p['mu']) / p['sd']) @ p['w']
    by_prompt = collections.defaultdict(list)
    for i, rid in enumerate(b['ids']):
        by_prompt[man[rid]['prompt']].append((rid, float(s[i])))
    thresh = float(p['thresh'])
    want = env.get('OG_PROBE_THRESH')
    if want and abs(float(want) - thresh) > 1e-3:
        raise SystemExit('probe threshold in file %r != OG_PROBE_THRESH %r (no recalibration allowed)' % (thresh, want))
    out = {}
    for q in prompts:
        hits = by_prompt.get(q)
        if not hits:
            raise SystemExit('no cached activation for prompt: %r' % q)
        out[q] = {'cache_ids': [h[0] for h in hits], 'score': hits[0][1], 'fire': hits[0][1] > thresh}
        if len({round(h[1], 3) for h in hits}) > 1:
            out[q]['note'] = 'several caches disagree; first used'
    return out, thresh


def mentions(values, text):
    return [v for v in values if (' %s ' % v) in (' %s ' % norm(text))]


def load_gen(d, arm):
    return {(r['suite'], r['id']): r for r in map(json.loads, open(Path(d) / 'gen' / ('%s.jsonl' % arm)))}


class Ctx:
    def __init__(self, env, arm):
        self.env, self.arm = env, arm
        A = arm.upper()
        self.out = Path(env['OG_%s_OUT' % A])
        self.rows = [json.loads(l) for l in open(env['OG_ROWS'])]
        self.meta = {(r['suite'], r['id']): r for r in self.rows}
        self.L, self.B = load_gen(self.out, 'L'), load_gen(self.out, 'B')
        self.cart = json.loads(Path(env['OG_%s_CARTRIDGE' % A]).read_text())
        self.train = json.loads(Path(env['OG_%s_TRAIN' % A]).read_text())
        self.pockets = json.loads(Path(env['OG_POCKETS']).read_text())
        self.split = json.loads(Path(env['OG_FACT_SPLIT']).read_text())['split']
        F = self.split['weights']
        self.fvals = sorted({norm(v) for f in F for v in self.split['values'][f]} - {''})
        self.keys = fact_keys(self.cart['facts'])
        self.owner, self.owner_counts = owner_from_pockets(self.pockets)
        self.akeys = attr_keys(self.train, self.keys + [self.owner], float(env.get('OG_ATTR_MIN_FRAC', '0.5')))
        self.probe, self.thresh = probe_scores(env, sorted({r['prompt'] for r in self.rows}))
        self.grades = {a: {s: json.loads((self.out / 'score' / ('p1-%s-%s.json' % (a, s))).read_text())
                           for s in ('dev', 'rewordings')} for a in ('L', 'B')}
        self.inv = {a: {(x['suite'], x['id']): set(map(str, x['specifics']))
                        for x in json.loads((self.out / 'score' / ('p1-%s-invented.json' % a)).read_text())}
                    for a in ('L', 'B')}

    def single(self, g, prompt):
        if g == 'K':
            return bool(names_in(prompt, self.keys))
        if g == 'E':
            return bool(names_in(prompt, [self.owner]))
        if g == 'P':
            return self.probe[prompt]['fire']
        if g == 'A':
            return bool(attr_match(prompt, self.akeys))
        if g == 'ALL':
            return True
        if g == 'NONE':
            return False
        raise SystemExit('unknown gate %s' % g)

    def fires(self, gate, prompt):
        return all(self.single(g, prompt) for g in gate.split('+'))


def evaluate(ctx, gate, scorers, tmp):
    fire = {k: ctx.fires(gate, r['prompt']) for k, r in ctx.meta.items()}
    S = {k: (ctx.L[k] if fire[k] else ctx.B[k]) for k in ctx.L}
    sd = Path(tmp) / gate.replace('+', '_'); sd.mkdir(parents=True, exist_ok=True)
    for s in ('dev', 'rewordings'):
        rows = [x for x in ctx.grades['L'][s] if fire[(s, x['id'])]] + [x for x in ctx.grades['B'][s] if not fire[(s, x['id'])]]
        (sd / ('p1-S-%s.json' % s)).write_text(json.dumps(rows))
        for a in ('L', 'B'):
            (sd / ('p1-%s-%s.json' % (a, s))).write_text(json.dumps(ctx.grades[a][s]))
    envs = dict(os.environ, DREAM_SPLIT=ctx.env['OG_FACT_SPLIT'], DISTILL_SCORE_DIR=str(sd),
                GATE_FACT_MAP=ctx.env['OG_FACT_ITEMS'])
    fs = {a: json.loads(subprocess.check_output([sys.executable, str(scorers / 'score_facts.py'), 'p1-' + a], env=envs))
          for a in ('S', 'B')}
    F = ctx.split['weights']
    credit = {s: sorted(f for f in F if fs['S'][s]['per_fact'][f]['pass'] and not fs['B'][s]['per_fact'][f]['pass'])
              for s in ('dev', 'rewordings')}
    c74p = sd / 'p1-S-controls74.jsonl'
    with open(c74p, 'w') as f:
        for k, r in S.items():
            if k[0] == 'controls74':
                f.write(json.dumps({'prompt': r['prompt'], 'text': r['text'], 'finish': r['finish'], 'arm': 'S'}) + '\n')
    r = subprocess.run([sys.executable, str(scorers / 'score_controls74.py'), '--controls', ctx.env['OG_CONTROLS74'],
                        '--gen', str(c74p), '--arm', 'S'], capture_output=True, text=True)
    c74base = json.loads(r.stdout)
    diff = lambda ks: [k for k in ks if S[k]['text'] != ctx.B[k]['text'] or S[k]['finish'] != ctx.B[k]['finish']]
    c14 = [k for k in S if k[0] == 'controls14']
    c14_exact = sum(S[k]['text'] == ctx.B[k]['text'] and S[k]['finish'] == ctx.B[k]['finish'] == 'stop' for k in c14)
    c74 = [k for k in S if k[0] == 'controls74']
    cmp_ = [k for k in S if k[0] in COMPARISON]
    leak = []
    for k in S:
        if k[0] in ('nearmiss10', 'otherperson12'):
            asked = ' %s ' % norm(S[k]['prompt'])
            named = mentions(ctx.fvals, S[k]['text'])
            leak.append({'row': '%s/%s' % k, 'suite': k[0], 'idk_group': ctx.meta[k].get('idk_group'), 'gate_fired': fire[k],
                         'values_in_S': named, 'values_not_in_question': [v for v in named if (' %s ' % v) not in asked]})
    nm = [x for x in leak if x['suite'] == 'nearmiss10']
    new_inv = sorted('%s/%s' % k for k in S if fire[k] and (ctx.inv['L'].get(k, set()) - ctx.inv['B'].get(k, set())))
    sb = diff(cmp_)
    terse_abstain = [k for k in sb if ABSTAIN.search(S[k]['text'] or '') and len(S[k]['text']) < len(ctx.B[k]['text'])]
    ctl_fail = sorted('%s/%s' % k for k in diff(c14 + c74))
    line = {
        'recall_dev_credited': '%d/6' % len(credit['dev']), 'recall_rewordings_credited': '%d/6' % len(credit['rewordings']),
        'controls14_S_eq_B': '%d/%d' % (c14_exact, len(c14)),
        'controls74_S_eq_B': '%d/%d' % (len(c74) - len(diff(c74)), len(c74)),
        'controls74_vs_frozen_base_info': '%d/%d' % (c74base['exact'], c74base['n']),
        'other_person_with_value': '%d/5' % sum(bool(x['values_in_S']) for x in leak if x['idk_group'] == 'other_person'),
        'ken_unknown_with_value': '%d/7' % sum(bool(x['values_in_S']) for x in leak if x['idk_group'] == 'ken_unknown'),
        'strict_nearmiss_leaks_F6': '%d/10' % sum(bool(x['values_not_in_question']) for x in nm),
        'strict_nearmiss_leak_rows': [x['row'].split('/')[1] for x in nm if x['values_not_in_question']],
        'pocket_values_22_strict': '%d/22' % sum(bool(x['values_in_S']) for x in leak),
        'S_ne_B_96': '%d/%d' % (len(sb), len(cmp_)),
        'S_ne_B_96_rows': ['%s/%s' % k for k in sb],
        'S_ne_B_96_abstain_and_shorter_than_B': len(terse_abstain),
        'invented_new_vs_B': new_inv,
        'controls_failed_S_ne_B': ctl_fail,
        'fired': {s: '%d/%d' % (sum(fire[k] for k in fire if k[0] == s), sum(1 for k in fire if k[0] == s))
                  for s in ('dev', 'rewordings', 'controls14', 'controls74', 'nearmiss10', 'otherperson12')},
    }
    return line, fire, fs


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('cmd', choices=['run', 'config'])
    a.add_argument('--arm', default='p1')
    a.add_argument('--gates', default=None)
    a.add_argument('--out', default=None)
    x = a.parse_args(argv)
    envp = os.environ.get('OG_ENV', str(Path(__file__).parent / 'private' / 'og.env'))
    env = load_env(envp)
    ctx = Ctx(env, x.arm)
    cfg = {'version': VERSION, 'script_sha256': sha(__file__), 'env_file': Path(envp).name, 'env_sha256': sha(envp),
           'arm': x.arm, 'owner_source': 'pockets.json subject counts', 'owner_counts': ctx.owner_counts,
           'fact_keys_K': ctx.keys, 'attr_keys_A': ctx.akeys, 'probe_thresh': ctx.thresh,
           'inputs': {k: (sha(v)[:12] if Path(v).is_file() else 'dir') for k, v in env.items()
                      if k.startswith('OG_') and v and Path(v).exists()}}
    if x.cmd == 'config':
        print(json.dumps(cfg, indent=1)); return 0
    gates = (x.gates or env.get('OG_GATES', 'K,P,E,K+P,P+E,K+E,K+P+E,P+E+A')).split(',')
    res = {'label': 'PREDICTION (calibration-format caches; REBUILT offline gate, not the lost 112afdd20ff9)',
           'config': cfg, 'probe_rows': {'%s/%s' % k: {'score': round(ctx.probe[r['prompt']]['score'], 3),
                                                        'fire': ctx.probe[r['prompt']]['fire'],
                                                        'cache_ids': ctx.probe[r['prompt']]['cache_ids']}
                                         for k, r in ctx.meta.items()}, 'gates': {}}
    with tempfile.TemporaryDirectory() as tmp:
        for g in gates:
            line, fire, fs = evaluate(ctx, g, Path(env['OG_SCORERS']), tmp)
            res['gates'][g] = {'line': line, 'score_facts_S': fs['S']}
    out = json.dumps(res, indent=1, sort_keys=True)
    if x.out:
        Path(x.out).write_text(out + '\n')
    print(json.dumps({g: {k: v for k, v in r['line'].items() if not isinstance(v, (list, dict))} for g, r in res['gates'].items()}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
