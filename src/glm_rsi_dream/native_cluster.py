"""Attention-only windows with paired receipts and host-local recovery guards."""
import time

from .controller import Cluster
from .dataset import admitted
from .files import read_json, write_json
from .manifest import MANIFEST
from .safety import SafetyError


class AttentionCluster(Cluster):
    def preflight(self):
        pins = super().preflight()
        self.check_worker_listener()
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(self.run_dir / 'tokenizer.json'))
        normal = '[gMASK]<sop><|system|>' + self.config['system_prompt']
        lesson = read_json(self.run_dir / 'lesson.json')
        statements = []
        for fact in lesson['facts']:
            statements.append(fact['statement'])
        teacher = normal + '\nReference facts:\n' + '\n'.join(statements)
        normal_ids = tokenizer.encode(normal, add_special_tokens=False).ids
        teacher_ids = tokenizer.encode(teacher, add_special_tokens=False).ids
        newline = tokenizer.encode('\n', add_special_tokens=False).ids
        if len(newline) != 1:
            raise SafetyError('Prefix padding requires a single newline token')
        while len(teacher_ids) < MANIFEST['memory']['slots'] or len(teacher_ids) % 128:
            teacher_ids += newline
        prefix = {'normal_ids': normal_ids, 'teacher_ids': teacher_ids}
        existing = self.run_dir / 'pair/prefix.json'
        if existing.exists() and read_json(existing) != prefix:
            raise SafetyError('Frozen prefix identity changed')
        write_json(existing, prefix)
        probes = []
        for rank in range(2):
            urls = [self.config['safety']['dashboard_url']] + self.config['safety']['protected_urls']
            for url in urls:
                code = ('import urllib.request\n'
                        'with urllib.request.urlopen(' + repr(url) + ', timeout=5) as r:\n'
                        ' assert r.status==200 and r.read(65536), "protected probe failed"\n'
                        'print("ok")')
                self.transport.call(rank, 'host_probe', code=code)
                probes.append({'rank': rank, 'url': url, 'passed': True})
        write_json(self.run_dir / 'protected-preflight.json', probes)
        return pins

    def check_worker_listener(self):
        from urllib.parse import urlparse
        parsed = urlparse(self.config['api_url'])
        default_port = 443 if parsed.scheme == 'https' else 80
        port = parsed.port or default_port
        return self.transport.call(1, 'no_listener', port=port)

    def ready(self):
        super().ready()
        self.check_worker_listener()

    def arm_guards(self):
        for rank in range(2):
            owned = []
            for item in self.state['candidate_history']:
                if item['rank'] == rank:
                    owned.append(item['id'])
            spec = {'original': self.state['originals'][rank]['id'], 'candidates': owned,
                    'deadline': time.time() + self.config['safety']['window_seconds'],
                    'safety': self.config['safety'], 'api_url': self.config['api_url']}
            self.transport.call(rank, 'arm_guard', root=self.transport.root(rank, self.run_id), spec=spec)

    def _restore(self):
        super()._restore()
        try:
            self.check_worker_listener()
            probes = []
            for rank in range(2):
                for url in [self.config['safety']['dashboard_url']] + self.config['safety']['protected_urls']:
                    code = ('import urllib.request\n'
                            'with urllib.request.urlopen(' + repr(url) + ', timeout=5) as r:\n'
                            ' assert r.status==200 and r.read(65536)\nprint("ok")')
                    self.transport.call(rank, 'host_probe', code=code)
                    probes.append({'rank': rank, 'url': url, 'passed': True})
                self.transport.call(rank, 'finish_guard', root=self.transport.root(rank, self.run_id))
            write_json(self.run_dir / 'protected-restored.json', probes)
        except Exception as error:
            self.state['phase'] = 'restore_failed'
            self.state['restore_errors'].append('protected recovery: ' + str(error))
            self.save()
            raise

    def infer(self, body, policy=None):
        policy = dict(policy or {})
        tag = body.get('metadata', {}).get('dream_eval')
        if tag and policy.get('use_memory'):
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(self.run_dir / 'tokenizer.json'))
            text = tokenizer.decode(body['prompt'], skip_special_tokens=False)
            question = text.split('<|user|>')[-1].split('<|assistant|>')[0]
            lesson = read_json(self.run_dir / 'lesson.json')
            policy['use_memory'] = admitted(question, lesson['routes'])
        reply, receipts = super().infer(body, policy if tag else None)
        if tag:
            enabled = tag['arm'] == 'main' and policy.get('use_memory', False)
            selected = policy.get('selected_by_rank')
            for rank, receipt in enumerate(receipts):
                if receipt.get('memory_activated') != enabled:
                    raise SafetyError('Paired memory activation differs from request policy')
                if enabled:
                    length = MANIFEST['memory']['slots'] if selected else len(read_json(self.run_dir / 'pair/prefix.json')['teacher_ids'])
                    if receipt.get('memory_length') != length or receipt.get('cache_writes') != 1:
                        raise SafetyError('Native memory extent is unproven')
                    if selected and receipt.get('artifact_sha256') != selected[rank]['sha256']:
                        raise SafetyError('Rank artifact identity differs from selected memory')
                elif receipt.get('cache_writes') != 0 or receipt.get('offpath_delta') != 0:
                    raise SafetyError('Off-path memory effect')
        return reply, receipts
