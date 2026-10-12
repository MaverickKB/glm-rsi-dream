"""Freeze task identity and apply explicit, configurable scope routing."""
import re

from .files import digest, read_json, write_json
from .safety import SafetyError


def matches_group(text, terms):
    for term in terms:
        if re.search(r'(?<!\w)' + re.escape(term.casefold()) + r'(?!\w)', text.casefold()):
            return True
    return False


def admitted(question, routes):
    for route in routes:
        matched = True
        for group in route['all']:
            if not matches_group(question, group):
                matched = False
                break
        if matched:
            return True
    return False


def validate(lesson, evaluation):
    if not lesson['facts'] or not lesson['routes']:
        raise ValueError('Supply facts and bounded routes')
    facts = {}
    for fact in lesson['facts']:
        if fact['id'] in facts or not fact['statement'].strip():
            raise ValueError('Fact IDs must be unique and statements nonempty')
        facts[fact['id']] = fact
    ids = set()
    questions = set()
    for split, rows in [('train', lesson['training']), ('dev', lesson['development']),
                        ('heldout', evaluation['heldout'])]:
        if not rows:
            raise ValueError('Task split is empty: ' + split)
        covered = set()
        for row in rows:
            if row['id'] in ids or row['prompt'].casefold() in questions:
                raise ValueError('Splits overlap by ID or question')
            if row['fact_id'] not in facts or not row['aliases']:
                raise ValueError('Task must identify a fact and answer aliases')
            if not admitted(row['prompt'], lesson['routes']):
                raise ValueError('Fact task bypasses its declared scope: ' + row['id'])
            ids.add(row['id'])
            questions.add(row['prompt'].casefold())
            covered.add(row['fact_id'])
        if covered != set(facts):
            raise ValueError('Every split must cover all facts')
    if not evaluation['protected'] or not evaluation['near_miss']:
        raise ValueError('Protected and near-miss controls are required')
    for prompt in evaluation['protected'] + evaluation['near_miss']:
        if admitted(prompt, lesson['routes']):
            raise ValueError('Control unexpectedly enters memory scope')
    return lesson, evaluation


def freeze(cluster, lesson_path, eval_path):
    if cluster.run_dir.exists():
        raise SafetyError('Use a new private run directory for each loop')
    lesson, evaluation = validate(read_json(lesson_path), read_json(eval_path))
    cluster.run_dir.mkdir(parents=True, mode=0o700)
    write_json(cluster.run_dir / 'lesson.json', lesson)
    write_json(cluster.run_dir / 'eval.json', evaluation)
    value = {'lesson_sha256': digest(lesson), 'eval_sha256': digest(evaluation),
             'config_sha256': digest(cluster.config)}
    write_json(cluster.run_dir / 'freeze.json', value)
    return lesson, evaluation


def verify(cluster):
    frozen = read_json(cluster.run_dir / 'freeze.json')
    for name in ('lesson', 'eval'):
        if digest(read_json(cluster.run_dir / (name + '.json'))) != frozen[name + '_sha256']:
            raise SafetyError('Frozen dataset changed: ' + name)
    if digest(cluster.config) != frozen['config_sha256']:
        raise SafetyError('Frozen configuration changed')
    return frozen


def close_fitting(cluster, selected):
    marker = cluster.run_dir / 'evaluation-started.json'
    if marker.exists():
        raise SafetyError('Holdout already opened; fitting stays closed')
    verify(cluster)
    write_json(marker, {'selected': selected, 'fitting_closed': True})
