"""Gold isolation, one-to-one scoring and sealed resumable evaluations."""

import json
from pathlib import Path

import pytest

from glm_rsi_dream.predream.extraction_eval import (
    align_candidates,
    load_gold,
    run_evaluation,
    score_alignment,
    validate_alignment,
)


def cases():
    return [
        {'name': 'gold-a', 'document_id': 'doc-0001', 'source_text': 'Jun uses drawer B2.',
         'text': 'Jun uses drawer B2.', 'prompt_evidence': 'Jun uses drawer B2.',
         'category': 'new_durable', 'gold': {'extract': True},
         'expected': {'verdict': 'accept', 'route': 'pocket', 'pocket': 'fixture'}},
        {'name': 'gold-b', 'document_id': 'doc-0002', 'source_text': 'Jun feels tired tonight.',
         'text': 'Jun feels tired tonight.', 'prompt_evidence': 'Jun feels tired tonight.',
         'category': 'personal_ephemera', 'gold': {'extract': True},
         'expected': {'verdict': 'accept', 'route': 'memory', 'pocket': None}},
    ]


def test_gold_corpus_is_large_balanced_and_source_grounded():
    path = Path(__file__).with_name('funnel_gold.json')
    gold, documents = load_gold(path)
    assert len(gold) >= 150 and len(documents) == 24
    assert len(set(row['text'] for row in gold)) == len(gold)
    counts = {}
    for row in gold:
        counts[row['category']] = counts.get(row['category'], 0) + 1
        assert row['gold']['extract'] is True
        if row['category'] == 'secret':
            assert 'TEST' in row['text'] and row['expected']['verdict'] == 'reject'
        if row['category'] == 'update':
            assert row['gold']['prior_value'] in row['text']
        if row['category'] == 'duplicate':
            assert row['gold']['duplicate_of']
    assert len(counts) == 7 and set(counts.values()) == {24}


def test_source_funnel_labels_preserve_catalog_and_blind_novelty_separately():
    gold = json.loads(Path(__file__).with_name('funnel_gold.json').read_text())
    sources = json.loads(Path(__file__).with_name('funnel_sources_gold.json').read_text())
    by_id = {}
    for row in gold:
        by_id[row['name']] = row
    count = 0
    for document in sources:
        for expected in document['expected']:
            row = by_id[expected['id']]
            assert expected['text'] == row['text']
            assert document['source'] == row['source_text']
            assert expected['catalog_novelty'] == row['novelty']
            if row['expected']['verdict'] == 'reject':
                assert expected['route'] == 'reject'
            else:
                assert expected['route'] == row['expected']['route']
            blind_novelty = 'known' if row['category'] == 'known' else 'new'
            assert expected['novelty'] == blind_novelty
            count += 1
    assert count == len(gold)


def test_precision_recall_counts_extra_and_missing_claims():
    alignment = {'matches': [{'candidate_id': 'a', 'gold_id': 'gold-a', 'reason': 'same fact'}],
                 'unmatched': [{'candidate_id': 'b', 'reason': 'wrong value'}]}
    report = score_alignment(cases(), [alignment], 2)
    assert report['tp'] == 1 and report['fp'] == 1 and report['fn'] == 1
    assert report['precision'] == 0.5 and report['recall'] == 0.5
    assert report['per_category']['personal_ephemera']['recall'] == 0
    assert report['missing_gold_ids'] == ['gold-b']


def test_alignment_cannot_count_one_candidate_twice_or_invent_ids():
    candidate = {'id': 'a', 'text': 'Jun uses drawer B2.'}
    alignment = {'matches': [{'candidate_id': 'a', 'gold_id': 'gold-a', 'reason': 'same'}],
                 'unmatched': [{'candidate_id': 'a', 'reason': 'extra'}]}
    with pytest.raises(ValueError, match='more than once'):
        validate_alignment(alignment, [candidate], cases())
    alignment['unmatched'] = []
    alignment['matches'][0]['gold_id'] = 'invented'
    with pytest.raises(ValueError, match='unknown'):
        validate_alignment(alignment, [candidate], cases())


def test_no_outputs_means_zero_recall_without_precision_denominator():
    metrics = score_alignment(cases(), [{'matches': [], 'unmatched': []}], 0)
    assert metrics['recall'] == 0 and metrics['precision'] is None
    assert metrics['fn'] == 2


class ScriptedClient:
    def __init__(self):
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        if messages[0]['content'].startswith('Extract atomic'):
            return '{"candidates":[]}'
        raise AssertionError('no other model calls should be needed for empty extraction')


class NonemptyClient:
    def __init__(self):
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        instruction = messages[0]['content']
        if instruction.startswith('Extract atomic'):
            text = messages[1]['content']
            return json.dumps({'candidates': [{'text': text, 'prompt_evidence': text,
                                               'probe_question': 'Which drawer does Jun use?'}]})
        if instruction.startswith('Align extracted'):
            payload = json.loads(messages[1]['content'])
            assert set(payload['gold'][0]) == {'gold_id', 'text'}
            assert set(payload['candidates'][0]) == {'candidate_id', 'text'}
            return json.dumps({'matches': [{'candidate_id': payload['candidates'][0]['candidate_id'],
                                           'gold_id': payload['gold'][0]['gold_id'],
                                           'reason': 'same fact'}], 'unmatched': []})
        if instruction.startswith('Compare'):
            return '{"novelty":"new","agrees":null,"reason":"unknown prior"}'
        assert messages[1]['content'] == 'Which drawer does Jun use?'
        return 'I do not know.'


class DiscoveryOnlyClient(NonemptyClient):
    def chat(self, messages):
        instruction = messages[0]['content']
        assert instruction.startswith(('Extract atomic', 'Align extracted'))
        return super().chat(messages)


def test_discovery_measurement_uses_same_extractor_without_prior_or_queue(tmp_path):
    path = tmp_path / 'gold.json'
    path.write_text(json.dumps(cases()[:1]))
    root = tmp_path / 'run'
    client = DiscoveryOnlyClient()
    report = run_evaluation(path, root, client=client, discovery_only=True)
    assert report['tp'] == 1 and report['precision'] == report['recall'] == 1
    assert report['measurement'] == 'discovery'
    assert len(client.messages) == 1
    assert not (root / 'collection/queue/candidates.jsonl').exists()
    receipt = json.loads((root / 'discoveries/doc-0001.json').read_text())
    assert receipt['candidates'][0]['prompt_evidence'] == cases()[0]['prompt_evidence']
    with pytest.raises(ValueError, match='changed'):
        run_evaluation(path, root, client=client, discovery_only=False)


def test_nonempty_collection_matches_only_after_blind_probe(tmp_path):
    path = tmp_path / 'gold.json'
    path.write_text(json.dumps(cases()[:1]))
    client = NonemptyClient()
    report = run_evaluation(path, tmp_path / 'run', client=client)
    assert report['tp'] == 1 and report['fp'] == 0 and report['fn'] == 0
    assert report['precision'] == report['recall'] == 1
    assert len(client.messages) == 3
    assert 'Jun uses drawer B2.' not in json.dumps(client.messages[1])


def test_exact_matches_are_removed_before_semantic_alignment():
    class ResidualClient:
        def chat(self, messages):
            payload = json.loads(messages[1]['content'])
            assert messages[0]['content'].startswith('Verify one')
            assert payload == {'candidate': 'Tonight Jun is tired.',
                               'gold': 'Jun feels tired tonight.'}
            return '{"same_fact":true,"reason":"Same person, state and time."}'

    predictions = [{'id': 'a', 'text': 'Jun uses drawer B2.'},
                   {'id': 'b', 'text': 'Tonight Jun is tired.'}]
    result = align_candidates(predictions, cases(), client=ResidualClient())
    assert len(result['matches']) == 2 and result['unmatched'] == []


def test_partial_gold_claim_fails_separate_semantic_completeness_check():
    class PartialClient:
        def chat(self, messages):
            assert messages[0]['content'].startswith('Verify one')
            return '{"same_fact":false,"reason":"The second approval boundary is absent."}'

    gold = [{'name': 'gold-a', 'text': 'Jun approved collection but left publication unapproved.'}]
    predictions = [{'id': 'a', 'text': 'Jun approved collection.'}]
    result = align_candidates(predictions, gold, client=PartialClient())
    assert result['matches'] == []
    assert result['unmatched'][0]['candidate_id'] == 'a'


def test_extra_exact_duplicate_is_unmatched_when_gold_is_exhausted():
    class NoCalls:
        def chat(self, messages):
            raise AssertionError('no semantic ambiguity remains')

    predictions = [{'id': 'a', 'text': 'Jun uses drawer B2.'},
                   {'id': 'b', 'text': 'Jun uses drawer B2.'}]
    result = align_candidates(predictions, cases()[:1], client=NoCalls())
    assert result['matches'][0]['candidate_id'] == 'a'
    assert result['unmatched'][0]['candidate_id'] == 'b'


def test_gold_labels_are_not_sent_and_completed_run_is_immutable(tmp_path):
    path = tmp_path / 'gold.json'
    path.write_text(json.dumps(cases()))
    client = ScriptedClient()
    root = tmp_path / 'run'
    report = run_evaluation(path, root, client=client)
    assert report['fn'] == 2
    assert len(client.messages) == 2
    for messages in client.messages:
        payload = messages[1]['content']
        assert 'gold-a' not in payload and 'gold-b' not in payload
        assert 'new_durable' not in payload and 'expected' not in payload
        assert payload in ('Jun uses drawer B2.', 'Jun feels tired tonight.')
    with pytest.raises(ValueError, match='completed'):
        run_evaluation(path, root, client=client)
    changed = cases()
    changed[0]['expected']['route'] = 'memory'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='changed'):
        run_evaluation(path, root, client=client)
