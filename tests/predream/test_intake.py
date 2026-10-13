"""Document decoding and journal-based collection without retention decisions."""

import json
import subprocess
import zipfile

import pytest

from glm_rsi_dream.predream.extraction_eval import materialize_documents
from glm_rsi_dream.predream.intake import collect_intake, read_document


class NoFactsGLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return '{"candidates":[]}'


def test_plain_html_and_docx_keep_source_facts(tmp_path):
    documents = {}
    for index in range(4):
        documents['doc-' + str(index + 1).zfill(4)] = 'Jun uses cabinet B2.\nThe sample is < 5 grams.'
    paths = materialize_documents(documents, tmp_path)
    for identifier, path in paths.items():
        decoded = read_document(path)
        for line in documents[identifier].splitlines():
            assert line in decoded
    docx = paths['doc-0004']
    with zipfile.ZipFile(docx) as archive:
        assert '[Content_Types].xml' in archive.namelist()
        assert '_rels/.rels' in archive.namelist()


def test_html_hides_script_and_style(tmp_path):
    path = tmp_path / 'input.html'
    path.write_text('<style>hidden css</style><p>Jun &amp; Mira.</p><script>hidden js</script>')
    assert read_document(path).strip() == 'Jun & Mira.'


def test_pdf_calls_text_converter_and_fails_for_scanned_input(tmp_path, monkeypatch):
    calls = []

    def convert(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='Jun uses cabinet B2.', stderr='')

    monkeypatch.setattr(subprocess, 'run', convert)
    assert read_document(tmp_path / 'input.pdf') == 'Jun uses cabinet B2.'
    assert calls[0][-1] == '-'

    def scanned(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', scanned)
    with pytest.raises(ValueError, match='OCR'):
        read_document(tmp_path / 'input.pdf')


def test_collection_skips_unchanged_files_and_collects_changed_versions(tmp_path):
    intake = tmp_path / 'intake'
    intake.mkdir()
    path = intake / 'note.txt'
    path.write_text('Please check the inventory.')
    client = NoFactsGLM()
    root = tmp_path / 'run'
    first = collect_intake(intake, root, client=client)
    assert len(first['collected']) == 1
    second = collect_intake(intake, root, client=client)
    assert second['collected'] == [] and len(client.calls) == 1
    path.write_text('Please check the revised inventory.')
    third = collect_intake(intake, root, client=client)
    assert len(third['collected']) == 1 and len(client.calls) == 2
    assert first['collected'][0]['source_ref'] != third['collected'][0]['source_ref']
    assert len((root / 'intake.jsonl').read_text().splitlines()) == 2


def test_pdf_timeout_is_journaled_and_retryable(tmp_path, monkeypatch):
    intake = tmp_path / 'intake'
    intake.mkdir()
    (intake / 'note.pdf').write_bytes(b'PDF input')

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(subprocess, 'run', timeout)
    root = tmp_path / 'run'
    first = collect_intake(intake, root, client=NoFactsGLM())
    assert len(first['errors']) == 1
    receipt = json.loads((root / 'intake.jsonl').read_text())
    assert receipt['status'] == 'failed'

    def converted(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout='Can you help?', stderr='')

    monkeypatch.setattr(subprocess, 'run', converted)
    resumed = collect_intake(intake, root, client=NoFactsGLM())
    assert len(resumed['collected']) == 1 and not resumed['errors']


def test_bad_source_remains_retryable_without_blocking_successful_history(tmp_path):
    intake = tmp_path / 'intake'
    intake.mkdir()
    (intake / 'a.txt').write_text('Please check the inventory.')
    bad = intake / 'b.docx'
    bad.write_text('not a zip')
    root = tmp_path / 'run'
    client = NoFactsGLM()
    first = collect_intake(intake, root, client=client)
    assert len(first['collected']) == 1 and len(first['errors']) == 1
    bad.unlink()
    (intake / 'b.txt').write_text('Please check the courier.')
    second = collect_intake(intake, root, client=client)
    assert len(second['collected']) == 1 and not second['errors']
    assert len(client.calls) == 2


def test_intake_ignores_hidden_unsupported_and_symlink_inputs(tmp_path):
    intake = tmp_path / 'intake'
    intake.mkdir()
    outside = tmp_path / 'outside.txt'
    outside.write_text('not in intake')
    (intake / 'link.txt').symlink_to(outside)
    (intake / '.hidden.txt').write_text('hidden')
    (intake / 'image.png').write_bytes(b'not text')
    client = NoFactsGLM()
    result = collect_intake(intake, tmp_path / 'run', client=client)
    assert not client.calls and len(result['skipped']) == 1
    with pytest.raises(ValueError, match='outside'):
        collect_intake(intake, intake / 'run', client=client)


def test_journal_records_empty_sources_as_collected(tmp_path):
    intake = tmp_path / 'intake'
    intake.mkdir()
    (intake / 'input.md').write_text('Can you help?')
    root = tmp_path / 'run'
    collect_intake(intake, root, client=NoFactsGLM())
    receipt = json.loads((root / 'intake.jsonl').read_text())
    assert receipt['candidate_ids'] == [] and receipt['status'] == 'collected'
    assert receipt['text_sha256']
