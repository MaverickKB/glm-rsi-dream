"""Option selections save directly and localize failures without losing settings."""
import json
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest

NODE = shutil.which('node')
CLIENT = Path(__file__).parent / 'fixtures/admin_portal_autonomy.cjs'
pytestmark = pytest.mark.skipif(NODE is None, reason='Portal behavior requires Node')


def select_levels(levels, failures=(), approve_at=()):
    resources = files('glm_rsi_dream').joinpath('resources')
    payload = {'script': str(resources.joinpath('admin_portal/portal.js')),
               'spec': json.loads(resources.joinpath('rsi-admin-openapi.json').read_text()),
               'levels': levels, 'failures': failures, 'approveAt': approve_at}
    result = subprocess.run([NODE, str(CLIENT)], input=json.dumps(payload), text=True,
                            capture_output=True, check=True, timeout=10)
    return json.loads(result.stdout)


def test_selection_saves_each_level_without_confirmation_and_uses_latest_revision():
    result = select_levels(['supervised', 'automatic', 'manual', 'manual'])
    writes = [request for request in result['requests'] if request['method'] == 'PUT']
    assert [request['body']['autonomy'] for request in writes] == ['supervised', 'automatic', 'manual']
    assert [request['body']['expected_revision'] for request in writes] == [1, 2, 3]
    for request in writes:
        assert request['body']['auto_routes'] == ['memory']
        assert request['body']['min_score'] == 0.9
    assert [state['control']['autonomy'] for state in result['states']] == ['supervised', 'automatic', 'manual', 'manual']
    assert all(state['issue'] is None for state in result['states'])
    for layer in result['layers']:
        assert '>Automatic</h3>' in layer['html']
        assert 'Personal notes' in layer['html']
        assert 'authority-banner' not in layer['html']
        assert ('data-action="control-edit"' in layer['html']) == layer['advanced']
        assert ('Access details' in layer['html']) == layer['advanced']


@pytest.mark.parametrize(('status', 'message', 'button'), [
    (403, 'Needs your approval to turn on.', 'Approve'),
    (401, 'Sign in to apply this level.', 'Sign in'),
    (409, 'Settings changed. Refresh and try again.', 'Refresh'),
    (500, "Couldn't save this level.", 'Try again'),
])
def test_failed_selection_keeps_saved_level_and_one_inline_action(status, message, button):
    result = select_levels(['automatic', 'automatic'], [status], [1] if status == 403 else [])
    failed, recovered = result['states']
    assert failed['control']['autonomy'] == 'manual'
    assert failed['issue']['level'] == 'automatic'
    assert failed['issue']['message'] == message
    assert failed['issue']['button'] == button
    assert failed['html'].count('role="alert"') == 1
    assert 'Internal permission details' not in failed['html']
    assert failed['error'] == ''
    assert recovered['control']['autonomy'] == 'automatic'
    assert recovered['issue'] is None


def test_explicit_approval_uses_the_host_renewal_endpoint():
    result = select_levels(['automatic', 'automatic'], [403], [1])
    writes = [request for request in result['requests'] if request['method'] != 'GET']
    assert [(request['method'], request['path']) for request in writes] == [
        ('PUT', '/api/v1/control'), ('POST', '/api/v1/control/approve')]
    assert writes[0]['body'] == writes[1]['body']
    assert result['states'][1]['control']['autonomy'] == 'automatic'
