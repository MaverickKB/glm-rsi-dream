"""Hosted UI uses one password session and one bounded timeout setting."""
import json
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest

NODE = shutil.which('node')
CLIENT = Path(__file__).parent / 'fixtures/admin_portal_session.cjs'
pytestmark = pytest.mark.skipif(NODE is None, reason='Portal behavior requires Node')


def run_session(*, authenticated=True, setup=False, timeout=None, minutes=None, logout_others=False):
    payload = {'script': str(files('glm_rsi_dream').joinpath('resources/admin_portal/portal.js')),
               'status': {'hosted': True, 'authenticated': authenticated, 'password_enabled': True,
                          'setup_required': setup, 'session_seconds': 86400},
               'minutes': minutes, 'logoutOthers': logout_others}
    if timeout is not None:
        payload['timeout'] = timeout
    result = subprocess.run([NODE, str(CLIENT)], input=json.dumps(payload), capture_output=True,
                            text=True, check=True, timeout=10)
    return json.loads(result.stdout)


@pytest.mark.parametrize('setup', [False, True])
def test_locked_host_shows_one_password_and_never_fetches_private_data(setup):
    result = run_session(authenticated=False, setup=setup)
    assert not result['dataAvailable']
    assert all(request['path'] == '/session/status' for request in result['requests'])
    assert result['welcome'].count('type="password"') == 1
    assert 'name="token"' not in result['welcome']
    assert ('Set password' if setup else 'Sign in') in result['welcome']


def test_valid_session_reuses_cookie_without_browser_engine_key_or_login_prompts():
    result = run_session(logout_others=True)
    assert result['dataAvailable']
    assert all('Authorization' not in (request.get('headers') or {}) for request in result['requests'])
    assert not any(request['path'] == '/session/login' for request in result['requests'])
    assert result['requests'][-1]['path'] == '/session/logout-others'
    for layer in result['layers']:
        assert 'Session timeout' in layer['html']
        for name in ('15 min', '1 hour', '8 hours', '1 day'):
            assert name in layer['html']
        assert ('Custom timeout' in layer['html']) == layer['advanced']
        assert ('Turn lock off' in layer['html']) == layer['advanced']
        assert ('name="token"' in layer['connection']) == layer['advanced']
        assert ('Engine URL' in layer['connection']) == layer['advanced']


@pytest.mark.parametrize(('timeout', 'minutes', 'expected'), [
    ('900', None, 900), ('3600', None, 3600), ('28800', None, 28800),
    ('86400', None, 86400), ('custom', '5', 300), ('custom', '20', 1200),
])
def test_timeout_saves_the_single_server_value(timeout, minutes, expected):
    result = run_session(timeout=timeout, minutes=minutes)
    writes = [request for request in result['requests'] if request['method'] == 'PUT']
    assert writes == [{'path': '/session/settings', 'method': 'PUT',
                       'headers': {'Accept': 'application/json', 'Content-Type': 'application/json'},
                       'body': {'session_seconds': expected}}]
    assert result['error'] is None


@pytest.mark.parametrize('minutes', ['4', '1441', 'not a number'])
def test_custom_timeout_outside_range_does_not_send_a_mutation(minutes):
    result = run_session(timeout='custom', minutes=minutes)
    assert result['error'] == 'Choose a timeout from 5 minutes to 1 day.'
    assert not any(request['method'] == 'PUT' for request in result['requests'])
