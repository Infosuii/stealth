import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1]/'scripts/upload_checked.py'
    spec = importlib.util.spec_from_file_location('checked_upload_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_accepted_upload_is_recorded_before_poll_failure_and_cannot_repeat(tmp_path, monkeypatch):
    from stealth.bundle import sha256_file
    m = module(); pkl = tmp_path/'bot.pkl'; pkl.write_bytes(b'checked bytes')
    sha = sha256_file(pkl); receipt = tmp_path/'receipt.json'
    manifest_path = tmp_path/'bot.manifest.json'
    monkeypatch.setattr(m, 'check_upload_artifact', lambda p:dict(sha256=sha, manifest={}, manifest_path=manifest_path))

    class Client:
        uploads = 0
        polls = 0
        fail = True
        def get_upload_status(self, model_id):
            self.polls += 1
            if self.polls == 1:return {'id':'old'}
            if self.fail:raise TimeoutError('network interruption')
            return dict(id='new', content_hash=sha, file_size_bytes=pkl.stat().st_size,
                        python_version='3.11', status='pending', sandbox_ready=1)
        def upload_model(self, model_id, path, python_version):
            assert json.loads(receipt.read_text())['state'] == 'intent_written'
            assert Path(path).read_bytes() == pkl.read_bytes()
            self.uploads += 1
            return {'upload_id':'new'}
    client = Client()
    with pytest.raises(TimeoutError):m.upload_once(client, 'model', pkl, receipt)
    assert json.loads(receipt.read_text())['upload_id'] == 'new'
    with pytest.raises(RuntimeError, match='existing upload receipt'):
        m.upload_once(client, 'model', pkl, receipt)
    client.fail = False
    result = m.verify(client, receipt)
    assert client.uploads == 1 and result['state'] == 'stored_hash_verified'
    assert result['execution_status'] == 'pending'


def test_ambiguous_network_write_retains_intent_and_never_retries(tmp_path, monkeypatch):
    from stealth.bundle import sha256_file
    m = module(); pkl = tmp_path/'bot.pkl'; pkl.write_bytes(b'checked bytes')
    receipt = tmp_path/'receipt.json'
    monkeypatch.setattr(m, 'check_upload_artifact', lambda p:dict(sha256=sha256_file(pkl)))
    class Client:
        calls = 0
        def get_upload_status(self, model_id):return {'id':'old'}
        def upload_model(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError()
    client = Client()
    with pytest.raises(RuntimeError, match='uncertain'):
        m.upload_once(client, 'model', pkl, receipt)
    assert json.loads(receipt.read_text())['state'] == 'uncertain'
    with pytest.raises(RuntimeError, match='existing upload receipt'):
        m.upload_once(client, 'model', pkl, receipt)
    assert client.calls == 1
