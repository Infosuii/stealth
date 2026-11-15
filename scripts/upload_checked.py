#!/usr/bin/env python3
"""Upload one checked pickle with a durable receipt; verification never uploads."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stealth import data
from stealth.predict import (approval_gate, ENV_UPLOAD, check_upload_artifact,
                             artifact_snapshot, write_active)


def save(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(doc, indent=2) + '\n')
    tmp.replace(path)


def public_status(status):
    keys = ('id', 'upload_id', 'model_id', 'status', 'python_version', 'sandbox_ready',
            'content_hash', 'file_size_bytes', 'created_at', 'last_run_at',
            'last_run_status', 'error_message', 'lane_active', 'note')
    return {k:status[k] for k in keys if k in status}


def verify(client, receipt, *, timeout=120, poll_seconds=10, active_path=None):
    receipt = Path(receipt)
    doc = json.loads(receipt.read_text())
    gate = check_upload_artifact(Path(doc['pkl']))
    if gate['sha256'] != doc['sha256']:
        raise RuntimeError('local pickle changed after upload intent; do not upload again')
    deadline = time.monotonic() + timeout
    while True:
        status = public_status(client.get_upload_status(doc['model_id']))
        doc['last_status'] = status
        remote_id = status.get('upload_id') or status.get('id')
        hash_matches = status.get('content_hash') == doc['sha256']
        id_matches = bool(remote_id) and (not doc.get('upload_id') or remote_id == doc['upload_id'])
        if hash_matches and id_matches:
            doc['upload_id'] = remote_id
            doc['state'] = 'stored_hash_verified'
            if status.get('error_message'):
                doc['state'] = 'server_run_failed'
                save(receipt, doc)
                raise RuntimeError('platform reports an execution error; receipt retained, never re-upload')
            if status.get('python_version') != '3.11' or status.get('file_size_bytes') != Path(doc['pkl']).stat().st_size:
                save(receipt, doc)
                raise RuntimeError('remote interpreter or file-size mismatch; no repeat upload')
            doc['execution_status'] = status.get('status')
            doc['server_sandbox_ready'] = status.get('sandbox_ready')
            doc['verified_at'] = datetime.now(timezone.utc).isoformat()
            save(receipt, doc)
            manifest = gate['manifest']
            manifest['upload_id'] = remote_id
            manifest['uploaded_at'] = doc['started_at']
            save(gate['manifest_path'], manifest)
            if active_path is not None:
                write_active(doc['model_id'], remote_id, doc['sha256'], Path(doc['pkl']), active_path)
            return doc
        save(receipt, doc)
        if time.monotonic() >= deadline:
            raise TimeoutError('upload not yet verified; use --verify with this receipt, never repeat upload')
        time.sleep(poll_seconds)


def upload_once(client, model_id, pkl, receipt, *, timeout=120, poll_seconds=10, active_path=None):
    receipt = Path(receipt)
    if receipt.exists():
        raise RuntimeError('existing upload receipt; use --verify, never repeat upload')
    gate = check_upload_artifact(Path(pkl))
    before = public_status(client.get_upload_status(model_id))
    doc = dict(state='intent_written', model_id=model_id, pkl=str(Path(pkl).resolve()),
               sha256=gate['sha256'], started_at=datetime.now(timezone.utc).isoformat(), previous_status=before)
    # This record survives a crash or an ambiguous SDK exception before an ID is returned.
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x') as stream:
        stream.write(json.dumps(doc, indent=2) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    try:
        with artifact_snapshot(Path(pkl), gate['sha256']) as (snapshot, _):
            response = client.upload_model(model_id, str(snapshot), python_version='3.11')
        doc['upload_id'] = response.get('upload_id') or response.get('id')
        doc['acceptance'] = public_status(response)
        doc['state'] = 'accepted' if doc['upload_id'] else 'accepted_without_id'
        save(receipt, doc)
        print('upload accepted; upload_id=' + str(doc['upload_id']), flush=True)
    except Exception as exc:
        doc['state'] = 'uncertain'
        doc['error_class'] = type(exc).__name__
        save(receipt, doc)
        raise RuntimeError('upload outcome uncertain; receipt saved; verify read-only, do not retry upload') from None
    return verify(client, receipt, timeout=timeout, poll_seconds=poll_seconds, active_path=active_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pkl', type=Path)
    p.add_argument('--model-id')
    p.add_argument('--receipt', required=True, type=Path)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--i-have-approval', action='store_true')
    p.add_argument('--timeout', type=float, default=120)
    args = p.parse_args()
    if args.verify:
        if not args.receipt.is_file():
            raise RuntimeError('no upload receipt to verify')
        doc = json.loads(args.receipt.read_text())
        check_upload_artifact(Path(doc['pkl']))
    else:
        approval_gate('upload model pickle', ENV_UPLOAD, args.i_have_approval)
        if args.pkl is None or not args.model_id:
            raise RuntimeError('--pkl and --model-id are required for upload')
        if args.receipt.exists():
            raise RuntimeError('existing receipt; use --verify')
        check_upload_artifact(args.pkl)
    client = data.make_client()
    kwargs = dict(timeout=args.timeout, active_path=data.REPO_ROOT/'models/active.json')
    doc = verify(client, args.receipt, **kwargs) if args.verify else upload_once(
        client, args.model_id, args.pkl, args.receipt, **kwargs)
    print(json.dumps({k:doc.get(k) for k in ('state','upload_id','sha256','execution_status','server_sandbox_ready')}, indent=2))


if __name__ == '__main__':
    main()
