import urllib.request
import urllib.error
import json
import os
import uuid
import sys
import pytest

sys.stdout.reconfigure(encoding='utf-8')
BASE = 'http://127.0.0.1:5000'


def test_live_e2e():
    """Live end-to-end integration test against the running Edvora Flask server."""
    try:
        health_req = urllib.request.urlopen(f'{BASE}/api/health', timeout=3)
        assert health_req.status == 200
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        pytest.skip("Live server is not running on 127.0.0.1:5000, skipping live E2E check.")

    # 1. Check HTML
    html = urllib.request.urlopen(f'{BASE}/').read().decode('utf-8')
    assert 'Edvora' in html, 'Missing Edvora brand name'
    assert 'AI Study Assistant' in html, 'Missing AI Study Assistant header'
    assert 'Student Score' not in html, 'Found score predictor text'
    assert 'Attendance' not in html, 'Found attendance text'
    assert 'Assignment Score' not in html, 'Found assignment score'
    assert 'Prediction History' not in html, 'Found prediction history'

    # 2. Upload file
    session_id = 'test-e2e-' + str(uuid.uuid4())[:8]
    boundary = 'WebKitFormBoundary' + uuid.uuid4().hex

    with open('sample_os_notes.txt', 'rb') as f:
        file_bytes = f.read()

    header_part = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="session_id"\r\n\r\n{session_id}\r\n'
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="sample_os_notes.txt"\r\n'
        f'Content-Type: text/plain\r\n\r\n'
    ).encode('utf-8')

    footer_part = f'\r\n--{boundary}--\r\n'.encode('utf-8')
    body = header_part + file_bytes + footer_part

    req = urllib.request.Request(
        f'{BASE}/api/chat/upload',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    res = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
    assert res.get('success') is True
    assert res.get('document', {}).get('filename') == 'sample_os_notes.txt'

    # 3. Query document
    req_msg = urllib.request.Request(
        f'{BASE}/api/chat/message',
        data=json.dumps({
            'session_id': session_id,
            'message': "What is Belady's Anomaly and which algorithm suffers from it?"
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    reply1 = json.loads(urllib.request.urlopen(req_msg).read().decode('utf-8'))
    assert reply1.get('success') is True
    assert len(reply1.get('reply', '')) > 20

    # 4. Multi-turn Follow-up
    req_msg2 = urllib.request.Request(
        f'{BASE}/api/chat/message',
        data=json.dumps({
            'session_id': session_id,
            'message': 'How does LRU handle page replacement in comparison?'
        }).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    reply2 = json.loads(urllib.request.urlopen(req_msg2).read().decode('utf-8'))
    assert reply2.get('success') is True
    assert len(reply2.get('reply', '')) > 20

    # 5. Reset
    req_reset = urllib.request.Request(
        f'{BASE}/api/chat/reset',
        data=json.dumps({'session_id': session_id}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    res_reset = json.loads(urllib.request.urlopen(req_reset).read().decode('utf-8'))
    assert res_reset.get('success') is True

    # 6. Verify History after Reset
    hist_req = urllib.request.Request(f'{BASE}/api/chat/history?session_id={session_id}')
    hist_res = json.loads(urllib.request.urlopen(hist_req).read().decode('utf-8'))
    assert len(hist_res.get('messages', [])) == 0


if __name__ == '__main__':
    test_live_e2e()
    print("ALL LIVE E2E CHECKS PASSED PERFECTLY!")
