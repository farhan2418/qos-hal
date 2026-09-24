"""
Integration tests for qos_hal.daemon.server.DaemonServer.

Unlike test_dispatch.py, these go through a real Unix socket — testing
message framing (including partial/chunked sends), concurrent clients,
and that exceptions raised deep in a Backend call still produce a clean
JSON error response rather than crashing the connection or the daemon.
"""

import json
import os
import socket
import threading
import time

import pytest

from qos_hal.daemon.dispatch import make_dispatcher
from qos_hal.daemon.server import DaemonServer
from tests.daemon_helpers import FakeBackend


@pytest.fixture
def running_server(tmp_path):
    """Starts a real DaemonServer on a temp Unix socket, backed by a fresh
    FakeBackend, and tears it down after the test."""
    sock_path = str(tmp_path / "qos_hal_test.sock")
    backend = FakeBackend()
    server = DaemonServer(sock_path, make_dispatcher(backend))

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # sleep — avoids flakiness if the OS is briefly slow to bind.
    deadline = time.monotonic() + 2.0 

    while True:
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.connect(sock_path)
            probe.close()
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > deadline:
                raise TimeoutError("daemon never became ready to accept connections")
            time.sleep(0.01)
    yield sock_path, backend
    server.stop()


def _connect(sock_path: str) -> socket.socket:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    return client


def _send_and_recv(client: socket.socket, request: dict, chunked: bool = False) -> dict:
    payload = json.dumps(request).encode() + b"\n"
    if chunked:
        # Simulate a slow/fragmented client: send one byte at a time to
        # exercise the buffer-accumulation logic in _handle_connection,
        # rather than always sending one clean recv()-sized chunk.
        for i in range(0, len(payload), 3):
            client.sendall(payload[i:i + 3])
            time.sleep(0.001)
    else:
        client.sendall(payload)

    buf = b""
    while b"\n" not in buf:
        chunk = client.recv(4096)
        if not chunk:
            raise ConnectionError("server closed connection unexpectedly")
        buf += chunk
    line, _ = buf.split(b"\n", 1)
    return json.loads(line)


def test_basic_request_response(running_server):
    sock_path, _ = running_server
    client = _connect(sock_path)
    resp = _send_and_recv(client, {"method": "list_available_backends", "params": {}})
    assert resp == {"ok": True, "result": ["fake_a", "fake_b"]}
    client.close()


def test_multiple_requests_on_same_connection(running_server):
    """Confirms the connection stays open and framing correctly separates
    successive messages, rather than only working for a single request."""
    sock_path, _ = running_server
    client = _connect(sock_path)

    r1 = _send_and_recv(client, {"method": "list_available_backends", "params": {}})
    r2 = _send_and_recv(client, {"method": "get_topology", "params": {}})
    r3 = _send_and_recv(client, {"method": "cancel_job", "params": {"job_id": "FAKE_x"}})

    assert r1["ok"] and r2["ok"] and r3["ok"]
    assert r2["result"]["backend_name"] == "fake"
    client.close()


def test_chunked_partial_sends_still_frame_correctly(running_server):
    """The core framing guarantee: even if a client's message arrives in
    tiny fragments across multiple recv() calls, the server must still
    parse exactly one complete JSON object per line."""
    sock_path, _ = running_server
    client = _connect(sock_path)
    resp = _send_and_recv(client, {"method": "list_available_backends", "params": {}}, chunked=True)
    assert resp == {"ok": True, "result": ["fake_a", "fake_b"]}
    client.close()


def test_malformed_json_returns_protocol_error_not_crash(running_server):
    sock_path, _ = running_server
    client = _connect(sock_path)
    client.sendall(b"{not valid json\n")
    buf = b""
    while b"\n" not in buf:
        buf += client.recv(4096)
    resp = json.loads(buf.split(b"\n", 1)[0])
    assert resp["ok"] is False
    assert resp["error"]["type"] == "ProtocolError"
    client.close()

    # The connection/server must still be alive and usable afterward —
    # a bad message from one client must not take down the daemon.
    client2 = _connect(sock_path)
    resp2 = _send_and_recv(client2, {"method": "list_available_backends", "params": {}})
    assert resp2["ok"] is True
    client2.close()


def test_request_missing_method_key_returns_protocol_error(running_server):
    sock_path, _ = running_server
    client = _connect(sock_path)
    resp = _send_and_recv(client, {"params": {}})
    assert resp["ok"] is False
    assert resp["error"]["type"] == "ProtocolError"
    client.close()


def test_backend_exception_becomes_clean_error_envelope(running_server):
    """A Backend-level failure (here: unknown method, raised as ValueError
    by dispatch()) must surface as a normal {"ok": false} response, not
    an unhandled exception that kills the connection."""
    sock_path, _ = running_server
    client = _connect(sock_path)
    resp = _send_and_recv(client, {"method": "not_a_real_method", "params": {}})
    assert resp["ok"] is False
    assert resp["error"]["type"] == "ValueError"
    client.close()


def test_concurrent_clients_each_get_correct_isolated_responses(running_server):
    """Thread-per-connection: many clients hitting the daemon at once must
    each get back the response that matches their own request, never a
    response meant for a different client."""
    sock_path, _ = running_server
    n_clients = 8
    results = [None] * n_clients

    def worker(i):
        client = _connect(sock_path)
        resp = _send_and_recv(client, {"method": "get_job_status", "params": {"job_id": f"FAKE_{i}"}})
        results[i] = resp
        client.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert all(r is not None and r["ok"] and r["result"] == "done" for r in results)


def test_calls_are_serialized_through_the_backend_lock(running_server, monkeypatch):
    """Proves the lock actually does something: if two clients call a slow
    Backend method concurrently, the calls must not overlap in time."""
    sock_path, backend = running_server

    call_intervals = []
    lock = threading.Lock()

    original_status = backend.get_job_status

    def slow_status(job_id):
        start = time.monotonic()
        time.sleep(0.1)
        end = time.monotonic()
        with lock:
            call_intervals.append((start, end))
        return original_status(job_id)

    backend.get_job_status = slow_status

    def worker():
        client = _connect(sock_path)
        _send_and_recv(client, {"method": "get_job_status", "params": {"job_id": "FAKE_x"}})
        client.close()

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(call_intervals) == 3
    # No two intervals may overlap — that's what "serialized" means here.
    call_intervals.sort()
    for (s1, e1), (s2, e2) in zip(call_intervals, call_intervals[1:]):
        assert e1 <= s2, f"calls overlapped: ({s1},{e1}) vs ({s2},{e2})"


def test_client_closing_connection_does_not_crash_server(running_server):
    sock_path, _ = running_server
    client = _connect(sock_path)
    client.close()  # close without sending anything

    # Server must still be responsive to a fresh client afterward.
    client2 = _connect(sock_path)
    resp = _send_and_recv(client2, {"method": "list_available_backends", "params": {}})
    assert resp["ok"] is True
    client2.close()


def test_get_result_with_unserializable_raw_does_not_crash_connection(running_server):
    """End-to-end regression test for the bug where IBMBackend's raw
    SamplerV2 result (stashed in JobResult.raw) made it past dispatch()
    unconverted and only failed at json.dumps() time in server.py —
    outside any error handling, silently killing the connection thread
    instead of returning a response. dispatch.py now nulls raw itself
    (see test_dispatch.py), so this call must complete normally here."""
    sock_path, backend = running_server

    class _UnserializableProviderPayload:
        pass

    backend.raw_result = _UnserializableProviderPayload()
    client = _connect(sock_path)
    resp = _send_and_recv(client, {"method": "get_result", "params": {"job_id": "FAKE_smoke123"}})
    assert resp["ok"] is True
    assert resp["result"]["raw"] is None
    client.close()


def test_unserializable_dispatch_result_becomes_error_not_dead_connection(tmp_path):
    """Direct test of the generic safety net in server.py, independent of
    dispatch.py: if ANY dispatch_fn returns something json.dumps() can't
    handle, the connection must survive with a clean {"ok": false} error
    response, not die silently the way it used to (json.dumps() used to
    run outside _handle_line's try/except, in _handle_connection, so a
    bad payload from dispatch_fn crashed the whole connection thread with
    no response sent at all)."""

    class _Unserializable:
        pass

    def bad_dispatch(request):
        return {"looks_fine": "on the surface", "but_this_is_not": _Unserializable()}

    sock_path = str(tmp_path / "unserializable_test.sock")
    server = DaemonServer(sock_path, bad_dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    deadline = time.monotonic() + 2.0
    while True:
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.connect(sock_path)
            probe.close()
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > deadline:
                raise TimeoutError("daemon never became ready to accept connections")
            time.sleep(0.01)

    try:
        client = _connect(sock_path)
        resp = _send_and_recv(client, {"method": "whatever", "params": {}})
        assert resp["ok"] is False
        assert "TypeError" in resp["error"]["type"]
        client.close()

        # And the connection/server survives to serve a next request —
        # the whole point of the fix.
        client2 = _connect(sock_path)
        resp2 = _send_and_recv(client2, {"method": "whatever", "params": {}})
        assert resp2["ok"] is False
        client2.close()
    finally:
        server.stop()
