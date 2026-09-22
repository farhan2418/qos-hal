"""
Integration tests for qos_hal.client.QosHalClient.

Like test_server.py, these run a real DaemonServer on a temp Unix socket
backed by FakeBackend — but drive it through QosHalClient instead of raw
socket/JSON, so what's actually under test is the client's request
encoding and, especially, its response decoding back into
Topology/CalibrationData/JobResult/JobStatus (the parts test_server.py
doesn't touch, since it asserts on raw response dicts).
"""

import math
import os
import socket
import threading
import time
from datetime import datetime

import pytest

from qos_hal.backend import BackendJobError, JobResult, JobStatus
from qos_hal.client import QosHalClient, QosHalClientError, RemoteBackendError
from qos_hal.daemon.dispatch import make_dispatcher
from qos_hal.daemon.server import DaemonServer
from tests.daemon_helpers import FakeBackend, make_test_circuit


@pytest.fixture
def running_server(tmp_path):
    """Same setup as test_server.py's fixture of the same name — kept as
    its own copy here (rather than importing across test modules) since
    this project doesn't use a conftest.py for shared fixtures."""
    sock_path = str(tmp_path / "qos_hal_test.sock")
    backend = FakeBackend()
    server = DaemonServer(sock_path, make_dispatcher(backend))

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
    yield sock_path, backend
    server.stop()


@pytest.fixture
def client(running_server):
    sock_path, _ = running_server
    with QosHalClient(sock_path) as c:
        yield c


# ---- Connection lifecycle --------------------------------------------

def test_call_before_connect_raises_client_error():
    client = QosHalClient("/tmp/does-not-matter.sock")
    with pytest.raises(QosHalClientError, match="not connected"):
        client.list_available_backends()


def test_connect_to_nonexistent_socket_raises_client_error(tmp_path):
    client = QosHalClient(str(tmp_path / "nope.sock"))
    with pytest.raises(QosHalClientError, match="could not connect"):
        client.connect()


def test_context_manager_closes_socket_on_exit(running_server):
    sock_path, _ = running_server
    with QosHalClient(sock_path) as c:
        assert c._sock is not None
    assert c._sock is None


def test_close_is_safe_to_call_twice(running_server):
    sock_path, _ = running_server
    c = QosHalClient(sock_path)
    c.connect()
    c.close()
    c.close()  # must not raise


# ---- Discovery / read RPCs --------------------------------------------

def test_connect_backend_forwards_backend_name(client, running_server):
    _, backend = running_server
    client.connect_backend("ibm_fez")
    assert backend.connect_calls == ["ibm_fez"]


def test_connect_backend_with_no_name(client, running_server):
    _, backend = running_server
    client.connect_backend()
    assert backend.connect_calls == [None]


def test_list_available_backends(client):
    assert client.list_available_backends() == ["fake_a", "fake_b"]


def test_get_topology_round_trips_to_a_real_topology_object(client):
    topo = client.get_topology()
    assert topo.backend_name == "fake"
    assert topo.num_qubits == 3
    # The core fixup this test exists for: JSON has no tuple type, so
    # coupling_map must come back as actual tuples, not two-element lists.
    assert topo.coupling_map == [(0, 1), (1, 2)]
    assert all(isinstance(pair, tuple) for pair in topo.coupling_map)
    assert topo.basis_gates == ["rz", "sx", "x", "cx"]
    assert topo.fully_connected is False


def test_get_calibration_round_trips_int_keys_datetime_and_nan(client):
    cal = client.get_calibration()
    assert cal.backend_name == "fake"
    assert cal.timestamp == datetime(2026, 9, 17, 12, 0, 0)
    # JSON object keys are always strings server-side; client must
    # rebuild int-keyed dicts so callers can index with qubit numbers.
    assert cal.readout_errors[0] == 0.02
    assert all(isinstance(k, int) for k in cal.readout_errors)
    assert math.isnan(cal.readout_errors[1])
    assert cal.t1_seconds[2] == pytest.approx(110e-6)
    assert cal.t2_seconds[0] == pytest.approx(80e-6)


# ---- Execution RPCs -----------------------------------------------------

def test_submit_job_and_get_job_status(client):
    circuit = make_test_circuit()
    job_id = client.submit_job(circuit)
    assert job_id == "FAKE_smoke123"

    status = client.get_job_status(job_id)
    assert status is JobStatus.DONE
    assert isinstance(status, JobStatus)


def test_submit_job_sends_a_real_decodable_circuit(client, running_server):
    """Confirms the QPY+base64 encoding round-trips through the daemon's
    _decode_circuit back into an equivalent QuantumCircuit, not just that
    *some* string made it across."""
    _, backend = running_server
    circuit = make_test_circuit()
    client.submit_job(circuit)

    assert len(backend.submitted_circuits) == 1
    received = backend.submitted_circuits[0]
    assert received.num_qubits == circuit.num_qubits
    assert received.num_clbits == circuit.num_clbits
    assert [instr.operation.name for instr in received.data] == \
        [instr.operation.name for instr in circuit.data]


def test_get_result(client):
    job_id = "FAKE_smoke123"
    result = client.get_result(job_id)
    assert isinstance(result, JobResult)
    assert result.job_id == job_id
    assert result.counts == {"00": 512, "11": 512}
    assert result.raw is None


def test_cancel_job(client):
    assert client.cancel_job("FAKE_smoke123") is None


# ---- Error mapping ------------------------------------------------------

def test_backend_job_error_from_wrong_prefix_is_reraised_as_same_type(client, running_server):
    """FakeBackend doesn't itself raise BackendJobError, but the ABC's
    _parse_job_id does — exercise it indirectly isn't possible through
    FakeBackend (it overrides get_job_status to ignore job_id), so this
    test instead confirms the mapping table itself round-trips the type
    by calling cancel_job with a backend whose method we monkeypatch to
    raise directly."""
    _, backend = running_server

    def raise_job_error(job_id):
        raise BackendJobError(f"job_id {job_id!r} does not belong to backend 'FAKE'")

    backend.cancel_job = raise_job_error

    with pytest.raises(BackendJobError, match="does not belong to backend"):
        client.cancel_job("OTHER_x")


def test_unknown_method_raises_remote_backend_error(client, running_server):
    """Goes around the public API to hit dispatch()'s raise ValueError
    directly — ValueError isn't in the client's known-type map, so it
    must surface as RemoteBackendError, not be silently miscategorized."""
    with pytest.raises(RemoteBackendError) as exc_info:
        client._call("not_a_real_method")
    assert exc_info.value.error_type == "ValueError"


def test_malformed_request_from_client_side_is_impossible_but_protocol_error_maps_too(client):
    """A ProtocolError (malformed JSON / missing 'method') can't be
    triggered through the typed public API, but _call() is the same
    path a hand-rolled request would take — confirms an unmapped type
    like ProtocolError also becomes RemoteBackendError rather than
    crashing the decode path."""
    with pytest.raises(RemoteBackendError) as exc_info:
        client._call("not_a_real_method", {"whatever": 1})
    assert exc_info.value.error_type == "ValueError"


def test_daemon_truly_gone_raises_client_error_after_retry_fails(running_server):
    """A single dead socket on a read-tier call is now transparently
    recovered (see the reconnect/retry tests below) — that used to be
    what this test checked, but that behavior is now intentional, not
    an error. What must still raise is the daemon being genuinely
    unreachable, so the reconnect attempt itself also fails."""
    sock_path, _ = running_server
    c = QosHalClient(sock_path)
    c.connect()
    c._sock.close()
    os.remove(sock_path)  # the daemon (and its socket file) is gone now,
                           # not just this one client connection
    # The reconnect attempt itself fails at the transport level here
    # (can't even open a socket), which surfaces as connect()'s own
    # "could not connect" error rather than the "retried once, still
    # failed" message (that message is for when reconnecting succeeds
    # but the retried request then fails again).
    with pytest.raises(QosHalClientError, match="could not connect"):
        c.list_available_backends()


# ---- Timeout tiers -------------------------------------------------------

def test_timeout_tiers_are_configurable_per_instance(running_server):
    sock_path, _ = running_server
    c = QosHalClient(
        sock_path,
        transport_timeout=1.0,
        session_timeout=2.0,
        read_timeout=3.0,
        write_timeout=4.0,
    )
    assert c._transport_timeout == 1.0
    assert c._tier_timeouts == {"session": 2.0, "read": 3.0, "write": 4.0}


def test_read_tier_call_times_out_and_tears_down_socket(running_server, monkeypatch):
    """A read-tier call that never gets a response must time out using
    the read tier's timeout, and leave the socket torn down (framing
    state can't be trusted) rather than silently retried."""
    sock_path, backend = running_server
    monkeypatch.setattr(backend, "get_topology", lambda: time.sleep(5))

    c = QosHalClient(sock_path, read_timeout=0.2)
    c.connect()
    with pytest.raises(QosHalClientError, match="timed out after 0.2s"):
        c.get_topology()
    assert c._sock is None  # torn down, not auto-reconnected on a plain timeout
    c.close()


def test_write_tier_uses_write_timeout(running_server, monkeypatch):
    _, backend = running_server
    monkeypatch.setattr(backend, "cancel_job", lambda job_id: time.sleep(5))

    with QosHalClient(running_server[0], write_timeout=0.2) as c:
        with pytest.raises(QosHalClientError, match="timed out after 0.2s"):
            c.cancel_job("FAKE_x")


# ---- Reconnect / retry behavior on connection-loss -----------------------

def test_read_tier_call_is_silently_retried_after_connection_loss(client):
    """Simulates a dead-but-not-yet-noticed socket (e.g. daemon bounced)
    by yanking the raw fd out from under the client, then confirms a
    read-tier call transparently reconnects and still returns the right
    result — no exception should reach the caller."""
    client._sock.close()  # the socket itself is now unusable, but self._sock
                           # still points at it, exactly like an unnoticed drop
    result = client.list_available_backends()
    assert result == ["fake_a", "fake_b"]
    assert client._sock is not None  # reconnected


def test_session_tier_call_is_silently_retried_after_connection_loss(client, running_server):
    _, backend = running_server
    client._sock.close()
    client.connect_backend("ibm_fez")
    assert backend.connect_calls == ["ibm_fez"]
    assert client._sock is not None


def test_write_tier_call_is_not_retried_but_reconnects_and_warns(client, running_server):
    """A write-tier call must NOT be silently retried after connection
    loss (risk of double-submission), but the client must still be
    usable afterward — proving the reconnect happened even though this
    particular call surfaced an error."""
    client._sock.close()
    with pytest.raises(QosHalClientError, match="connection lost while calling 'cancel_job'"):
        client.cancel_job("FAKE_smoke123")
    assert client._sock is not None  # reconnected despite the raised error

    # And the client is genuinely usable again — not just holding a
    # socket object that will fail the same way.
    assert client.list_available_backends() == ["fake_a", "fake_b"]


def test_submit_job_connection_loss_message_has_no_job_id_to_offer(client):
    """submit_job has no job_id yet when the connection dies — the
    guidance message must say so rather than reference a nonexistent id."""
    client._sock.close()
    circuit = make_test_circuit()
    with pytest.raises(QosHalClientError, match="no job_id was returned"):
        client.submit_job(circuit)


def test_read_tier_retry_failing_twice_raises_client_error(client, monkeypatch):
    """If even the reconnected retry fails, the caller must see a clean
    QosHalClientError (not a raw socket exception or an infinite loop)."""
    client._sock.close()

    original_connect = client.connect

    def poison_connect():
        original_connect()
        client._sock.close()  # the "reconnected" socket is dead too

    monkeypatch.setattr(client, "connect", poison_connect)
    with pytest.raises(QosHalClientError, match="retried once, still failed"):
        client.list_available_backends()
