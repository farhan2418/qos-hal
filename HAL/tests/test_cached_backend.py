"""Tests for qos_hal.cached_backend.CachedBackend."""

from datetime import datetime

import pytest

from qos_hal.backend import Backend, CalibrationData, JobResult, JobStatus, Topology
from qos_hal.cached_backend import CachedBackend


class _CountingBackend(Backend):
    """A fake Backend that counts real calls, so tests can assert exactly
    how many times the wrapped backend was actually hit (vs. served from
    cache)."""

    def __init__(self):
        self._connected = False
        self.topology_calls = 0
        self.calibration_calls = 0
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.submit_calls = 0

    @property
    def is_connected(self):
        return self._connected

    @property
    def _identifier(self):
        return "FAKE"

    def connect(self, backend_name=None):
        self.connect_calls += 1
        self._connected = True

    def disconnect(self):
        self.disconnect_calls += 1
        self._connected = False

    def get_topology(self):
        self.topology_calls += 1
        return Topology(
            backend_name="fake",
            num_qubits=5,
            coupling_map=[(0, 1)],
            basis_gates=["rz", "sx", "x", "cx"],
        )

    def get_calibration(self):
        self.calibration_calls += 1
        return CalibrationData(
            backend_name="fake",
            timestamp=datetime.now(),
            gate_errors={},
            readout_errors={},
            t1_seconds={},
            t2_seconds={},
        )

    def list_available_backends(self):
        return ["fake"]

    def submit_job(self, circuit):
        self.submit_calls += 1
        return self._make_job_id("raw123")

    def get_job_status(self, job_id):
        return JobStatus.DONE

    def get_result(self, job_id):
        return JobResult(job_id=job_id, counts={"00": 100})

    def cancel_job(self, job_id):
        pass


class _FakeClock:
    """A controllable stand-in for time.monotonic()."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr("qos_hal.cached_backend.time.monotonic", fake)
    return fake


def test_get_topology_is_cached_on_second_call(clock):
    inner = _CountingBackend()
    cached = CachedBackend(inner, topology_ttl_seconds=100)

    cached.get_topology()
    cached.get_topology()
    cached.get_topology()

    assert inner.topology_calls == 1


def test_get_topology_refetches_after_ttl_expires(clock):
    inner = _CountingBackend()
    cached = CachedBackend(inner, topology_ttl_seconds=100)

    cached.get_topology()
    assert inner.topology_calls == 1

    clock.advance(99)
    cached.get_topology()
    assert inner.topology_calls == 1  # still within TTL

    clock.advance(2)  # total elapsed: 101s > 100s TTL
    cached.get_topology()
    assert inner.topology_calls == 2


def test_get_calibration_has_independent_ttl_from_topology(clock):
    inner = _CountingBackend()
    cached = CachedBackend(inner, topology_ttl_seconds=1000, calibration_ttl_seconds=10)

    cached.get_topology()
    cached.get_calibration()

    clock.advance(20)  # expires calibration (ttl=10) but not topology (ttl=1000)

    cached.get_topology()
    cached.get_calibration()

    assert inner.topology_calls == 1
    assert inner.calibration_calls == 2


def test_disconnect_clears_cached_data(clock):
    inner = _CountingBackend()
    cached = CachedBackend(inner, topology_ttl_seconds=1000)

    cached.get_topology()
    assert inner.topology_calls == 1

    cached.disconnect()
    cached.get_topology()

    # Cache was cleared on disconnect, so this must be a real fetch again,
    # even though the TTL (1000s) hadn't expired.
    assert inner.topology_calls == 2


def test_default_ttls_differ_topology_vs_calibration():
    inner = _CountingBackend()
    cached = CachedBackend(inner)
    assert cached._topology_ttl != cached._calibration_ttl
    assert cached._topology_ttl > cached._calibration_ttl


@pytest.mark.parametrize("method,args", [
    ("connect", (None,)),
    ("submit_job", ("some_circuit",)),
    ("get_job_status", ("FAKE_raw123",)),
    ("get_result", ("FAKE_raw123",)),
    ("cancel_job", ("FAKE_raw123",)),
])
def test_non_cached_methods_delegate_to_wrapped_backend(method, args):
    inner = _CountingBackend()
    cached = CachedBackend(inner)
    getattr(cached, method)(*args)  # must not raise; exercises delegation


def test_is_connected_and_identifier_delegate_to_wrapped_backend():
    inner = _CountingBackend()
    cached = CachedBackend(inner)

    assert cached.is_connected == inner.is_connected
    cached.connect()
    assert cached.is_connected == inner.is_connected is True
    assert cached._identifier == "FAKE"
