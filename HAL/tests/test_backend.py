"""Tests for the Backend ABC contract itself (not any concrete backend)."""

import pytest

from qos_hal.backend import Backend, BackendConnectionError, BackendJobError


class _DummyBackend(Backend):
    """Minimal complete Backend implementation, shared across tests.

    Every abstract method is implemented; job-related methods raise
    NotImplementedError since most tests here don't need them. connect()
    tracks call count so auto-connect behavior can be asserted precisely.
    """

    def __init__(self, identifier="DUMMY", fail_connect=False):
        self._connected = False
        self.connect_calls = 0
        self._fail_connect = fail_connect
        self.__identifier = identifier

    @property
    def is_connected(self):
        return self._connected

    @property
    def _identifier(self):
        return self.__identifier

    def connect(self, backend_name=None):
        self.connect_calls += 1
        if self._fail_connect:
            raise BackendConnectionError("simulated auth failure")
        self._connected = True

    def disconnect(self):
        self._connected = False

    def get_topology(self):
        raise NotImplementedError

    def get_calibration(self):
        raise NotImplementedError

    def list_available_backends(self):
        raise NotImplementedError

    def submit_job(self, circuit):
        raise NotImplementedError

    def get_job_status(self, job_id):
        raise NotImplementedError

    def get_result(self, job_id):
        raise NotImplementedError

    def cancel_job(self, job_id):
        raise NotImplementedError


def test_backend_cannot_be_instantiated_directly():
    """Backend has abstract methods, so it must not be directly instantiable."""
    with pytest.raises(TypeError):
        Backend()


def test_incomplete_subclass_cannot_be_instantiated():
    """A subclass that skips even one abstract method must fail to instantiate.

    This is the core value of using abc: an incomplete backend fails loudly
    at instantiation time, not silently at 2am during a job submission.
    """

    class IncompleteBackend(Backend):
        # Deliberately omits everything except is_connected/connect/_identifier.
        @property
        def is_connected(self):
            return False

        @property
        def _identifier(self):
            return "INCOMPLETE"

        def connect(self, backend_name=None):
            pass

    with pytest.raises(TypeError):
        IncompleteBackend()


def test_complete_subclass_can_be_instantiated():
    """A subclass implementing every abstract method instantiates cleanly."""
    backend = _DummyBackend()
    assert backend.is_connected is False
    backend.connect()
    assert backend.is_connected is True


def test_ensure_connected_autoconnects_when_disconnected():
    """_ensure_connected() should call connect() only if not already connected."""
    backend = _DummyBackend()
    assert backend.connect_calls == 0

    backend._ensure_connected()
    assert backend.connect_calls == 1
    assert backend.is_connected is True

    # Already connected: must NOT call connect() again.
    backend._ensure_connected()
    assert backend.connect_calls == 1


def test_ensure_connected_propagates_connection_failure():
    """If connect() raises, _ensure_connected() must let that propagate unchanged."""
    backend = _DummyBackend(fail_connect=True)
    with pytest.raises(BackendConnectionError):
        backend._ensure_connected()


# ---------------------------------------------------------------------
# Composite job_id scheme: _make_job_id / _parse_job_id
# ---------------------------------------------------------------------

def test_make_job_id_prefixes_with_identifier():
    backend = _DummyBackend(identifier="IBM_FEZ")
    assert backend._make_job_id("d2c1abc123") == "IBM_FEZ_d2c1abc123"


def test_parse_job_id_recovers_raw_id():
    backend = _DummyBackend(identifier="IBM_FEZ")
    job_id = backend._make_job_id("d2c1abc123")
    assert backend._parse_job_id(job_id) == "d2c1abc123"


def test_parse_job_id_rejects_foreign_backend_job_id():
    """A job_id minted by a different backend instance must be rejected loudly,
    not silently misrouted to the wrong provider."""
    fez = _DummyBackend(identifier="IBM_FEZ")
    torino_job_id = "IBM_TORINO_xyz789"

    with pytest.raises(BackendJobError):
        fez._parse_job_id(torino_job_id)


def test_make_then_parse_job_id_roundtrip_is_lossless():
    backend = _DummyBackend(identifier="IBM_TORINO")
    raw = "some_raw_id_with_underscores_too"
    assert backend._parse_job_id(backend._make_job_id(raw)) == raw
