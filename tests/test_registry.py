"""Tests for qos_hal.registry."""

import pytest

from qos_hal.backend import Backend
from qos_hal.registry import _REGISTRY, available_backends, get_backend, register


class _FakeBackend(Backend):
    """Trivial complete Backend subclass, used only to test registry mechanics."""

    @property
    def is_connected(self):
        return True

    @property
    def _identifier(self):
        return "FAKE"

    def connect(self, backend_name=None):
        pass

    def disconnect(self):
        pass

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


@pytest.fixture(autouse=True)
def _clean_registry():
    """Isolate each test from real registrations (e.g. "ibm") and from each other."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def test_register_and_get_backend():
    register("fake", _FakeBackend)
    backend = get_backend("fake")
    assert isinstance(backend, _FakeBackend)


def test_get_backend_returns_a_fresh_instance_each_call():
    register("fake", _FakeBackend)
    a = get_backend("fake")
    b = get_backend("fake")
    assert a is not b


def test_lookup_is_case_insensitive():
    register("IBM", _FakeBackend)
    assert isinstance(get_backend("ibm"), _FakeBackend)
    assert isinstance(get_backend("Ibm"), _FakeBackend)


def test_get_unregistered_backend_raises_key_error():
    with pytest.raises(KeyError):
        get_backend("nonexistent")


def test_register_rejects_non_backend_class():
    class NotABackend:
        pass

    with pytest.raises(TypeError):
        register("bad", NotABackend)


def test_register_same_class_twice_is_allowed():
    register("fake", _FakeBackend)
    register("fake", _FakeBackend)  # must not raise
    assert isinstance(get_backend("fake"), _FakeBackend)


def test_register_different_class_under_same_name_raises():
    class OtherFakeBackend(_FakeBackend):
        pass

    register("fake", _FakeBackend)
    with pytest.raises(ValueError):
        register("fake", OtherFakeBackend)


def test_available_backends_lists_registered_names_sorted():
    register("zeta", _FakeBackend)
    register("alpha", _FakeBackend)
    assert available_backends() == ["alpha", "zeta"]


def test_ibm_backend_registers_itself_on_import():
    import qos_hal.ibm  # noqa: F401 — import is the side effect under test

    assert "ibm" in available_backends()
    backend = get_backend("ibm")
    assert backend.__class__.__name__ == "IBMBackend"
