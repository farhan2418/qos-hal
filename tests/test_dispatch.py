"""
Unit tests for qos_hal.daemon.dispatch.

These call the dispatcher function directly (no socket, no server thread)
— they test routing and serialization logic in isolation from the
networking concerns covered in test_server.py.
"""

import math

import pytest

from qos_hal.daemon.dispatch import make_dispatcher
from tests.daemon_helpers import FakeBackend, encode_circuit, make_test_circuit


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def dispatch(backend):
    return make_dispatcher(backend)


def test_connect_with_no_backend_name(dispatch, backend):
    result = dispatch({"method": "connect", "params": {}})
    assert result is None
    assert backend.connect_calls == [None]


def test_connect_with_explicit_backend_name(dispatch, backend):
    dispatch({"method": "connect", "params": {"backend_name": "ibm_fez"}})
    assert backend.connect_calls == ["ibm_fez"]


def test_list_available_backends(dispatch):
    result = dispatch({"method": "list_available_backends", "params": {}})
    assert result == ["fake_a", "fake_b"]


def test_get_topology_serializes_to_plain_dict(dispatch):
    result = dispatch({"method": "get_topology", "params": {}})
    assert result["backend_name"] == "fake"
    assert result["num_qubits"] == 3
    # coupling_map tuples become JSON-safe lists
    assert result["coupling_map"] == [[0, 1], [1, 2]]
    assert result["fully_connected"] is False


def test_get_calibration_datetime_becomes_isoformat_string(dispatch):
    result = dispatch({"method": "get_calibration", "params": {}})
    assert result["timestamp"] == "2026-09-17T12:00:00"


def test_get_calibration_nan_is_preserved_not_dropped(dispatch):
    result = dispatch({"method": "get_calibration", "params": {}})
    # dict keys become strings after JSON round-trip semantics in our own
    # _to_jsonable (int keys aren't converted; json.dumps would do that
    # later — here we're testing the dict dispatch() returns pre-dumps).
    assert 1 in result["readout_errors"]
    assert math.isnan(result["readout_errors"][1])
    assert result["readout_errors"][0] == 0.02


def test_submit_job_decodes_qpy_circuit_correctly(dispatch, backend):
    circuit = make_test_circuit()
    encoded = encode_circuit(circuit)

    job_id = dispatch({"method": "submit_job", "params": {"circuit": encoded}})

    assert job_id == "FAKE_smoke123"
    assert len(backend.submitted_circuits) == 1
    # Confirm the round-tripped circuit is faithful, not just "a circuit".
    decoded = backend.submitted_circuits[0]
    assert decoded.num_qubits == circuit.num_qubits
    assert decoded.count_ops() == circuit.count_ops()


def test_get_job_status_serializes_enum_to_its_string_value(dispatch):
    result = dispatch({"method": "get_job_status", "params": {"job_id": "FAKE_smoke123"}})
    assert result == "done"  # JobStatus.DONE.value, not "JobStatus.DONE"


def test_get_result_round_trip(dispatch):
    result = dispatch({"method": "get_result", "params": {"job_id": "FAKE_smoke123"}})
    assert result["job_id"] == "FAKE_smoke123"
    assert result["counts"] == {"00": 512, "11": 512}


def test_cancel_job_returns_none(dispatch):
    result = dispatch({"method": "cancel_job", "params": {"job_id": "FAKE_smoke123"}})
    assert result is None


def test_unknown_method_raises_value_error(dispatch):
    with pytest.raises(ValueError):
        dispatch({"method": "not_a_real_method", "params": {}})


def test_missing_required_param_raises_key_error(dispatch):
    # submit_job requires "circuit" in params — omitting it should fail
    # loudly (KeyError from the dict lookup), not silently pass None
    # through to a QPY decoder.
    with pytest.raises(KeyError):
        dispatch({"method": "submit_job", "params": {}})
