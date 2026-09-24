"""
Unit tests for qos_hal.target_builder.build_target.

Pure function under test, plain hand-built Topology/CalibrationData in —
Target out — so none of this needs a daemon, a socket, or real IBM
credentials. See tests/test_client.py for the thinner integration tests
that confirm QosHalClient.get_target()/transpile() actually wire this
into a real daemon round-trip.
"""

import math
from datetime import datetime

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Target

from qos_hal.backend import CalibrationData, Topology
from qos_hal.target_builder import build_target


@pytest.fixture
def topology():
    return Topology(
        backend_name="fake",
        num_qubits=3,
        coupling_map=[(0, 1), (1, 0), (1, 2), (2, 1)],
        basis_gates=["rz", "sx", "x", "cx", "measure", "reset"],
    )


@pytest.fixture
def calibration():
    return CalibrationData(
        backend_name="fake",
        timestamp=datetime(2026, 9, 17, 12, 0, 0),
        gate_errors={
            "rz_0": 0.0,  # rz is a virtual/software gate on IBM hardware -- 0 error is realistic
            "sx_0": 0.0003,
            "cx_0_1": 0.008,
            "cx_1_0": 0.008,
            "cx_1_2": 0.012,
            "cx_2_1": 0.012,
            # sx_1, sx_2, x_* deliberately absent -- exercises the
            # missing-key -> nan -> None fallback path
        },
        readout_errors={0: 0.02, 1: float("nan"), 2: 0.03},
        t1_seconds={0: 100e-6, 1: 95e-6, 2: float("nan")},
        t2_seconds={0: 80e-6, 1: 75e-6, 2: 90e-6},
    )


def test_returns_a_real_target(topology, calibration):
    target = build_target(topology, calibration)
    assert isinstance(target, Target)
    assert target.num_qubits == 3


def test_includes_every_basis_gate_name(topology, calibration):
    target = build_target(topology, calibration)
    assert set(target.operation_names) == {"rz", "sx", "x", "cx", "measure", "reset"}


def test_two_qubit_gate_properties_keyed_by_exact_edge(topology, calibration):
    target = build_target(topology, calibration)
    props = target["cx"]
    assert props[(0, 1)].error == pytest.approx(0.008)
    assert props[(1, 2)].error == pytest.approx(0.012)
    # Not in coupling_map at all -- must not have been fabricated.
    assert (0, 2) not in props


def test_one_qubit_gate_properties_present_for_every_qubit(topology, calibration):
    """Known simplification (documented in build_target's docstring):
    basis_gates is one flat list, so a 1-qubit gate is assumed available
    on every qubit, not just ones with calibration data for it."""
    target = build_target(topology, calibration)
    props = target["sx"]
    assert set(props.keys()) == {(0,), (1,), (2,)}
    assert props[(0,)].error == pytest.approx(0.0003)


def test_missing_gate_error_becomes_none_not_nan(topology, calibration):
    """sx_1/sx_2 are absent from calibration.gate_errors in the fixture
    -- must come through as InstructionProperties(error=None), not a
    literal nan leaking into qiskit internals that don't expect it."""
    target = build_target(topology, calibration)
    props = target["sx"]
    assert props[(1,)].error is None
    assert props[(2,)].error is None


def test_measure_uses_readout_errors_not_gate_errors(topology, calibration):
    """measure error must come from the dedicated, guaranteed-complete
    readout_errors dict, not from gate_errors (which may or may not have
    a matching "measure_<q>" key depending on the backend)."""
    target = build_target(topology, calibration)
    props = target["measure"]
    assert props[(0,)].error == pytest.approx(0.02)
    assert props[(2,)].error == pytest.approx(0.03)


def test_readout_error_nan_becomes_none(topology, calibration):
    target = build_target(topology, calibration)
    assert target["measure"][(1,)].error is None


def test_qubit_properties_carry_t1_t2(topology, calibration):
    target = build_target(topology, calibration)
    assert target.qubit_properties[0].t1 == pytest.approx(100e-6)
    assert target.qubit_properties[0].t2 == pytest.approx(80e-6)


def test_qubit_properties_nan_t1_becomes_none(topology, calibration):
    target = build_target(topology, calibration)
    assert target.qubit_properties[2].t1 is None


def test_unrecognized_basis_gate_name_is_skipped_not_crashed(calibration):
    """Defensive path: shouldn't happen in practice (basis_gates is
    already filtered upstream by IBMBackend), but a future backend
    implementation being less strict must not crash target construction."""
    topology = Topology(
        backend_name="fake",
        num_qubits=2,
        coupling_map=[(0, 1)],
        basis_gates=["rz", "totally_made_up_op"],
    )
    target = build_target(topology, calibration)
    assert "totally_made_up_op" not in target.operation_names
    assert "rz" in target.operation_names


def test_real_transpile_call_succeeds_against_the_built_target(topology, calibration):
    """The actual end-to-end point of this module: a circuit with
    measurement transpiles cleanly against a Target built purely from
    qos_hal's own JSON-safe dataclasses -- no CouplingMap import, no
    basis_gates=/coupling_map= footgun, no IBM SDK object anywhere."""
    target = build_target(topology, calibration)

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    transpiled = transpile(qc, target=target, optimization_level=1)
    used_names = set(transpiled.count_ops().keys())
    assert used_names <= {"rz", "sx", "x", "cx", "measure", "reset"}
