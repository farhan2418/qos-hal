"""
Unit tests for qos_hal.ibm.ibm_backend's basis_gates filtering.

IBMBackend itself needs real IBM credentials to construct (connect() talks
to QiskitRuntimeService), so it isn't exercised end-to-end here. What IS
unit-testable without any of that is _filter_translatable_gates(), the
pure function get_topology() uses to turn a real Target's raw
operation_names into something safe to hand to qiskit.transpile()'s
basis_gates= argument — see the module-level comment in ibm_backend.py
for the full "basis_gates vs target" design history this resolves.
"""

from qos_hal.ibm.ibm_backend import _filter_translatable_gates


def test_keeps_real_gate_names():
    names = ["rz", "sx", "x", "cx", "ecr", "id"]
    assert _filter_translatable_gates(names) == names


def test_keeps_measure_delay_reset():
    """These aren't "gates" in the strict sense, but Qiskit's own standard
    gate name mapping includes them, and transpile() needs them present —
    dropping them would break any circuit that measures or has timing."""
    names = ["measure", "delay", "reset"]
    assert _filter_translatable_gates(names) == names


def test_drops_dynamic_circuits_control_flow_ops():
    """The concrete failure mode this fix exists for: real IBM backends'
    target.operation_names includes control-flow instructions that
    transpile()'s BasisTranslator has no equivalence-library entry for at
    all -- passing them straight through as basis_gates broke real
    transpilation."""
    names = ["rz", "sx", "cx", "if_else", "while_loop", "for_loop", "switch_case", "break_loop", "continue_loop"]
    assert _filter_translatable_gates(names) == ["rz", "sx", "cx"]


def test_drops_unrecognized_vendor_internal_names():
    """Standing in for the real "xslow"-style names found during earlier
    manual verification against real hardware -- anything not in
    Qiskit's own standard gate name mapping gets dropped, whatever it's
    called."""
    names = ["rz", "sx", "x", "cx", "xslow", "some_future_vendor_op"]
    assert _filter_translatable_gates(names) == ["rz", "sx", "x", "cx"]


def test_preserves_input_order():
    names = ["cx", "measure", "rz", "if_else", "sx"]
    assert _filter_translatable_gates(names) == ["cx", "measure", "rz", "sx"]


def test_empty_input():
    assert _filter_translatable_gates([]) == []


def test_all_filtered_out():
    assert _filter_translatable_gates(["if_else", "while_loop"]) == []
