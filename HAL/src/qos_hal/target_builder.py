"""
Builds a real qiskit.transpiler.Target directly from qos_hal's own
JSON-safe Topology/CalibrationData dataclasses.

Why this exists: transpile(basis_gates=..., coupling_map=...) has two
real problems for a HAL client that's supposed to know nothing about
IBM-specific SDK objects:
  1. coupling_map= wants a qiskit.transpiler.CouplingMap, not a raw
     list[tuple[int,int]] -- passing the raw list directly causes
     transpile() to misinterpret its length as "one coupling map per
     circuit" (TranspilerError: "Only a single input coupling map can
     be used..."), which is exactly the failure this module exists to
     avoid pushing onto every caller.
  2. It ignores calibration data entirely, so transpile() can't do
     noise-aware layout/routing (preferring lower-error qubits/links) --
     it can only match topology, not quality.
qiskit.transpiler.Target solves both: it's a generic Qiskit class (NOT
IBM-specific), buildable entirely from plain data, and transpile()
accepts it directly via target=. Building one is qos_hal's job -- the
daemon still only ever sends JSON; this module runs entirely on the
client side, turning that JSON back into a Target.

Deliberately a free function taking plain dataclasses (not a
QosHalClient method) so it's unit-testable with no daemon, no socket,
and no real IBM credentials -- see tests/test_target_builder.py.
"""

from __future__ import annotations

import math

from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.transpiler import InstructionProperties, QubitProperties, Target

from qos_hal.backend import CalibrationData, Topology

# Same canonical name->Gate registry IBMBackend.get_topology() already
# filters basis_gates against (see ibm_backend.py's
# _filter_translatable_gates) -- reused here so "a name survived
# filtering" and "we know how to instantiate it" stay the same
# question, answered in exactly one place.
_STANDARD_GATE_MAPPING = get_standard_gate_name_mapping()


def _nan_to_none(value: float) -> float | None:
    """CalibrationData's own convention is: every key always present,
    float('nan') means "unavailable" (see its docstring on why -- so
    e.g. len(t1_seconds) always equals num_qubits). Target's
    convention for "unavailable" is a plain None field. This is the
    one place those two conventions meet, so the translation happens
    explicitly here rather than leaking nan into qiskit internals that
    were never written expecting it.
    """
    return None if math.isnan(value) else value


def build_target(topology: Topology, calibration: CalibrationData) -> Target:
    """Build a Target for topology.backend_name from qos_hal's own data.

    Folds in calibration (gate errors, readout errors, T1/T2) so
    transpile(target=...) can do noise-aware layout/routing, not just
    topology matching -- this was a deliberate choice over the cheaper
    basis_gates+coupling_map-only approach.

    Known simplification: Topology.basis_gates is a single flat list
    (whichever standard gate names the backend supports at all), not a
    per-qubit map -- so every 1-qubit gate in basis_gates is assumed
    available on every qubit. This matches how IBM backends actually
    expose one uniform single-qubit gate set backend-wide; it would
    need revisiting if qos_hal ever targets a backend where that's not
    true.
    """
    target = Target(
        num_qubits=topology.num_qubits,
        qubit_properties=[
            QubitProperties(
                t1=_nan_to_none(calibration.t1_seconds.get(q, float("nan"))),
                t2=_nan_to_none(calibration.t2_seconds.get(q, float("nan"))),
            )
            for q in range(topology.num_qubits)
        ],
    )

    for name in topology.basis_gates:
        gate = _STANDARD_GATE_MAPPING.get(name)
        if gate is None:
            # Shouldn't happen -- topology.basis_gates is already filtered
            # to this exact mapping's keys by the backend that produced
            # it (see ibm_backend.py's _filter_translatable_gates) -- but
            # stay defensive rather than crash on data from a hypothetical
            # future backend implementation that isn't as strict.
            continue

        num_qubits = gate.num_qubits
        if num_qubits == 1:
            props = {}
            for q in range(topology.num_qubits):
                if name == "measure":
                    # Dedicated, guaranteed-complete data source (see
                    # CalibrationData's docstring) -- preferred over
                    # gate_errors, which may or may not carry a
                    # "measure_<q>" entry depending on what the backend's
                    # raw target happened to expose.
                    error = calibration.readout_errors.get(q, float("nan"))
                else:
                    error = calibration.gate_errors.get(f"{name}_{q}", float("nan"))
                props[(q,)] = InstructionProperties(error=_nan_to_none(error))
            target.add_instruction(gate, props)
        elif num_qubits == 2:
            props = {}
            for q1, q2 in topology.coupling_map:
                error = calibration.gate_errors.get(f"{name}_{q1}_{q2}", float("nan"))
                props[(q1, q2)] = InstructionProperties(error=_nan_to_none(error))
            target.add_instruction(gate, props)
        else:
            # e.g. global_phase -- a 0-qubit "instruction" with no
            # per-qubit properties to attach.
            target.add_instruction(gate, {None: InstructionProperties()})

    return target
