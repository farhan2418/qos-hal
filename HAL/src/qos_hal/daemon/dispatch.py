"""
Dispatch layer: translates a parsed JSON request ({"method", "params"})
into an actual call on the shared Backend instance, and turns whatever
comes back into a JSON-safe structure.

Design decisions (settled during architecture discussion):
  - circuit is transmitted as QPY, base64-encoded into a string. QPY is
    Qiskit's own serialization format, built to round-trip circuit
    objects exactly (unlike QASM3, which can lose some circuit
    features) — chosen because submit_job() receives an already
    ISA-compiled circuit, where faithfulness matters more than having a
    human-readable wire format.
  - Dataclass results (Topology, CalibrationData, JobResult) are
    converted with dataclasses.asdict(), with one explicit fixup:
    CalibrationData.timestamp (a datetime) is converted to
    .isoformat(), since plain JSON has no datetime type.
  - float('nan') values inside CalibrationData are left as-is. Python's
    json module emits these as a non-standard `NaN` token and parses it
    back correctly — acceptable here because both the daemon and its
    Python SDK client are Python. A future non-Python client would need
    its own handling for this.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime
import io

from qiskit import qpy

from qos_hal.backend import Backend, JobStatus


def _decode_circuit(circuit_qpy_b64: str):
    """Reverse of the client-side encoding: base64 -> QPY bytes -> QuantumCircuit."""
    raw = base64.b64decode(circuit_qpy_b64)
    circuits = qpy.load(io.BytesIO(raw))
    return circuits[0]


def _to_jsonable(value):
    """Recursively convert dataclasses/datetimes/enums into JSON-safe values.

    Everything else (str, int, float incl. nan, bool, None, list, dict)
    is returned unchanged and left to json.dumps to handle.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, JobStatus):
        return value.value
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def make_dispatcher(backend: Backend):
    """Build a dispatch_fn closure bound to one shared Backend instance.

    The returned function is what DaemonServer calls for every request;
    it is called with the _backend_lock already held (see server.py), so
    nothing here needs its own locking.
    """

    def dispatch(request: dict):
        method = request.get("method")
        params = request.get("params") or {}

        if method == "connect":
            backend.connect(params.get("backend_name"))
            return None

        if method == "list_available_backends":
            return backend.list_available_backends()

        if method == "get_topology":
            return _to_jsonable(backend.get_topology())

        if method == "get_calibration":
            return _to_jsonable(backend.get_calibration())

        if method == "submit_job":
            circuit = _decode_circuit(params["circuit"])
            return backend.submit_job(circuit)

        if method == "get_job_status":
            status = backend.get_job_status(params["job_id"])
            return _to_jsonable(status)

        if method == "get_result":
            job_result = backend.get_result(params["job_id"])
            # job_result.raw carries whatever provider-specific object the
            # backend implementation stashed there (e.g. IBMBackend's raw
            # SamplerV2 result) — never assumed JSON-safe, and historically
            # wasn't: _to_jsonable() doesn't recognize it, so it passed
            # through unchanged and only failed later at json.dumps() time,
            # in server.py, outside any error handling — crashing the whole
            # connection instead of returning a clean error (see server.py's
            # _handle_line for the matching fix). Rather than attempt to
            # serialize whatever a given backend happens to have put there,
            # raw is always dropped to None over the wire; the real object
            # remains available to anything using the Backend directly
            # in-process (i.e. not through this daemon protocol).
            return {"job_id": job_result.job_id, "counts": job_result.counts, "raw": None}

        if method == "cancel_job":
            backend.cancel_job(params["job_id"])
            return None

        raise ValueError(f"Unknown method: {method!r}")

    return dispatch
