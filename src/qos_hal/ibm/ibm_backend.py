"""
IBMBackend — concrete Backend implementation wrapping qiskit-ibm-runtime.

Credential handling: reads from environment variables (not a locally saved
account, and not constructor args) so this class works unmodified inside a
daemon with no interactive login. See QOS_IBM_TOKEN / QOS_IBM_INSTANCE below.
"""

import os

from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from qos_hal.backend import (
    Backend,
    BackendConnectionError,
    BackendJobError,
    CalibrationData,
    JobResult,
    JobStatus,
    Topology,
)

QOS_IBM_TOKEN_ENV = "QOS_IBM_TOKEN"
QOS_IBM_INSTANCE_ENV = "QOS_IBM_INSTANCE"

# Resolves a previously open design question ("basis_gates vs target"):
# target.operation_names on a real IBM backend is NOT the same thing as a
# transpile()-safe basis_gates list. It includes dynamic-circuits
# control-flow instructions (if_else, while_loop, for_loop, switch_case,
# break_loop, continue_loop) and can include vendor-internal instruction
# names that qiskit.transpile()'s BasisTranslator has no equivalence-library
# entry for at all -- passing target.operation_names straight through broke
# transpilation on real hardware. get_standard_gate_name_mapping() is
# Qiskit's own canonical registry of every gate name it can actually
# translate to/from (it already includes "measure", "delay", and "reset"
# alongside real gates), so filtering operation_names down to that set is
# what makes basis_gates+coupling_map alone (no target= object, no
# QiskitRuntimeService on the client side) sufficient for a real
# transpile() call. This is intentionally the "filter to standard gates
# only" option from the three originally raised, not the "expose target
# directly" option -- keeps the wire protocol JSON-only.
_TRANSLATABLE_GATE_NAMES = frozenset(get_standard_gate_name_mapping().keys())


def _filter_translatable_gates(operation_names: list[str]) -> list[str]:
    """Pure helper, extracted out of get_topology() so this filtering
    logic is unit-testable without a real IBM Target object — see the
    module-level comment above for why this filtering exists at all."""
    return [name for name in operation_names if name in _TRANSLATABLE_GATE_NAMES]

# Maps IBM's own job status strings to our normalized JobStatus enum.
_IBM_STATUS_MAP = {
    "QUEUED": JobStatus.QUEUED,
    "VALIDATING": JobStatus.QUEUED,
    "RUNNING": JobStatus.RUNNING,
    "DONE": JobStatus.DONE,
    "CANCELLED": JobStatus.CANCELLED,
    "ERROR": JobStatus.FAILED,
}


class IBMBackend(Backend):
    """Backend implementation targeting IBM Quantum Platform."""

    def __init__(self):
        self._service: QiskitRuntimeService | None = None
        self._backend = None  # resolved IBM backend object, set by connect()

    # ---- Lifecycle -------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._service is not None and self._backend is not None

    @property
    def _identifier(self) -> str:
        # e.g. "ibm_fez" -> "IBM_FEZ". Requires an active connection, since
        # the identifier is the *resolved* backend's own name, not "IBM"
        # generically (multiple IBM backends must have distinct identifiers).
        if self._backend is None:
            raise BackendConnectionError(
                "_identifier requested before connect() resolved a backend"
            )
        return self._backend.name.upper()

    def connect(self, backend_name: str | None = None) -> None:
        token = os.environ.get(QOS_IBM_TOKEN_ENV)
        instance = os.environ.get(QOS_IBM_INSTANCE_ENV)
        if not token or not instance:
            raise BackendConnectionError(
                f"Missing IBM credentials: set {QOS_IBM_TOKEN_ENV} and "
                f"{QOS_IBM_INSTANCE_ENV} environment variables."
            )

        try:
            self._service = QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=token,
                instance=instance,
            )
            self._backend = (
                self._service.backend(backend_name)
                if backend_name
                else self._service.least_busy()
            )
        except Exception as e:
            self._service = None
            self._backend = None
            raise BackendConnectionError(f"Failed to connect to IBM backend: {e}") from e

    def disconnect(self) -> None:
        # No real teardown needed: QiskitRuntimeService holds no persistent
        # connection, just configured auth. Dropping references is enough
        # to make is_connected False and force a fresh connect() next time.
        self._service = None
        self._backend = None

    # ---- Read-only device state --------------------------------------

    def get_topology(self) -> Topology:
        self._ensure_connected()
        target = self._backend.target
        raw_coupling_map = target.build_coupling_map()

        if raw_coupling_map is None:
            fully_connected = True
            n = target.num_qubits
            coupling_map = [(i, j) for i in range(n) for j in range(n) if i != j]
        else:
            fully_connected = False
            coupling_map = list(raw_coupling_map.get_edges())

        raw_operation_names = list(target.operation_names)
        basis_gates = _filter_translatable_gates(raw_operation_names)

        return Topology(
            backend_name=self._backend.name,
            num_qubits=target.num_qubits,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            simulator=self._backend.configuration().simulator,
            fully_connected=fully_connected,
        )

    def get_calibration(self) -> CalibrationData:
        self._ensure_connected()
        target = self._backend.target
        properties = self._backend.properties()

        gate_errors: dict[str, float] = {}
        for gate_name, qargs_dict in target.items():
            for qargs, props in qargs_dict.items():
                if qargs is None:
                    # A global instruction (applies to any qubit, with no
                    # qubit-specific properties) has no tuple to build a
                    # per-qubit key from — key on the gate name alone.
                    key = gate_name
                else:
                    key = f"{gate_name}_{'_'.join(map(str, qargs))}"
                gate_errors[key] = (
                    props.error
                    if props is not None and props.error is not None
                    else float("nan")
                )

        def _safe(fn, q):
            try:
                value = fn(q)
                return value if value is not None else float("nan")
            except Exception:
                return float("nan")

        n = target.num_qubits
        readout_errors = {q: _safe(properties.readout_error, q) for q in range(n)}
        t1_seconds = {q: _safe(properties.t1, q) for q in range(n)}
        t2_seconds = {q: _safe(properties.t2, q) for q in range(n)}

        return CalibrationData(
            backend_name=self._backend.name,
            timestamp=properties.last_update_date,
            gate_errors=gate_errors,
            readout_errors=readout_errors,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
        )

    def list_available_backends(self) -> list[str]:
        """Thin wrapper around QiskitRuntimeService.backends() — this data
        is already provided by the SDK, we're just exposing it through the
        provider-neutral contract."""
        self._ensure_connected()
        return [b.name for b in self._service.backends()]

    # ---- Execution -----------------------------------------------------

    def submit_job(self, circuit) -> str:
        """Submit an already-ISA-compiled circuit via the Sampler primitive.

        Sampler-only for now (see design discussion: Estimator returns a
        fundamentally different result shape that doesn't fit JobResult).

        Does NOT transpile. If `circuit` isn't ISA-compliant for this
        backend, qiskit-ibm-runtime raises its own validation error, which
        we wrap into BackendJobError and propagate — the caller (System
        Services / compiler layer) is expected to catch this and re-run
        its transpilation using get_topology()/get_calibration().
        """
        self._ensure_connected()
        sampler = Sampler(mode=self._backend)
        try:
            job = sampler.run([circuit])
        except Exception as e:
            raise BackendJobError(f"submit_job failed (is the circuit ISA-compiled for {self._backend.name}?): {e}") from e

        return self._make_job_id(job.job_id())

    def get_job_status(self, job_id: str) -> JobStatus:
        self._ensure_connected()
        raw_id = self._parse_job_id(job_id)
        try:
            ibm_job = self._service.job(raw_id)
            raw_status = ibm_job.status()
        except Exception as e:
            raise BackendJobError(f"Failed to get status for job {job_id!r}: {e}") from e

        return _IBM_STATUS_MAP.get(str(raw_status), JobStatus.FAILED)

    def get_result(self, job_id: str) -> JobResult:
        self._ensure_connected()
        raw_id = self._parse_job_id(job_id)
        try:
            ibm_job = self._service.job(raw_id)
            raw_result = ibm_job.result()
        except Exception as e:
            raise BackendJobError(f"Failed to get result for job {job_id!r}: {e}") from e
        counts = raw_result[0].data.c.get_counts()
        return JobResult(job_id=job_id, counts=counts, raw=raw_result)

    def cancel_job(self, job_id: str) -> None:
        self._ensure_connected()
        raw_id = self._parse_job_id(job_id)
        try:
            ibm_job = self._service.job(raw_id)
            ibm_job.cancel()
        except Exception as e:
            raise BackendJobError(f"Failed to cancel job {job_id!r}: {e}") from e
