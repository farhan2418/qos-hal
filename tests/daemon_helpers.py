"""Shared test double for daemon tests (server.py / dispatch.py).

Not a test file itself — no test_ prefix, so pytest won't collect it.
Imported by test_dispatch.py and test_server.py.
"""

from datetime import datetime

from qiskit import QuantumCircuit

from qos_hal.backend import Backend, CalibrationData, JobResult, JobStatus, Topology


class FakeBackend(Backend):
    """A fully in-memory Backend double — no network, no real IBM calls.

    Tracks call counts so tests can assert on how the daemon actually
    invoked it (e.g. exactly one submit_job call happened).
    """

    def __init__(self):
        self._connected = False
        self.connect_calls = []  # list of backend_name args passed
        self.submitted_circuits = []
        self.raw_result = None  # settable by tests to simulate a
                                 # provider-specific, non-JSON-safe raw payload

    @property
    def is_connected(self):
        return self._connected

    @property
    def _identifier(self):
        return "FAKE"

    def connect(self, backend_name=None):
        self.connect_calls.append(backend_name)
        self._connected = True

    def disconnect(self):
        self._connected = False

    def get_topology(self):
        return Topology(
            backend_name="fake", num_qubits=3, coupling_map=[(0, 1), (1, 2)],
            basis_gates=["rz", "sx", "x", "cx"],
        )

    def get_calibration(self):
        return CalibrationData(
            backend_name="fake",
            timestamp=datetime(2026, 9, 17, 12, 0, 0),
            gate_errors={"cx_0_1": 0.01},
            readout_errors={0: 0.02, 1: float("nan"), 2: 0.03},
            t1_seconds={0: 100e-6, 1: 95e-6, 2: 110e-6},
            t2_seconds={0: 80e-6, 1: 75e-6, 2: 90e-6},
        )

    def list_available_backends(self):
        return ["fake_a", "fake_b"]

    def submit_job(self, circuit):
        assert isinstance(circuit, QuantumCircuit)
        self.submitted_circuits.append(circuit)
        return self._make_job_id("smoke123")

    def get_job_status(self, job_id):
        return JobStatus.DONE

    def get_result(self, job_id):
        return JobResult(job_id=job_id, counts={"00": 512, "11": 512}, raw=self.raw_result)

    def cancel_job(self, job_id):
        pass


def make_test_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def encode_circuit(circuit: QuantumCircuit) -> str:
    """Client-side QPY+base64 encoding, matching dispatch.py's _decode_circuit."""
    import base64
    import io

    from qiskit import qpy

    buf = io.BytesIO()
    qpy.dump(circuit, buf)
    return base64.b64encode(buf.getvalue()).decode()
