"""
Real-hardware verification script for the qos_hal IBM backend.

Run this on your WSL2 machine with real IBM credentials set:

    export QOS_IBM_TOKEN="your-api-key"
    export QOS_IBM_INSTANCE="your-crn"
    python3 verify_ibm.py

This exercises every method on IBMBackend against your real account, in
order, printing what happens at each step. It's deliberately verbose and
defensive — the goal is to surface exactly where reality diverges from
what backend.py/ibm_backend.py assume, not just to "pass" or "fail".

WARNING: submit_job() below actually queues a real job on real IBM
hardware. This consumes your account's QPU time/quota. The circuit used
is trivial (2-qubit Bell state, 100 shots) to keep the cost minimal, but
this is NOT a simulator run — it is genuinely billed usage.
"""

import math
import sys
import time

from qiskit import QuantumCircuit, transpile

from qos_hal.backend import BackendConnectionError, BackendJobError, JobStatus
from qos_hal.registry import get_backend
import qos_hal.ibm  # noqa: F401 — registers "ibm"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    backend = get_backend("ibm")  # or "ibm_fez" or whatever you have access to

    # ---- 1. connect() ----------------------------------------------
    section("1. connect()")
    try:
        backend.connect()  # no backend_name -> should resolve via least_busy()
        print(f"OK — connected. is_connected = {backend.is_connected}")
        print(f"    resolved backend identifier: {backend._identifier}")
    except BackendConnectionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # ---- 2. list_available_backends() ------------------------------
    section("2. list_available_backends()")
    try:
        names = backend.list_available_backends()
        print(f"OK — {len(names)} backends available:")
        for n in names:
            print(f"    - {n}")
    except Exception as e:
        print(f"FAILED: {e}")

    # ---- 3. get_topology() -------------------------------------------
    section("3. get_topology()")
    try:
        topology = backend.get_topology()
        print(f"OK — backend_name={topology.backend_name}")
        print(f"    num_qubits={topology.num_qubits}")
        print(f"    fully_connected={topology.fully_connected}")
        print(f"    basis_gates={topology.basis_gates}")
        print(f"    coupling_map (first 5 edges): {topology.coupling_map[:5]}")

    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # ---- 4. get_calibration() ------------------------------------------
    section("4. get_calibration()")
    try:
        calib = backend.get_calibration()
        nan_t1_count = sum(1 for v in calib.t1_seconds.values() if math.isnan(v))
        print(f"OK — timestamp={calib.timestamp}")
        print(f"    gate_errors: {len(calib.gate_errors)} entries")
        print(f"    t1_seconds: {len(calib.t1_seconds)} entries, {nan_t1_count} are NaN")
        if len(calib.t1_seconds) != topology.num_qubits:
            print(f"    WARNING: t1_seconds has {len(calib.t1_seconds)} entries, "
                  f"expected {topology.num_qubits} (Topology/CalibrationData mismatch!)")
    except Exception as e:
        print(f"FAILED: {e}")

    # # ---- 5. submit_job() — REAL, BILLED JOB ----------------------------
    # section("5. submit_job() — this queues a REAL job on real hardware")
    confirm = input("Type 'yes' to actually submit a real 2-qubit Bell state job: ")
    if confirm.strip().lower() != "yes":
        print("Skipped remaining steps (5-8) — no job submitted.")
        return

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    print("Transpiling to ISA using our own get_topology() data...")
    from qiskit.transpiler import CouplingMap
    # coupling_map = CouplingMap(couplinglist=topology.coupling_map)
    isa_circuit = transpile(
        qc,
        # basis_gates=topology.basis_gates,
        # coupling_map=coupling_map,
        target=backend._backend.target,
        optimization_level=1,
    )
    print(f"    transpiled circuit depth: {isa_circuit.depth()}")

    try:
        job_id = backend.submit_job(isa_circuit)
        print(f"OK — job_id = {job_id}")
    except BackendJobError as e:
        print(f"FAILED (this is exactly the kind of thing we're checking for): {e}")
        sys.exit(1)

    # ---- 6. get_job_status() polling loop -----------------------------
    section("6. get_job_status() — polling until DONE")
    deadline = time.monotonic() + 600  # 10 min safety timeout
    while True:
        status = backend.get_job_status(job_id)
        print(f"    status: {status.value}")
        if status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            break
        if time.monotonic() > deadline:
            print("    Timed out waiting for job to finish.")
            break
        time.sleep(5)

    # ---- 7. get_result() -----------------------------------------------
    section("7. get_result()")
    if status == JobStatus.DONE:
        try:
            result = backend.get_result(job_id)
            print(f"OK — counts: {result.counts}")
            print("    (Sanity check: a Bell state should show mostly '00' and '11')")
        except Exception as e:
            print(f"FAILED: {e}")
            print("    This is the known risk area — check whether the classical")
            print("    register name assumption ('meas') is what broke.")
    else:
        print(f"Skipped — job ended in status {status.value}, not DONE")

    # ---- 8. cancel_job() on an already-finished job (should be a no-op or clean error) --
    section("8. cancel_job() on a finished job (checking failure mode)")
    try:
        backend.cancel_job(job_id)
        print("OK — cancel_job() did not raise (check IBM dashboard for actual effect)")
    except BackendJobError as e:
        print(f"Raised BackendJobError as expected for a non-cancellable job: {e}")

    section("Done")
    print("Review the output above for any FAILED or WARNING lines.")


if __name__ == "__main__":
    main()