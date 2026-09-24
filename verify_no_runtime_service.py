"""
Verification script: submit a real job using ONLY QosHalClient for
backend selection, transpilation, and execution -- no
qiskit_ibm_runtime import here, and no raw Topology/CouplingMap
handling either. client.transpile() builds a real qiskit.transpiler.Target
from the daemon's get_topology()/get_calibration() data internally (see
qos_hal/target_builder.py) and calls qiskit.transpile(target=...) for you.

Run this with the daemon already running (`python -m qos_hal.daemon`)
and QOS_IBM_TOKEN / QOS_IBM_INSTANCE set in the daemon's environment.
"""

import time

from qiskit import QuantumCircuit

from qos_hal.backend import JobStatus
from qos_hal.client import QosHalClient

# 1. Simple Bell-state circuit. QuantumCircuit(2, 2) creates a classical
#    register named "c" by default -- IBMBackend.get_result() specifically
#    reads result[0].data.c, so keep that default naming.
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

with QosHalClient() as client:
    # 2. Ask the daemon to connect (least_busy() picks a backend server-side --
    #    the client never needs to know which one, or how).
    client.connect_backend()

    # 3. Transpile entirely through the client -- builds a Target from
    #    fresh topology+calibration internally, no basis_gates=/
    #    coupling_map=/CouplingMap handling on this side at all.
    transpiled = client.transpile(qc, optimization_level=1)
    print(f"transpiled circuit depth={transpiled.depth()}, "
          f"ops={transpiled.count_ops()}")

    # 4. Submit + poll, same as before -- still entirely through the client.
    job_id = client.submit_job(transpiled)
    print("submitted:", job_id)

    while True:
        status = client.get_job_status(job_id)
        print("status:", status)
        if status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            break
        time.sleep(10)

    if status is JobStatus.DONE:
        result = client.get_result(job_id)
        print("counts:", result.counts)
        print("raw (should be None over the wire):", result.raw)

