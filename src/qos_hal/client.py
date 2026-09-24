"""
Thin Python SDK client for talking to the qos_hal daemon over its Unix
domain socket.

This is the client-side mirror of daemon/server.py + daemon/dispatch.py:
  - Same wire format: one JSON object per line, both directions.
      Request:  {"method": "...", "params": {...}}
      Response: {"ok": true, "result": ...}
               | {"ok": false, "error": {"type": "...", "message": "..."}}
  - Same circuit encoding: QPY, base64-encoded into a string (see
    dispatch.py's module docstring for why QPY was chosen).
  - Same JSON->dataclass reconstruction concerns as dispatch.py's
    _to_jsonable, run in reverse: JSON has no tuple, datetime, or
    non-string-dict-key types, so this module has to rebuild those by
    hand rather than just doing dataclasses.asdict() in reverse.

Design decisions (kept consistent with the rest of the daemon layer):
  - One QosHalClient = one persistent socket connection, opened in
    connect() (or via the `with QosHalClient(...) as client:` context
    manager) and closed in close(). Every RPC reuses that connection —
    this mirrors how the daemon keeps one long-lived Backend instance,
    rather than paying connection setup cost per call.
  - Not thread-safe. The daemon serializes concurrent Backend calls with
    its own lock (see server.py); a single QosHalClient socket has no
    such protection, so concurrent callers should use their own lock or
    one QosHalClient per thread.
  - Server-side error "type" strings are mapped back to the matching
    qos_hal.backend exception class when recognized (so callers can
    `except BackendConnectionError` exactly as they would against a
    local Backend), and wrapped in RemoteBackendError otherwise (e.g.
    ProtocolError, or any exception type raised on the server that this
    client doesn't know about).
  - dict/JSON round-trip fixups applied here, all reversing something
    dispatch.py did on the way out:
      * Topology.coupling_map: JSON gives list-of-lists -> tuples.
      * CalibrationData.readout_errors/t1_seconds/t2_seconds: JSON
        object keys are always strings -> int-keyed dicts.
      * CalibrationData.timestamp: ISO string -> datetime.
      * get_job_status result: plain string -> JobStatus enum member.
    float('nan') needs no fixup: both sides are Python's json module,
    which round-trips NaN correctly (see dispatch.py's docstring) —
    this is a documented Python-to-Python assumption, not something a
    future non-Python client could rely on.

Timeout tiers and reconnect behavior (settled during architecture
discussion — see git log for the fuller reasoning):
  - Every RPC method is classified into one of three tiers, each with
    its own timeout, so a slow IBM auth handshake doesn't need the same
    budget as a local status poll, and so classifying a *new* method
    later is a one-line decision rather than a fresh judgment call:
      * "session" (default 60s) — connect(): the one call that may do
        real provider authentication + backend discovery.
      * "read"    (default 15s) — get_topology/get_calibration/
        list_available_backends/get_job_status/get_result: live
        provider reads on an already-authenticated session.
      * "write"   (default 15s) — submit_job/cancel_job: provider
        writes. Same timeout number as "read" today, but kept as a
        separate tier on purpose, because retry-safety (below) depends
        on the tier, not the timeout value.
    A fourth, separate "transport" timeout (default 5s) applies only to
    opening the raw Unix socket itself in connect() — local IPC, so a
    slow open means something is actually wrong (daemon hung/dead).
  - On a genuine connection-loss (broken pipe, reset, or the daemon
    closing the socket — anything that is NOT a plain timeout, since a
    timeout means the connection is fine but slow) mid-call:
      * "session"/"read" tier: the socket is transparently reconnected
        and the same request is retried once. These calls have no
        side effects that could double-fire, so hiding the hiccup from
        the caller is safe and reduces noise for ordinary discovery
        calls.
      * "write" tier: the socket is still reconnected (so the client
        remains usable for the *next* call), but THIS request is never
        silently retried — because whether the daemon actually
        processed it before the connection died is unknown, and
        blindly retrying submit_job could double-submit a real job on
        real hardware. Instead this raises QosHalClientError telling
        the caller to check get_job_status() (for cancel_job, which
        carries a job_id) or the provider/backend state directly (for
        submit_job, which doesn't have one yet) before deciding whether
        to retry themselves.
    A plain timeout (connection fine, just slow) never triggers a
    reconnect or retry on its own — the socket is torn down so it
    isn't left in an ambiguous framing state, but the caller decides
    whether to call again.
"""

from __future__ import annotations

import base64
import io
import json
import socket
from datetime import datetime
from typing import Any

from qiskit import qpy
from qiskit import transpile as _qiskit_transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target

from qos_hal.backend import (
    BackendConnectionError,
    BackendError,
    BackendJobError,
    CalibrationData,
    JobResult,
    JobStatus,
    Topology,
)
from qos_hal.target_builder import build_target

DEFAULT_SOCKET_PATH = "/tmp/qos_hal.sock"

# Default per-tier timeouts, in seconds. See module docstring for the
# reasoning behind each tier and its default.
DEFAULT_TRANSPORT_TIMEOUT = 5.0
DEFAULT_SESSION_TIMEOUT = 60.0
DEFAULT_READ_TIMEOUT = 15.0
DEFAULT_WRITE_TIMEOUT = 15.0

# Every dispatchable RPC method, classified into a timeout/retry tier.
# Adding a new daemon method later means adding one line here — decide
# whether it opens a session, reads live provider state, or writes to
# the provider, and the timeout + retry behavior follow automatically.
_METHOD_TIERS: dict[str, str] = {
    "connect": "session",
    "list_available_backends": "read",
    "get_topology": "read",
    "get_calibration": "read",
    "get_job_status": "read",
    "get_result": "read",
    "submit_job": "write",
    "cancel_job": "write",
}

# Tiers whose connection-loss handling is "reconnect + silently retry
# once". Everything not in this set ("write") reconnects but surfaces
# the loss to the caller instead — see module docstring.
_RETRY_TIERS = frozenset({"session", "read"})

# Server-side exception type names that map back to a real qos_hal
# exception class. Anything not in this table (e.g. "ProtocolError",
# or a type raised somewhere the daemon author didn't anticipate)
# becomes a RemoteBackendError instead of being silently swallowed
# into a generic type.
_ERROR_TYPE_MAP: dict[str, type[BackendError]] = {
    "BackendConnectionError": BackendConnectionError,
    "BackendJobError": BackendJobError,
    "BackendError": BackendError,
}


class RemoteBackendError(BackendError):
    """Raised for a server-side error whose type isn't one of the known
    qos_hal.backend exception classes (e.g. ProtocolError, or an
    unmapped exception raised inside dispatch()/Backend).

    Keeps the original server-side type name and message rather than
    collapsing everything into a plain RuntimeError, so callers can
    still inspect what actually went wrong on the daemon side.
    """

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        super().__init__(f"{error_type}: {message}")


class QosHalClientError(BackendError):
    """Raised for problems with the client<->daemon connection itself
    (socket errors, timeouts, malformed responses, lost-then-recovered
    connections on non-retryable calls) — distinct from BackendError
    subclasses raised for legitimate backend/job failures, so callers
    can tell "the daemon told me no" from "I couldn't reach the daemon
    (or couldn't be sure it heard me)".
    """


class _ConnectionLost(Exception):
    """Internal signal: the socket died mid-call (not a timeout).

    Caught in _call() to drive the reconnect/retry-or-surface behavior
    described in the module docstring. Never raised to callers of
    QosHalClient directly — it always becomes either a transparent
    retry or a QosHalClientError.
    """


def _encode_circuit(circuit: QuantumCircuit) -> str:
    """QPY + base64 encode, matching dispatch.py's _decode_circuit."""
    buf = io.BytesIO()
    qpy.dump(circuit, buf)
    return base64.b64encode(buf.getvalue()).decode()


def _tuple_pairs(value: list) -> list[tuple[int, int]]:
    """JSON has no tuple type — coupling_map comes back as list-of-lists."""
    return [tuple(pair) for pair in value]


def _int_keyed(value: dict) -> dict[int, float]:
    """JSON object keys are always strings — per-qubit dicts need int keys."""
    return {int(k): v for k, v in value.items()}


def _topology_from_dict(data: dict) -> Topology:
    return Topology(
        backend_name=data["backend_name"],
        num_qubits=data["num_qubits"],
        coupling_map=_tuple_pairs(data["coupling_map"]),
        basis_gates=data["basis_gates"],
        simulator=data["simulator"],
        fully_connected=data["fully_connected"],
    )


def _calibration_from_dict(data: dict) -> CalibrationData:
    return CalibrationData(
        backend_name=data["backend_name"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        gate_errors=data["gate_errors"],
        readout_errors=_int_keyed(data["readout_errors"]),
        t1_seconds=_int_keyed(data["t1_seconds"]),
        t2_seconds=_int_keyed(data["t2_seconds"]),
    )


def _job_result_from_dict(data: dict) -> JobResult:
    return JobResult(job_id=data["job_id"], counts=data["counts"], raw=data["raw"])


def _write_tier_loss_message(method: str, params: dict) -> str:
    """Builds the "check before you retry" guidance for a write-tier
    call whose connection died mid-flight — see module docstring."""
    job_id = params.get("job_id")
    if job_id is not None:
        return (
            f"connection lost while calling {method!r} — its effect on the backend "
            f"is unknown. Call get_job_status({job_id!r}) before deciding whether to "
            f"retry; the connection itself has been reconnected."
        )
    return (
        f"connection lost while calling {method!r} — its effect on the backend is "
        f"unknown, and no job_id was returned to check. Verify the provider/backend "
        f"state directly before resubmitting; the connection itself has been "
        f"reconnected."
    )


class QosHalClient:
    """Python SDK client for the qos_hal daemon.

    Usage:
        with QosHalClient() as client:
            client.connect_backend()
            topo = client.get_topology()
            job_id = client.submit_job(circuit)
            ...

    Every method here is a blocking RPC to the daemon over the shared
    socket connection; there is no separate "local" state to keep in
    sync (unlike a Backend implementation, this class holds no cache
    and no circuit — it is purely a protocol translator).
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        *,
        transport_timeout: float = DEFAULT_TRANSPORT_TIMEOUT,
        session_timeout: float = DEFAULT_SESSION_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
    ):
        self._socket_path = socket_path
        self._transport_timeout = transport_timeout
        self._tier_timeouts = {
            "session": session_timeout,
            "read": read_timeout,
            "write": write_timeout,
        }
        self._sock: socket.socket | None = None
        self._buffer = b""

    # ---- Connection lifecycle (to the daemon socket, distinct from
    # the daemon's own Backend.connect()/disconnect()) ------------------

    def connect(self) -> None:
        """Open the Unix socket connection to the daemon.

        Distinct from connect_backend(): this only establishes the
        transport; the daemon's shared Backend may already be connected
        (or may auto-connect lazily) independent of this call. Uses the
        "transport" timeout tier, not the "session" tier.
        """
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._transport_timeout)
        try:
            sock.connect(self._socket_path)
        except OSError as e:
            raise QosHalClientError(
                f"could not connect to qos_hal daemon at {self._socket_path!r}: {e}"
            ) from e
        self._sock = sock
        self._buffer = b""

    def close(self) -> None:
        """Close the socket connection. Safe to call more than once.

        This is the caller's own deliberate disconnect — unlike a
        connection-loss detected during a call, it never triggers any
        reconnect logic.
        """
        self._teardown_socket()

    def _teardown_socket(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._buffer = b""

    def __enter__(self) -> "QosHalClient":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- Core RPC machinery --------------------------------------------

    def _call(self, method: str, params: dict | None = None) -> Any:
        params = params or {}
        tier = _METHOD_TIERS.get(method, "read")  # unknown methods default to the safer (no-side-effect-assumed) tier
        request = {"method": method, "params": params}

        try:
            return self._send_and_receive(request, tier)
        except _ConnectionLost:
            self._teardown_socket()
            if tier in _RETRY_TIERS:
                self.connect()
                try:
                    return self._send_and_receive(request, tier)
                except _ConnectionLost as e2:
                    raise QosHalClientError(
                        f"lost connection to daemon during {method!r} "
                        f"(reconnected and retried once, still failed): {e2}"
                    ) from e2
            else:
                self.connect()  # reconnect so the client is usable for the NEXT call
                raise QosHalClientError(_write_tier_loss_message(method, params)) from None

    def _send_and_receive(self, request: dict, tier: str) -> Any:
        if self._sock is None:
            raise QosHalClientError(
                "not connected — call connect() first (or use 'with QosHalClient(...) as client:')"
            )

        method = request["method"]
        timeout = self._tier_timeouts[tier]

        try:
            # settimeout() itself can raise OSError (e.g. "Bad file
            # descriptor") if the socket already died under us since
            # the last call — that's connection-loss too, not a
            # framing/timeout concern, so it belongs in this same
            # try block rather than being checked separately.
            self._sock.settimeout(timeout)
            self._sock.sendall(json.dumps(request).encode("utf-8") + b"\n")
            response = self._read_response_line()
        except socket.timeout as e:
            # Connection is presumably fine, just slow — don't guess at
            # retry safety here. Tear down so the next call starts from
            # a clean framing state, but leave the retry decision to
            # the caller.
            self._teardown_socket()
            raise QosHalClientError(f"{method!r} call timed out after {timeout}s") from e
        except OSError as e:
            # Any other socket-level failure (broken pipe, reset,
            # etc.) is treated as connection-loss, handled by _call().
            raise _ConnectionLost(str(e)) from e

        if response["ok"]:
            return response["result"]

        error = response["error"]
        error_type, message = error["type"], error["message"]
        exc_cls = _ERROR_TYPE_MAP.get(error_type)
        if exc_cls is not None:
            raise exc_cls(message)
        raise RemoteBackendError(error_type, message)

    def _read_response_line(self) -> dict:
        while b"\n" not in self._buffer:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise _ConnectionLost("daemon closed the connection unexpectedly")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise QosHalClientError(f"malformed response from daemon: {e}") from e

    # ---- Backend-mirroring RPCs -----------------------------------------
    # Named connect_backend()/disconnect_backend() rather than
    # connect()/disconnect() to avoid colliding with this class's own
    # socket-lifecycle connect()/close() above — the daemon's dispatch
    # table has no "disconnect" method (the daemon owns the shared
    # Backend's lifetime, so a client-triggered disconnect isn't part
    # of the protocol), so only connect_backend() exists here.

    def connect_backend(self, backend_name: str | None = None) -> None:
        """Ask the daemon's shared Backend to connect (or switch target).

        Note this affects every client currently talking to the daemon,
        since all clients share one Backend instance — mirrors
        Backend.connect() on the daemon side exactly. Uses the
        "session" timeout tier and is safely retried on connection-loss
        (re-authenticating is not a destructive operation).
        """
        self._call("connect", {"backend_name": backend_name})

    def list_available_backends(self) -> list[str]:
        return self._call("list_available_backends")

    def get_topology(self) -> Topology:
        return _topology_from_dict(self._call("get_topology"))

    def get_calibration(self) -> CalibrationData:
        return _calibration_from_dict(self._call("get_calibration"))

    def submit_job(self, circuit: QuantumCircuit) -> str:
        """Submit an already ISA-compiled circuit; see Backend.submit_job
        for the compilation requirement — this client does not transpile.

        "write" tier: never silently retried on connection-loss — see
        module docstring.
        """
        return self._call("submit_job", {"circuit": _encode_circuit(circuit)})

    def get_job_status(self, job_id: str) -> JobStatus:
        return JobStatus(self._call("get_job_status", {"job_id": job_id}))

    def get_result(self, job_id: str) -> JobResult:
        return _job_result_from_dict(self._call("get_result", {"job_id": job_id}))

    def cancel_job(self, job_id: str) -> None:
        """"write" tier: never silently retried on connection-loss — see
        module docstring."""
        self._call("cancel_job", {"job_id": job_id})

    # ---- Client-side transpilation support ------------------------------
    # Neither of these talks to the daemon beyond the two "read" tier
    # calls they're built on (get_topology/get_calibration) -- the Target
    # construction itself is pure client-side work, done by
    # qos_hal.target_builder.build_target(). See that module's docstring
    # for why a real Target (not basis_gates=/coupling_map=) is used.

    def get_target(self) -> Target:
        """Fetch fresh topology + calibration and build a Target from them.

        Deliberately NOT cached: topology/calibration are already "read"
        tier daemon calls (see module docstring's timeout/retry table),
        and always re-fetching on every call — rather than caching the
        built Target client-side — matches IBM's own guidance to pull
        fresh calibration data close to submission time, since
        calibration drifts between a device's periodic calibration
        cycles. If you need to transpile many circuits against the same
        snapshot in one script, call this once yourself and pass the
        result to qiskit.transpile(target=...) directly instead of
        calling get_target() (or transpile()) once per circuit.
        """
        topology = self.get_topology()
        calibration = self.get_calibration()
        return build_target(topology, calibration)

    def transpile(self, circuit: QuantumCircuit, **transpile_kwargs) -> QuantumCircuit:
        """Convenience wrapper: get_target() + qiskit.transpile(target=...).

        For anything beyond the default transpile() behavior (a specific
        optimization_level, seed_transpiler, etc.), pass it straight
        through via transpile_kwargs — e.g.
        client.transpile(qc, optimization_level=3). Passing target= or
        backend= yourself here is redundant (target is always the one
        get_target() just built) and will conflict with it; use
        get_target() + qiskit.transpile() directly instead if you need
        to control the target yourself.
        """
        target = self.get_target()
        return _qiskit_transpile(circuit, target=target, **transpile_kwargs)
