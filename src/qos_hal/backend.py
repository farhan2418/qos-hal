"""
Backend ABC — the core, provider-neutral contract every hardware backend
must implement.

Design decisions (settled during architecture discussion):
  - connect() is explicit and takes an optional backend name; other methods
    may auto-connect using the default backend if the caller skipped it,
    but never override an explicit connect().
  - get_topology() / get_calibration() always return LIVE data. Caching is
    handled by a separate CachedBackend decorator, not by this class or its
    subclasses — this keeps every backend "dumb" and caching reusable.
  - submit_job() is non-blocking: it queues the job with the provider and
    returns a job_id immediately. Status/result are separate calls.
  - Read operations (get_topology/get_calibration) are kept strictly apart
    from write/execution operations (submit_job/cancel_job).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class BackendError(Exception):
    """Base class for all Backend-related errors."""


class BackendConnectionError(BackendError):
    """Raised when connect() fails, or auto-connect fails on a caller's behalf."""


class BackendJobError(BackendError):
    """Raised for job submission, status, result, or cancellation failures."""


# --------------------------------------------------------------------------
# Data contracts (the "nouns")
# --------------------------------------------------------------------------

class JobStatus(str, Enum):
    """Normalized job states, independent of provider-specific wording."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class Topology:
    """Static-ish connectivity/capability description of a backend."""

    backend_name: str
    num_qubits: int
    coupling_map: list[tuple[int, int]]
    basis_gates: list[str]
    simulator: bool = False
    fully_connected: bool = False  # True if coupling_map was synthesized
                                     # (backend reported no coupling map),
                                     # not read directly from the backend


@dataclass(frozen=True)
class CalibrationData:
    """Live, calibration-cycle-bound device quality data.

    Every qubit (0..num_qubits-1) and every known gate/qargs combination
    always appears as a key in the relevant dict — nothing is silently
    dropped if data is unavailable. Missing values are represented as
    float('nan'), NOT omitted, so that e.g. len(t1_seconds) always equals
    the backend's num_qubits.

    IMPORTANT: because nan != nan in Python/IEEE-754, never check for a
    missing value with `value == float('nan')` — it will always be False.
    Use `math.isnan(value)` instead.
    """

    backend_name: str
    timestamp: datetime
    gate_errors: dict[str, float]       # e.g. {"cx_0_1": 0.008, ...}
    readout_errors: dict[int, float]    # per-qubit readout error
    t1_seconds: dict[int, float]        # per-qubit T1
    t2_seconds: dict[int, float]        # per-qubit T2


@dataclass(frozen=True)
class JobResult:
    """Normalized measurement result for a completed job."""

    job_id: str
    counts: dict[str, int] = field(default_factory=dict)
    raw: dict | None = None  # provider-specific payload, preserved for provenance


# --------------------------------------------------------------------------
# The Backend ABC
# --------------------------------------------------------------------------

class Backend(ABC):
    """Abstract base class for all quantum hardware backends.

    Any concrete backend (IBM, a simulator, a future vendor) must implement
    every method below. This is the HAL-level contract — no permissions,
    reservation, or scheduling logic belongs here; that lives one layer up.
    """

    # ---- Lifecycle -------------------------------------------------

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether an active connection/session currently exists."""

    @property
    @abstractmethod
    def _identifier(self) -> str:
        """A short, stable, uppercase identifier for this backend instance.

        Used to prefix job IDs so they are self-describing (e.g. a daemon
        juggling multiple Backend instances can route a job_id back to the
        right one without a separate lookup table). For IBM, this is the
        resolved backend's own name, e.g. "IBM_FEZ".
        """

    @abstractmethod
    def connect(self, backend_name: str | None = None) -> None:
        """Authenticate and establish a session.

        Args:
            backend_name: Specific backend to target. If None, a sensible
                default is selected by the implementation.

        Raises:
            BackendConnectionError: if authentication or connection fails.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Cleanly tear down the current connection/session."""

    def _ensure_connected(self) -> None:
        """Auto-connect using the default backend if not already connected.

        Every concrete method that requires an active connection (topology,
        calibration, job operations) should call this first, rather than
        checking is_connected and calling connect() themselves. This keeps
        the auto-connect behavior identical across every such method and
        every backend implementation, instead of being re-implemented
        (and potentially drifting) in each one.

        This does NOT catch connection failures — connect() is already
        responsible for raising BackendConnectionError on failure, and
        that exception is left to propagate up unchanged.
        """
        if not self.is_connected:
            self.connect()

    def _make_job_id(self, raw_job_id: str) -> str:
        """Build the composite, self-describing job_id returned by submit_job().

        Format: "{backend_identifier}_{raw_job_id}", e.g. "IBM_FEZ_d2c1...".
        """
        return f"{self._identifier}_{raw_job_id}"

    def _parse_job_id(self, job_id: str) -> str:
        """Strip this backend's own prefix off a composite job_id.

        Raises:
            BackendJobError: if job_id doesn't belong to this backend
                instance (wrong prefix) — this catches the mistake of
                e.g. passing an IBM job_id to a simulator backend, or a
                job_id from a different IBM backend instance, early and
                loudly rather than silently querying the wrong provider.
        """
        prefix = f"{self._identifier}_"
        if not job_id.startswith(prefix):
            raise BackendJobError(
                f"job_id {job_id!r} does not belong to backend "
                f"{self._identifier!r} (expected prefix {prefix!r})"
            )
        return job_id[len(prefix):]

    # ---- Read-only device state (Flow A) ----------------------------

    @abstractmethod
    def get_topology(self) -> Topology:
        """Return the current backend's connectivity/capability info.

        Always fetched live — caching is the caller's responsibility
        (see CachedBackend).

        Implementations should call self._ensure_connected() first, so
        that calling this before connect() auto-connects to the default
        backend rather than failing outright.

        Raises:
            BackendConnectionError: if not connected and auto-connect fails.
        """

    @abstractmethod
    def get_calibration(self) -> CalibrationData:
        """Return current calibration data (gate errors, T1/T2, etc.).

        Always fetched live — caching is the caller's responsibility
        (see CachedBackend).

        Implementations should call self._ensure_connected() first, so
        that calling this before connect() auto-connects to the default
        backend rather than failing outright.

        Raises:
            BackendConnectionError: if not connected and auto-connect fails.
        """

    @abstractmethod
    def list_available_backends(self) -> list[str]:
        """List the names of physical backends this account can target.

        This is discovery, distinct from get_topology()/get_calibration()
        (which describe the CURRENTLY connected backend): it answers
        "what could I connect to", not "what am I connected to now".

        A client can use this to pick a specific backend_name to pass to
        connect(), rather than relying on the default (least_busy())
        auto-connect behavior.

        Implementations should call self._ensure_connected() first.

        Raises:
            BackendConnectionError: if not connected and auto-connect fails.
        """

    # ---- Execution (Flow B) ------------------------------------------

    @abstractmethod
    def submit_job(self, circuit) -> str:
        """Queue a compiled circuit for execution.

        Non-blocking: returns a job_id immediately: the job is queued with
        the provider, not necessarily started or completed.

        The circuit MUST already be ISA-compiled (transpiled to this
        backend's native basis gates and physical qubit layout) by the
        caller (the System Services / compiler layer) using topology and
        calibration data obtained via get_topology()/get_calibration().
        This method does not transpile on the caller's behalf.

        Implementations should call self._ensure_connected() first, and
        must return job IDs built via self._make_job_id() so they are
        self-describing (see _make_job_id / _parse_job_id).

        Returns:
            job_id: composite, self-describing identifier — see
                _make_job_id(). Used as-is for status/result/cancel calls.

        Raises:
            BackendConnectionError: if not connected and auto-connect fails.
        """

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        """Poll the current status of a previously submitted job.

        job_id must be a composite identifier previously returned by
        submit_job() on this same backend instance — implementations
        should call self._parse_job_id(job_id) to recover the raw,
        provider-specific job id before querying the provider.

        Implementations should call self._ensure_connected() first.

        Raises:
            BackendConnectionError: if not connected and auto-connect fails.
            BackendJobError: if job_id does not belong to this backend.
        """

    @abstractmethod
    def get_result(self, job_id: str) -> JobResult:
        """Retrieve the result of a completed job.

        job_id must be a composite identifier previously returned by
        submit_job() on this same backend instance — see get_job_status().

        Implementations should call self._ensure_connected() first.

        Raises:
            BackendConnectionError: if not connected and auto-connect fails.
            BackendJobError: if job_id does not belong to this backend, or
                the job is not yet complete or failed.
        """

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Request cancellation of a pending or running job.

        job_id must be a composite identifier previously returned by
        submit_job() on this same backend instance — see get_job_status().

        Implementations should call self._ensure_connected() first.

        Raises:
            BackendConnectionError: if not connected and auto-connect fails.
            BackendJobError: if job_id does not belong to this backend.
        """
