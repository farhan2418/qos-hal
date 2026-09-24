"""
CachedBackend — a generic, provider-agnostic caching decorator for any
Backend.

Design decisions (settled during architecture discussion):
  - Separate TTLs for topology vs. calibration: topology changes far less
    often than calibration in practice, so they're allowed to expire on
    different schedules.
  - Cache freshness is measured with time.monotonic(), never wall-clock
    time, so it's immune to system clock adjustments (NTP sync, manual
    changes, DST).
  - No force-refresh parameter (yet). This wrapper is intended for the
    client / service layer, which has no reason to reach past the cache.
    A force-refresh option can be added later without breaking this
    contract, once a caller with that specific need exists.
  - CachedBackend IS a Backend (not just a plain wrapper class), so it can
    be handed to anything that expects "a Backend" — the registry, a
    daemon, etc. — with caching entirely invisible to the caller.
  - Every method other than get_topology()/get_calibration() is a pure
    delegation to the wrapped backend; no other behavior is added or
    changed.
"""

from __future__ import annotations

import time

from qos_hal.backend import (
    Backend,
    CalibrationData,
    JobResult,
    JobStatus,
    Topology,
)

DEFAULT_TOPOLOGY_TTL_SECONDS = 3600.0       # topology rarely changes
DEFAULT_CALIBRATION_TTL_SECONDS = 300.0     # calibration drifts faster


class CachedBackend(Backend):
    """Wraps any Backend, adding TTL-based caching for topology/calibration.

    Usage:
        backend = CachedBackend(IBMBackend())
        backend.get_topology()      # fetches live, caches it
        backend.get_topology()      # returns cached value, no network call
        # ... after topology_ttl_seconds have elapsed ...
        backend.get_topology()      # fetches live again, re-caches
    """

    def __init__(
        self,
        wrapped: Backend,
        topology_ttl_seconds: float = DEFAULT_TOPOLOGY_TTL_SECONDS,
        calibration_ttl_seconds: float = DEFAULT_CALIBRATION_TTL_SECONDS,
    ):
        self._wrapped = wrapped
        self._topology_ttl = topology_ttl_seconds
        self._calibration_ttl = calibration_ttl_seconds

        # Each cache entry is (value, monotonic_timestamp_when_cached).
        self._topology_cache: tuple[Topology, float] | None = None
        self._calibration_cache: tuple[CalibrationData, float] | None = None

    # ---- Lifecycle: pure delegation ---------------------------------

    @property
    def is_connected(self) -> bool:
        return self._wrapped.is_connected

    @property
    def _identifier(self) -> str:
        return self._wrapped._identifier

    def connect(self, backend_name: str | None = None) -> None:
        self._wrapped.connect(backend_name)

    def disconnect(self) -> None:
        # Dropping cached data on disconnect is deliberate: a fresh
        # connect() (possibly to a different backend_name) must never
        # serve stale data cached under the previous connection.
        self._wrapped.disconnect()
        self._topology_cache = None
        self._calibration_cache = None

    # ---- Read-only device state: the actual caching logic -------------

    def get_topology(self) -> Topology:
        if self._topology_cache is not None:
            value, cached_at = self._topology_cache
            if time.monotonic() - cached_at < self._topology_ttl:
                return value

        value = self._wrapped.get_topology()
        self._topology_cache = (value, time.monotonic())
        return value

    def get_calibration(self) -> CalibrationData:
        if self._calibration_cache is not None:
            value, cached_at = self._calibration_cache
            if time.monotonic() - cached_at < self._calibration_ttl:
                return value

        value = self._wrapped.get_calibration()
        self._calibration_cache = (value, time.monotonic())
        return value

    def list_available_backends(self) -> list[str]:
        # Deliberately NOT cached: this is a distinct concern from
        # topology/calibration TTLs, and the set of available backends
        # is queried rarely enough that live delegation is fine.
        return self._wrapped.list_available_backends()

    # ---- Execution: pure delegation, never cached ----------------------

    def submit_job(self, circuit) -> str:
        return self._wrapped.submit_job(circuit)

    def get_job_status(self, job_id: str) -> JobStatus:
        return self._wrapped.get_job_status(job_id)

    def get_result(self, job_id: str) -> JobResult:
        return self._wrapped.get_result(job_id)

    def cancel_job(self, job_id: str) -> None:
        self._wrapped.cancel_job(job_id)
