"""
qos_hal: Hardware Abstraction Layer for Quantum OS.

This package defines a backend-agnostic interface (Backend ABC) for
interacting with quantum hardware providers. The IBM implementation
lives in qos_hal.ibm, and registers itself with qos_hal.registry as
"ibm" as a side effect of being imported (import qos_hal.ibm, or
get_backend() will raise KeyError since nothing is registered yet).
"""

from qos_hal.backend import Backend
from qos_hal.cached_backend import CachedBackend
from qos_hal.client import QosHalClient, QosHalClientError, RemoteBackendError
from qos_hal.registry import available_backends, get_backend, register

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "CachedBackend",
    "get_backend",
    "register",
    "available_backends",
    "QosHalClient",
    "QosHalClientError",
    "RemoteBackendError",
]
