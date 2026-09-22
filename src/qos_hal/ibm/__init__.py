"""IBM backend implementation for qos_hal."""

from qos_hal.ibm.ibm_backend import IBMBackend
from qos_hal.registry import register

register("ibm", IBMBackend)

__all__ = ["IBMBackend"]