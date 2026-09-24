"""
Backend registry — maps a short provider name (e.g. "ibm") to its
Backend implementation class, so callers (and eventually the daemon)
can do get_backend("ibm") without importing IBMBackend directly.

Design: register() is called once per backend class, typically at import
time of that backend's module. This keeps qos_hal.registry itself free of
any provider-specific imports — it never needs to know IBMBackend exists.
"""

from qos_hal.backend import Backend

_REGISTRY: dict[str, type[Backend]] = {}


def register(name: str, backend_cls: type[Backend]) -> None:
    """Register a Backend subclass under a short lookup name (e.g. "ibm").

    Raises:
        TypeError: if backend_cls isn't actually a Backend subclass.
        ValueError: if name is already registered to a different class
            (re-registering the SAME class under the same name is allowed,
            so re-importing a module doesn't break things).
    """
    if not (isinstance(backend_cls, type) and issubclass(backend_cls, Backend)):
        raise TypeError(f"{backend_cls!r} is not a Backend subclass")

    key = name.lower()
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not backend_cls:
        raise ValueError(
            f"Backend name {name!r} is already registered to {existing!r}, "
            f"cannot re-register to {backend_cls!r}"
        )
    _REGISTRY[key] = backend_cls


def get_backend(name: str) -> Backend:
    """Instantiate a fresh Backend by its registered name.

    Raises:
        KeyError: if no backend is registered under this name.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"No backend registered as {name!r}. Available: {available}")
    return _REGISTRY[key]()


def available_backends() -> list[str]:
    """List all currently registered backend names."""
    return sorted(_REGISTRY)
