"""
Daemon entry point. Builds the one shared Backend instance for the
daemon's lifetime, wires it into a dispatcher, and starts the Unix
socket server.

Run directly for local testing:  python -m qos_hal.daemon
(systemd unit wiring comes later — see project roadmap.)
"""

import logging
import sys

import qos_hal.ibm  # noqa: F401 — import registers "ibm" with the registry
from qos_hal.cached_backend import CachedBackend
from qos_hal.daemon.dispatch import make_dispatcher
from qos_hal.daemon.server import DaemonServer
from qos_hal.registry import get_backend

DEFAULT_SOCKET_PATH = "/tmp/qos_hal.sock"


def main(socket_path: str = DEFAULT_SOCKET_PATH) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # One shared Backend for the daemon's whole lifetime — every client
    # request is dispatched against this same instance (see design notes
    # in server.py / dispatch.py).
    backend = CachedBackend(get_backend("ibm"))
    dispatcher = make_dispatcher(backend)
    server = DaemonServer(socket_path, dispatcher)

    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOCKET_PATH)
