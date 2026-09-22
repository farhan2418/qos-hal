"""
Daemon server: accepts connections on a Unix domain socket, one thread per
connection, newline-delimited JSON request/response framing.

Design decisions (settled during architecture discussion):
  - One shared Backend instance (typically a CachedBackend wrapping
    IBMBackend) is created once at daemon startup and used by every
    client connection — not one Backend per client.
  - Thread-per-connection concurrency. A single lock serializes all calls
    into the shared Backend, since the underlying Qiskit SDK objects are
    not assumed to be thread-safe. IBM calls are network-bound, so this
    lock does not block other threads from framing/parsing their own
    messages — only from the moment they actually call into Backend.
  - Wire format: one JSON object per line, both directions.
      Request:  {"method": "...", "params": {...}}
      Response: {"ok": true, "result": ...}
               | {"ok": false, "error": {"type": "...", "message": "..."}}

This module deliberately knows nothing about what any given method NAME
means — that translation lives in dispatch.py. server.py only knows how
to accept connections, read/write framed JSON lines, and route each
parsed request to a dispatcher function.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from collections.abc import Callable

logger = logging.getLogger("qos_hal.daemon")


class DaemonServer:
    """Accepts client connections on a Unix domain socket.

    dispatch_fn receives a parsed request dict ({"method": ..., "params": ...})
    and must return a JSON-serializable result, or raise an exception on
    failure — this class handles turning either outcome into the correct
    {"ok": ...} response envelope.
    """

    def __init__(self, socket_path: str, dispatch_fn: Callable[[dict], object]):
        self._socket_path = socket_path
        self._dispatch_fn = dispatch_fn
        self._backend_lock = threading.Lock()
        self._server_sock: socket.socket | None = None
        self._shutdown = threading.Event()

    def serve_forever(self) -> None:
        """Bind the socket and accept connections until stop() is called."""
        # A stale socket file from a previous unclean shutdown must be
        # removed first, or bind() fails with "address already in use".
        if os.path.exists(self._socket_path):
            os.remove(self._socket_path)

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(self._socket_path)
        self._server_sock.listen()
        logger.info("Listening on %s", self._socket_path)

        try:
            while not self._shutdown.is_set():
                try:
                    conn, _ = self._server_sock.accept()
                except OSError:
                    break  # socket closed by stop()
                thread = threading.Thread(
                    target=self._handle_connection, args=(conn,), daemon=True
                )
                thread.start()
        finally:
            self._cleanup()

    def stop(self) -> None:
        self._shutdown.set()
        if self._server_sock is not None:
            self._server_sock.close()

    def _cleanup(self) -> None:
        if os.path.exists(self._socket_path):
            os.remove(self._socket_path)

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            buffer = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break  # client closed the connection
                buffer += chunk

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    response = self._handle_line(line)
                    conn.sendall(json.dumps(response).encode("utf-8") + b"\n")

    def _handle_line(self, line: bytes) -> dict:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            return {"ok": False, "error": {"type": "ProtocolError", "message": f"Invalid JSON: {e}"}}

        if not isinstance(request, dict) or "method" not in request:
            return {"ok": False, "error": {"type": "ProtocolError", "message": "Request must be a JSON object with a 'method' key"}}

        try:
            # All Backend calls are serialized through one lock, since the
            # underlying Qiskit SDK objects are not assumed thread-safe.
            # Framing/parsing above happens outside this lock, so slow
            # clients don't block other connections from being read.
            with self._backend_lock:
                result = self._dispatch_fn(request)
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": {"type": type(e).__name__, "message": str(e)}}
