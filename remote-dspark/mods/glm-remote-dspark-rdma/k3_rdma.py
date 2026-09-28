"""ctypes wrapper for the k3_rdma raw ibverbs RC RoCEv2 side-channel.

Loads ``libk3rdma.so`` (override with the ``K3RDMA_SO_PATH`` environment
variable, default ``./libk3rdma.so``) and exposes a small :class:`K3Rdma`
class around the C API.

The C library supports a single global endpoint per process, so only one
:class:`K3Rdma` should be open at a time.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

_DEFAULT_SO = os.environ.get("K3RDMA_SO_PATH", "./libk3rdma.so")


class K3Rdma:
    """Thin ctypes binding over ``libk3rdma.so``."""

    def __init__(self, so_path: Optional[str] = None) -> None:
        self.so_path = so_path or _DEFAULT_SO
        if not os.path.exists(self.so_path):
            raise FileNotFoundError(
                f"libk3rdma shared library not found at {self.so_path!r} "
                "(set K3RDMA_SO_PATH or build it with run.sh)"
            )
        self._lib = ctypes.CDLL(self.so_path)
        self._bind()

    # -- prototype setup ------------------------------------------------
    def _bind(self) -> None:
        lib = self._lib

        lib.k3rdma_open.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        lib.k3rdma_open.restype = ctypes.c_int

        lib.k3rdma_connect.argtypes = [ctypes.c_char_p]
        lib.k3rdma_connect.restype = ctypes.c_int

        lib.k3rdma_register.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.k3rdma_register.restype = ctypes.c_void_p

        lib.k3rdma_deregister.argtypes = [ctypes.c_void_p]
        lib.k3rdma_deregister.restype = None

        lib.k3rdma_send.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.k3rdma_send.restype = ctypes.c_int

        lib.k3rdma_post_recv.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.k3rdma_post_recv.restype = ctypes.c_int

        lib.k3rdma_wait_send.argtypes = [ctypes.c_int]
        lib.k3rdma_wait_send.restype = ctypes.c_int

        lib.k3rdma_wait_recv.argtypes = [ctypes.c_int]
        lib.k3rdma_wait_recv.restype = ctypes.c_int

        lib.k3rdma_poll_recv.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.k3rdma_poll_recv.restype = ctypes.c_int

        lib.k3rdma_last_error.argtypes = []
        lib.k3rdma_last_error.restype = ctypes.c_char_p

        lib.k3rdma_close.argtypes = []
        lib.k3rdma_close.restype = None

    # -- API ------------------------------------------------------------
    def open(
        self,
        is_server: bool,
        hca: Optional[str] = None,
        port: int = 1,
        gid_index: int = -1,
    ) -> str:
        """Open the endpoint and return this side's peer-exchange string."""
        hca_b = hca.encode() if hca else None
        buf = ctypes.create_string_buffer(512)
        rc = self._lib.k3rdma_open(
            1 if is_server else 0,
            hca_b,
            int(port),
            int(gid_index),
            buf,
            len(buf),
        )
        if rc != 0:
            raise RuntimeError(f"k3rdma_open failed: {self.last_error()}")
        return buf.value.decode()

    def connect(self, peer_info: str) -> None:
        rc = self._lib.k3rdma_connect(peer_info.encode())
        if rc != 0:
            raise RuntimeError(f"k3rdma_connect failed: {self.last_error()}")

    def register(self, buf) -> int:
        """Register ``buf`` (a ctypes buffer) and return an opaque MR handle."""
        addr = ctypes.addressof(buf)
        if addr == 0:
            raise ValueError("register: null buffer address")
        mr = self._lib.k3rdma_register(ctypes.c_void_p(addr), len(buf))
        if not mr:
            raise RuntimeError(f"k3rdma_register failed: {self.last_error()}")
        return int(mr)

    def deregister(self, mr: int) -> None:
        self._lib.k3rdma_deregister(ctypes.c_void_p(int(mr)))

    def send(self, mr: int, length: int) -> None:
        rc = self._lib.k3rdma_send(ctypes.c_void_p(int(mr)), int(length))
        if rc != 0:
            raise RuntimeError(f"k3rdma_send failed: {self.last_error()}")

    def post_recv(self, mr: int, maxlen: int) -> None:
        rc = self._lib.k3rdma_post_recv(ctypes.c_void_p(int(mr)), int(maxlen))
        if rc != 0:
            raise RuntimeError(f"k3rdma_post_recv failed: {self.last_error()}")

    def wait_send(self, timeout_ms: int) -> None:
        rc = self._lib.k3rdma_wait_send(int(timeout_ms))
        if rc != 0:
            raise RuntimeError(f"k3rdma_wait_send failed: {self.last_error()}")

    def wait_recv(self, timeout_ms: int) -> int:
        rc = self._lib.k3rdma_wait_recv(int(timeout_ms))
        if rc < 0:
            raise RuntimeError(f"k3rdma_wait_recv failed: {self.last_error()}")
        return rc

    def poll_recv(self) -> int | None:
        """Poll the recv CQ once for a completed message.

        Non-blocking. Returns the received byte length if a message completed,
        ``None`` if no completion is pending, and raises on a poll/status
        error. This mirrors ``wait_recv`` but without a deadline or sleep, so
        the caller can implement its own idle loop.
        """
        out_len = ctypes.c_int(0)
        rc = self._lib.k3rdma_poll_recv(ctypes.byref(out_len))
        if rc < 0:
            raise RuntimeError(f"k3rdma_poll_recv failed: {self.last_error()}")
        if rc == 0:
            return None
        return int(out_len.value)

    def last_error(self) -> str:
        raw = self._lib.k3rdma_last_error()
        return raw.decode(errors="replace") if raw else ""

    def close(self) -> None:
        self._lib.k3rdma_close()
