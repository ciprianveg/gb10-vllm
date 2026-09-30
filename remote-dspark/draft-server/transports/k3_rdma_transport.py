"""Raw-verbs RDMA transport for the K3 remote-draft side-channel.

This replaces the unreliable NCCL P2P side-channel with the validated
``libk3rdma`` raw ibverbs RC RoCEv2 transport (see ``k3_rdma.c`` /
``k3_rdma.py``). The TCPStore that already backs the rendezvous
(``StatelessProcessGroup``) is reused ONLY to exchange the small ibverbs
peer-info strings; every tensor payload rides the RDMA queue pair.

No GPUDirect on GB10: the GB10 (unified memory) end cannot do GPUDirect
RDMA, so every payload is staged through host memory. The client copies
D2H into a registered send buffer and H2D out of a registered recv buffer;
the 3090 end could do GPUDirect but the shared code path stays host-staged
for symmetry. The two registered buffers are allocated once in ``start()``.

Import path: both ``k3_rdma.py`` and this module are deployed next to the
installed vLLM package and imported as ``vllm.k3_rdma_transport`` /
``vllm.k3_rdma``. ``K3RDMA_SO_PATH`` overrides the shared-library path;
otherwise it is resolved next to this module.
"""

from __future__ import annotations

import ctypes
import os
import time
from typing import Any

from vllm.logger import init_logger

from vllm.k3_rdma import K3Rdma

logger = init_logger(__name__)

_SERVER_INFO_KEY = "k3rdma_server_info"
_CLIENT_INFO_KEY = "k3rdma_client_info"
# Client-generation signal: the verifier bumps this key each time it (re)creates
# its StatelessProcessGroup. The T1 server polls it while idle to detect a dead
# or reconnected client and reset its rendezvous instead of blocking on a dead
# QP for the full recv window.
_CLIENT_GEN_KEY = "k3rdma_client_gen"

# Default send/recv staging buffer size (bytes). Callers that know the
# worst-case request size (context rows * context width) should pass a
# larger ``max_msg``.
_DEFAULT_MAX_MSG = 64 << 20

# Default timeout for a single send completion.
_SEND_TIMEOUT_MS = 30000


def _resolve_so_path() -> str:
    """Resolve ``libk3rdma.so``: env override, else next to this module."""
    override = os.environ.get("K3RDMA_SO_PATH")
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "libk3rdma.so")


def _store_target(store: Any) -> Any:
    """Return the underlying c10d store from a StatelessProcessGroup or store."""
    return store.store if hasattr(store, "store") else store


def _store_set(store: Any, key: str, value: str | bytes) -> None:
    raw = value.encode() if isinstance(value, str) else value
    _store_target(store).set(key, raw)


def _store_get(store: Any, key: str) -> bytes:
    raw = _store_target(store).get(key)
    return raw if isinstance(raw, bytes) else bytes(raw)


def _reset_rendezvous(store: Any) -> None:
    """Drop both peer-info keys so the next get blocks for a fresh peer.

    The TCPStore rendezvous is persistent across cluster boots, so a
    re-established server would otherwise read the *previous* client's QP
    info and connect to a dead queue pair: every response then fails with
    ``send completion status 12 (transport retry counter exceeded)``. Deleting
    both keys forces the reconnecting client to republish its QP before the
    server reads it, and forces the server to republish its new QP before the
    client reads it. Best-effort: the keys are absent on the first boot.
    """
    target = _store_target(store)
    for key in (_CLIENT_INFO_KEY, _SERVER_INFO_KEY):
        try:
            target.delete_key(key)
        except Exception:  # noqa: BLE001 - absent on the first boot
            pass


class _K3RdmaEndpoint:
    """Shared open/connect/register plumbing for the two endpoint roles."""

    def __init__(
        self,
        store: Any,
        hca: str,
        gid_index: int,
        port: int = 1,
        max_msg: int = _DEFAULT_MAX_MSG,
        send_max: int | None = None,
        recv_max: int | None = None,
    ) -> None:
        self.store = store
        self.hca = hca
        self.gid_index = int(gid_index)
        self.port = int(port)
        self.max_msg = int(max_msg)
        # ``max_msg`` is the legacy single-size knob; ``send_max``/``recv_max``
        # let callers size the two staging buffers independently (requests are
        # large, responses are tiny). Both default to ``max_msg``.
        self.send_max = int(max_msg if send_max is None else send_max)
        self.recv_max = int(max_msg if recv_max is None else recv_max)
        if self.max_msg <= 0:
            raise ValueError("max_msg must be positive")
        if self.send_max <= 0:
            raise ValueError("send_max must be positive")
        if self.recv_max <= 0:
            raise ValueError("recv_max must be positive")
        self.rdma: K3Rdma | None = None
        # Host-staging buffers (no GPUDirect on GB10). Registered once in
        # start() and reused for every message.
        self._send_buf = ctypes.create_string_buffer(self.send_max)
        self._recv_buf = ctypes.create_string_buffer(self.recv_max)
        self._send_mr: int | None = None
        self._recv_mr: int | None = None
        # Tracks whether a recv WR is currently posted on the QP. ``_recv_bytes``
        # always posts and consumes its own WR, but ``try_recv_request`` leaves
        # its WR pending across idle polls so a late client send is still
        # received without exhausting the RQ (max_recv_wr=16).
        self._recv_posted: bool = False

    # -- lifecycle ------------------------------------------------------
    def _open_and_connect(self, is_server: bool) -> None:
        rdma = K3Rdma(_resolve_so_path())
        # Track the endpoint before open() so a failure mid-handshake can
        # still tear down the (possibly partially opened) C endpoint.
        self.rdma = rdma
        local_info = rdma.open(is_server, self.hca, self.port, self.gid_index)
        if is_server:
            # Re-enter the rendezvous cleanly: clear the peer-info keys so
            # the blocking get below cannot return the dead client QP left by
            # a previous cluster boot (see _reset_rendezvous). This is the
            # server-side half of the QP reset; the fresh QP is the one just
            # opened above, and the client re-publishes its QP on reconnect.
            _reset_rendezvous(self.store)
            _store_set(self.store, _SERVER_INFO_KEY, local_info)
            peer_info = _store_get(self.store, _CLIENT_INFO_KEY).decode()
        else:
            peer_info = _store_get(self.store, _SERVER_INFO_KEY).decode()
            _store_set(self.store, _CLIENT_INFO_KEY, local_info)
        rdma.connect(peer_info)
        self._send_mr = rdma.register(self._send_buf)
        self._recv_mr = rdma.register(self._recv_buf)
        logger.info(
            "K3 RDMA endpoint ready (is_server=%s, hca=%s, gid_index=%d, "
            "port=%d, max_msg=%d)",
            is_server,
            self.hca,
            self.gid_index,
            self.port,
            self.max_msg,
        )

    def close(self) -> None:
        # The C library is a single global endpoint, so ``rdma.close()`` MUST
        # always run even if deregistration raises; otherwise the channel can
        # never be re-established. Capture and null the instance attrs first
        # so a concurrent/duplicate close() is a no-op.
        rdma = self.rdma
        send_mr = self._send_mr
        recv_mr = self._recv_mr
        self.rdma = None
        self._send_mr = None
        self._recv_mr = None
        self._recv_posted = False
        if rdma is None:
            return
        if send_mr is not None:
            try:
                rdma.deregister(send_mr)
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.warning("K3 RDMA send MR deregister failed", exc_info=True)
        if recv_mr is not None:
            try:
                rdma.deregister(recv_mr)
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.warning("K3 RDMA recv MR deregister failed", exc_info=True)
        try:
            rdma.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.warning("K3 RDMA endpoint close failed", exc_info=True)

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _remaining_ms(deadline: float) -> int:
        """Milliseconds left until ``deadline`` (clamped to >= 1 ms)."""
        return max(1, int((deadline - time.monotonic()) * 1000.0))

    def _teardown_on_error(self, exc: Exception, context: str) -> RuntimeError:
        """Best-effort close the endpoint and wrap ``exc`` for the caller.

        A timed-out or failed WR stays posted on the QP; leaving it there
        makes the next exchange consume a stale completion. Tearing the
        endpoint down forces a clean re-establish instead.
        """
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.warning("K3 RDMA teardown after %s failed", context, exc_info=True)
        return RuntimeError(f"K3 RDMA {context} failed: {exc}")

    # -- data plane -----------------------------------------------------
    def _send_bytes(self, data: bytes, timeout_ms: int = _SEND_TIMEOUT_MS) -> None:
        if self.rdma is None or self._send_mr is None:
            raise RuntimeError("K3 RDMA endpoint is not started")
        if len(data) > self.send_max:
            raise RuntimeError(
                f"K3 RDMA message is {len(data)} bytes, exceeds the registered "
                f"send buffer ({self.send_max} bytes)"
            )
        # D2H host staging: copy the Python bytes into the registered MR.
        ctypes.memmove(self._send_buf, data, len(data))
        self.rdma.send(self._send_mr, len(data))
        self.rdma.wait_send(timeout_ms)

    def _recv_bytes(self, timeout_ms: int) -> bytes:
        if self.rdma is None or self._recv_mr is None:
            raise RuntimeError("K3 RDMA endpoint is not started")
        if self._recv_posted:
            # A recv WR is already posted (pre-posted credit from the
            # previous iteration): consume it instead of posting anew.
            # This keeps a recv ALWAYS posted while the endpoint is alive,
            # so inbound messages never hit an RNR race. Strict ping-pong
            # means one credit suffices.
            self._recv_posted = False
        else:
            self.rdma.post_recv(self._recv_mr, self.recv_max)
        byte_len = self.rdma.wait_recv(timeout_ms)
        if byte_len > self.recv_max:
            raise RuntimeError(
                f"K3 RDMA received {byte_len} bytes, exceeds the registered "
                f"recv buffer ({self.recv_max} bytes)"
            )
        # Return exactly byte_len bytes. NOTE: do NOT use ``self._recv_buf.raw``
        # here — ``.raw`` materialises the ENTIRE registered buffer (hundreds
        # of MB) as a bytes object before slicing, which dominated the round
        # trip. ``ctypes.string_at`` copies only byte_len bytes.
        return ctypes.string_at(ctypes.addressof(self._recv_buf), byte_len)


class K3RdmaServer(_K3RdmaEndpoint):
    """T1-side (rank 1) RDMA request/response endpoint."""

    def _prepost_recv_credit(self) -> None:
        """Post one recv WR ahead of the next request.

        Called at startup and before doing any other work in
        ``send_response`` so a recv is ALWAYS posted while this endpoint
        is alive (see ``_recv_bytes``). Strict ping-pong needs one credit.
        """
        if self.rdma is None or self._recv_mr is None:
            raise RuntimeError("K3 RDMA endpoint is not started")
        if not self._recv_posted:
            self.rdma.post_recv(self._recv_mr, self.recv_max)
            self._recv_posted = True

    def start(self) -> None:
        """Open, publish this side's info, connect to the client, register."""
        try:
            self._open_and_connect(is_server=True)
        except Exception:
            self.close()
            raise
        self._prepost_recv_credit()

    def recv_request(self, timeout_ms: int = 30000) -> bytes:
        """Block for one request message and return its bytes."""
        logger.info("K3 RDMA server recv_request begin")
        try:
            data = self._recv_bytes(timeout_ms)
        except Exception as exc:
            logger.warning("K3 RDMA server recv_request failed", exc_info=True)
            raise self._teardown_on_error(exc, "recv_request") from exc
        logger.info("K3 RDMA server recv_request got %d bytes", len(data))
        return data

    def try_recv_request(self, timeout_ms: int) -> bytes | None:
        """Poll for one request message; return ``None`` if idle until timeout.

        Unlike :meth:`recv_request`, this never blocks for the full recv window.
        It posts one recv WR (if none is pending), then polls the CQ with a
        short sleep until a message arrives or ``timeout_ms`` elapses. An idle
        (``None``) return leaves the posted WR pending so a late client send is
        still received; the caller is expected to re-invoke this method until a
        message arrives or it decides to tear the endpoint down. A real
        transport error still raises so the caller can tear down and
        re-establish.
        """
        if self.rdma is None or self._recv_mr is None:
            raise RuntimeError("K3 RDMA endpoint is not started")
        if not self._recv_posted:
            self.rdma.post_recv(self._recv_mr, self.recv_max)
            self._recv_posted = True
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            byte_len = self.rdma.poll_recv()
            if byte_len is not None:
                self._recv_posted = False
                if byte_len > self.recv_max:
                    raise RuntimeError(
                        f"K3 RDMA received {byte_len} bytes, exceeds the "
                        f"registered recv buffer ({self.recv_max} bytes)"
                    )
                return ctypes.string_at(ctypes.addressof(self._recv_buf), byte_len)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.001)

    def send_response(self, data: bytes) -> None:
        """Stage and send one response message."""
        logger.info("K3 RDMA server send_response %d bytes", len(data))
        try:
            # Re-arm the recv credit BEFORE sending: the client's next
            # request is triggered by this response, so posting first
            # closes the RNR window by construction.
            self._prepost_recv_credit()
            self._send_bytes(data)
        except Exception as exc:
            logger.warning("K3 RDMA server send_response failed", exc_info=True)
            raise self._teardown_on_error(exc, "send_response") from exc


class K3RdmaClient(_K3RdmaEndpoint):
    """Cluster-side (rank 0) RDMA request/response endpoint."""

    def start(self) -> None:
        """Open, publish this side's info, connect to the server, register."""
        try:
            self._open_and_connect(is_server=False)
        except Exception:
            self.close()
            raise

    def exchange(
        self,
        request: bytes,
        response_len: int,
        timeout_ms: int = 30000,
    ) -> bytes:
        """Send ``request`` and return exactly ``response_len`` response bytes."""
        if response_len < 0:
            raise ValueError("response_len must be non-negative")
        if response_len > self.recv_max:
            raise RuntimeError(
                f"K3 RDMA response of {response_len} bytes exceeds the "
                f"registered recv buffer ({self.recv_max} bytes)"
            )
        if self.rdma is None or self._recv_mr is None:
            raise RuntimeError("K3 RDMA endpoint is not started")
        # ONE deadline for the whole exchange: send + recv share the caller's
        # timeout budget instead of each waiting the full timeout (2x worst
        # case tripped the engine's 60s shm_broadcast watchdog).
        deadline = time.monotonic() + timeout_ms / 1000.0
        try:
            # Post the response recv BEFORE sending so the server's response
            # send never hits an RNR (the QP has finite rnr_retry).
            self.rdma.post_recv(self._recv_mr, response_len)
            self._send_bytes(request, self._remaining_ms(deadline))
            byte_len = self.rdma.wait_recv(self._remaining_ms(deadline))
        except Exception as exc:
            logger.warning("K3 RDMA client exchange failed", exc_info=True)
            raise self._teardown_on_error(exc, "exchange") from exc
        if byte_len != response_len:
            logger.warning(
                "K3 RDMA client exchange length mismatch: expected=%d got=%d",
                response_len,
                byte_len,
            )
            raise self._teardown_on_error(
                RuntimeError(
                    f"expected {response_len} response bytes, got {byte_len}"
                ),
                "exchange",
            )
        return ctypes.string_at(ctypes.addressof(self._recv_buf), byte_len)
