from __future__ import annotations

import threading

from .client import GlinerClient


_CLIENT: GlinerClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_gliner_client() -> GlinerClient:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = GlinerClient()
    return _CLIENT


def reset_gliner_client() -> None:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.stop()
        _CLIENT = None
