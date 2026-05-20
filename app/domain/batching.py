from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NerTaskSignature:
    labels: tuple[str, ...]
    threshold: float


def effective_max_items(*, configured_limit: int, adaptive_limit: int | None) -> int:
    resolved_limit = adaptive_limit if adaptive_limit is not None else configured_limit
    return max(1, resolved_limit)


def effective_max_tokens(*, configured_limit: int) -> int:
    return max(1, configured_limit)
