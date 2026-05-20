from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuBatchProfile:
    max_memory_gb: float
    batch_max_items: int
    batch_max_tokens: int


NER_GPU_BATCH_PROFILES = (
    GpuBatchProfile(max_memory_gb=8.0, batch_max_items=4, batch_max_tokens=1024),
    GpuBatchProfile(max_memory_gb=16.0, batch_max_items=12, batch_max_tokens=3072),
    GpuBatchProfile(max_memory_gb=float("inf"), batch_max_items=24, batch_max_tokens=6144),
)


def detect_cuda_total_memory_gb() -> float | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:
        return None
    return float(properties.total_memory) / float(1024**3)


def resolve_ner_gpu_batch_profile(memory_gb: float | None = None) -> GpuBatchProfile:
    detected_memory_gb = memory_gb if memory_gb is not None else detect_cuda_total_memory_gb()
    if detected_memory_gb is None:
        return NER_GPU_BATCH_PROFILES[-1]
    for profile in NER_GPU_BATCH_PROFILES:
        if detected_memory_gb <= profile.max_memory_gb:
            return profile
    return NER_GPU_BATCH_PROFILES[-1]
