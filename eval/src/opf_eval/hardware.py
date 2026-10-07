"""Accelerator facts shared by `opf_eval.nb.gpu_summary` and `opf_eval.review.clef`.

Both callers report memory in GiB (1024³ bytes) because torch and psutil count
bytes and the Clef-flash thresholds are written in GiB. Keeping the conversion
and the bfloat16 check in one place means the notebook's GPU summary and
`clef.plan_load` always describe the same card the same way. An L4 reports
about 22.0 to 22.5 GiB and a T4 about 14.7 to 15 GiB.

torch and psutil are imported inside each function, so importing this module
costs nothing.
"""

from __future__ import annotations

GIB = 1024**3


def to_gib(n_bytes: float) -> float:
    """Convert a byte count to GiB (1024³ bytes)."""
    return n_bytes / GIB


def cuda_memory_gib(index: int = 0) -> float:
    """Total memory of CUDA device `index` in GiB.

    The caller checks that CUDA is available. torch raises when it is not.
    """
    import torch

    return to_gib(torch.cuda.get_device_properties(index).total_memory)


def cuda_native_bf16(index: int = 0) -> bool:
    """Whether CUDA device `index` runs bfloat16 in hardware.

    Native support starts at compute capability 8.0 (A100, L4, H100). A T4
    (7.5) and a V100 (7.0) do not have it. `torch.cuda.is_bf16_supported()`
    alone is not enough because by default it also counts emulated bfloat16
    and so says True on a T4.
    """
    import torch

    try:
        return int(torch.cuda.get_device_capability(index)[0]) >= 8
    except Exception:  # noqa: BLE001 — fall back to torch's own check
        pass
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # older torch has no including_emulation argument
        return bool(torch.cuda.is_bf16_supported())


def unified_memory_gib() -> float | None:
    """Total system memory in GiB, which Apple Silicon shares with the GPU.

    Returns None when psutil is not installed.
    """
    try:
        import psutil
    except ImportError:
        return None
    return to_gib(psutil.virtual_memory().total)
