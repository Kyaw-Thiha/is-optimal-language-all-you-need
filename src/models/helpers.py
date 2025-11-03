"""Helper utilities shared across model runner implementations."""

from __future__ import annotations

from typing import Optional, Union

import torch

DeviceLike = Union[str, int, torch.device]


def resolve_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    """Translate a dtype string into the corresponding torch dtype."""

    if dtype_name is None:
        return None

    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
    }
    try:
        return aliases[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype alias '{dtype_name}'.") from exc


def coerce_device(device: DeviceLike) -> torch.device:
    """Normalize device specifications into torch.device instances."""

    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, int):
        if device < 0:
            raise ValueError("CUDA device index must be non-negative.")
        return torch.device(f"cuda:{device}")
    raise TypeError(f"Unsupported device specifier: {device!r}")
