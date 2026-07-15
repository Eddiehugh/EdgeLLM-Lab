"""Portable reference packing for signed INT4 values."""

from __future__ import annotations

import torch


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack two signed INT4 values into each uint8 element."""

    if values.is_floating_point() or values.dtype == torch.bool:
        raise TypeError("INT4 packing requires an integer tensor")
    flat = values.to(torch.int16).reshape(-1)
    if flat.numel() and bool(((flat < -8) | (flat > 7)).any()):
        raise ValueError("INT4 values must be in [-8, 7]")
    encoded = (flat + 8).to(torch.uint8)
    if encoded.numel() % 2:
        encoded = torch.cat(
            [encoded, torch.full((1,), 8, dtype=torch.uint8, device=encoded.device)]
        )
    low = encoded[0::2]
    high = encoded[1::2] << 4
    return low | high


def unpack_int4(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack uint8 storage into signed INT4 values represented as int8."""

    if packed.dtype != torch.uint8:
        raise TypeError("Packed INT4 storage must use torch.uint8")
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("INT4 unpack count must be an integer")
    if count < 0 or count > packed.numel() * 2:
        raise ValueError("Invalid INT4 unpack count")
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    return torch.stack((low, high), dim=-1).reshape(-1)[:count].to(torch.int8)
