# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 specialization of the generic FSDP compute-weight lifecycle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchao.prototype.mx_formats.kernels import (
    triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
)

from .._fsdp_weight import _ShardedFSDPWeight


# Everything here is internal to the MXFP8 component; nothing is re-exported.
__all__: list[str] = []

_MXFP8_WEIGHT_TILE_SIZE = 32


@dataclass(frozen=True, slots=True)
class _MXFP8LinearOperands:
    """The independent MXFP8 tensors owned by one FSDP unshard lifetime."""

    q_weight_dgrad_NK: torch.Tensor  # noqa: N815
    s_weight_fprop_blocked: torch.Tensor
    s_weight_dgrad_blocked: torch.Tensor

    @property
    def q_weight_fprop_KN(self) -> torch.Tensor:  # noqa: N802
        return self.q_weight_dgrad_NK.t()


def _quantize_mxfp8_weight(weight_NK: torch.Tensor) -> _MXFP8LinearOperands:
    """Quantize a BF16 weight using fixed square 32x32 scale tiles."""
    if weight_NK.ndim != 2:
        raise ValueError(
            "MXFP8 32x32 weight quantization requires a 2D weight, "
            f"got {weight_NK.ndim} dimensions."
        )
    if weight_NK.dtype != torch.bfloat16:
        raise ValueError(
            "MXFP8 32x32 weight quantization requires BF16 weights, "
            f"got {weight_NK.dtype}."
        )
    if any(size % _MXFP8_WEIGHT_TILE_SIZE for size in weight_NK.shape):
        raise ValueError(
            "MXFP8 32x32 weight quantization requires both matrix dimensions "
            f"divisible by {_MXFP8_WEIGHT_TILE_SIZE}, got {tuple(weight_NK.shape)}."
        )
    (
        q_weight_dgrad_NK,
        s_weight_fprop_blocked,
        s_weight_dgrad_blocked,
    ) = triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale(weight_NK.contiguous())
    return _MXFP8LinearOperands(
        q_weight_dgrad_NK=q_weight_dgrad_NK,
        s_weight_fprop_blocked=s_weight_fprop_blocked,
        s_weight_dgrad_blocked=s_weight_dgrad_blocked,
    )


class _LinearShardedWeightWithMXFP8Compute(_ShardedFSDPWeight):
    """The persistent BF16 linear parameter; quantizes to MXFP8 on unshard.

    This is the sharded state only. FSDP shards, all-gathers, reduces
    gradients into, and checkpoints this BF16 parameter. The MXFP8 operands it
    produces live on a ``_ComputeFSDPWeight`` for one unshard lifetime; that
    holder is generic, so quantization is the only thing a format supplies.
    """

    def _build_compute_representation(
        self,
        logical_weight: torch.Tensor,
        out: _MXFP8LinearOperands | None = None,
    ) -> _MXFP8LinearOperands:
        operands = _quantize_mxfp8_weight(logical_weight)
        if out is None:
            return operands
        out.q_weight_dgrad_NK.copy_(operands.q_weight_dgrad_NK)
        out.s_weight_fprop_blocked.copy_(operands.s_weight_fprop_blocked)
        out.s_weight_dgrad_blocked.copy_(operands.s_weight_dgrad_blocked)
        return out
