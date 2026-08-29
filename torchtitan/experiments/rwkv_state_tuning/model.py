# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-tuning replacements for RWKV7 TimeMix and ChannelMix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from torchtitan.models.rwkv7.model import RWKV7ChannelMix, RWKV7TimeMix
from torchtitan.models.rwkv7.provider import (
    load_flash_rwkv2,
    STATE_TUNING_CHANNELMIX_OPERATORS,
    STATE_TUNING_TIMEMIX_OPERATORS,
)
from torchtitan.protocols.module import Module


# Shape suffixes in this file:
# B: batch lanes, T: tokens per lane, C: model channels, H: recurrent heads,
# K: recurrent head width, P: sequence boundaries, Q: recurrent chunks.


class _RWKV7WKVState(Module):
    """Own the FP32 WKV state as an independently shardable FSDP unit."""

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.initial_wkv_state = nn.Parameter(
            torch.empty(num_heads, head_size, head_size, dtype=torch.float32)
        )

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.initial_wkv_state)

    def forward(self, batch_size: int) -> torch.Tensor:
        return (
            self.initial_wkv_state.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .contiguous()
        )


class RWKV7StateTuningTimeMix(RWKV7TimeMix):
    """TimeMix with trainable lane-broadcast initial shift and WKV state."""

    _provider_operators: ClassVar[tuple[str, ...]] = STATE_TUNING_TIMEMIX_OPERATORS
    _provider_mode: ClassVar[str] = "TimeMix state tuning"

    @dataclass(kw_only=True, slots=True)
    class Config(RWKV7TimeMix.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.initial_shift = nn.Parameter(torch.empty(config.dim, dtype=torch.bfloat16))
        self._wkv_state = _RWKV7WKVState(
            config.dim // config.head_size,
            config.head_size,
        )
        self.register_state_dict_post_hook(self._flatten_wkv_state_on_save)
        self.register_load_state_dict_pre_hook(self._unflatten_wkv_state_on_load)

    @property
    def initial_wkv_state(self) -> nn.Parameter:
        return self._wkv_state.initial_wkv_state

    @staticmethod
    def _flatten_wkv_state_on_save(module, state_dict, prefix, local_metadata) -> None:
        state_dict[f"{prefix}initial_wkv_state"] = state_dict.pop(
            f"{prefix}_wkv_state.initial_wkv_state"
        )

    @staticmethod
    def _unflatten_wkv_state_on_load(module, state_dict, prefix, *args) -> None:
        canonical_key = f"{prefix}initial_wkv_state"
        if canonical_key in state_dict:
            state_dict[f"{prefix}_wkv_state.initial_wkv_state"] = state_dict.pop(
                canonical_key
            )

    @staticmethod
    def _chunk_metadata(
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunks_per_sequence = sequence_length // 16
        sequence_chunk_offsets_P = torch.arange(
            0,
            (batch_size + 1) * chunks_per_sequence,
            chunks_per_sequence,
            dtype=torch.int32,
            device=device,
        )
        sequence_starts_B = (
            torch.arange(batch_size, dtype=torch.int32, device=device) * sequence_length
        )
        chunk_offsets_Q = (
            torch.arange(
                chunks_per_sequence,
                dtype=torch.int32,
                device=device,
            )
            * 16
        )
        chunk_starts_BQ = (
            sequence_starts_B[:, None] + chunk_offsets_Q[None, :]
        ).flatten()
        chunk_ends_BQ = chunk_starts_BQ + 16
        return sequence_chunk_offsets_P, chunk_starts_BQ, chunk_ends_BQ

    def forward(
        self,
        x_BTC: torch.Tensor,
        v_first_BTC: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flash_rwkv2 = load_flash_rwkv2(
            self._provider_operators,
            x_BTC,
            self._provider_mode,
            preloaded=self._flash_rwkv2,
        )
        batch_size, sequence_length, _ = x_BTC.shape
        shift_BC = self.initial_shift.unsqueeze(0).expand(batch_size, -1).contiguous()
        (
            xr_BTC,
            xw_BTC,
            xk_BTC,
            xv_BTC,
            xa_BTC,
            xg_BTC,
            _,
        ) = flash_rwkv2.statetune_tmix_tokenshift_bf16(
            x_BTC.contiguous(),
            shift_BC,
            self.x_r,
            self.x_w,
            self.x_k,
            self.x_v,
            self.x_a,
            self.x_g,
        )
        r_BTC, w_BTC, k_BTC, v_BTC, kk_BTC, ka_BTC, g_BTC = self._project_shifted(
            flash_rwkv2,
            xr_BTC,
            xw_BTC,
            xk_BTC,
            xv_BTC,
            xa_BTC,
            xg_BTC,
            v_first_BTC,
        )
        if self.layer_id == 0:
            v_first_BTC = v_BTC
        assert v_first_BTC is not None

        initial_state_BHKK = self._wkv_state(batch_size)
        sequence_offsets_P, chunk_starts_BQ, chunk_ends_BQ = self._chunk_metadata(
            batch_size,
            sequence_length,
            x_BTC.device,
        )
        recurrent_THK, _, _, _ = flash_rwkv2.statetune_tmix_wkv7_recurrent_fp32io16(
            initial_state_BHKK,
            sequence_offsets_P,
            chunk_starts_BQ,
            chunk_ends_BQ,
            r_BTC.view(-1, self.num_heads, self.head_size),
            w_BTC.view(-1, self.num_heads, self.head_size),
            k_BTC.view(-1, self.num_heads, self.head_size),
            v_BTC.view(-1, self.num_heads, self.head_size),
            kk_BTC.view(-1, self.num_heads, self.head_size),
            ka_BTC.view(-1, self.num_heads, self.head_size),
        )
        recurrent_BTC = recurrent_THK.view(batch_size, sequence_length, self.dim)
        return (
            self._readout(
                flash_rwkv2,
                recurrent_BTC,
                r_BTC,
                k_BTC,
                v_BTC,
                g_BTC,
            ),
            v_first_BTC,
        )


class RWKV7StateTuningChannelMix(RWKV7ChannelMix):
    """ChannelMix with a trainable lane-broadcast initial shift."""

    _provider_operators: ClassVar[tuple[str, ...]] = STATE_TUNING_CHANNELMIX_OPERATORS
    _provider_mode: ClassVar[str] = "ChannelMix state tuning"

    @dataclass(kw_only=True, slots=True)
    class Config(RWKV7ChannelMix.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.initial_shift = nn.Parameter(torch.empty(config.dim, dtype=torch.bfloat16))

    def forward(self, x_BTC: torch.Tensor) -> torch.Tensor:
        flash_rwkv2 = load_flash_rwkv2(
            self._provider_operators,
            x_BTC,
            self._provider_mode,
            preloaded=self._flash_rwkv2,
        )
        shift_BC = (
            self.initial_shift.unsqueeze(0).expand(x_BTC.shape[0], -1).contiguous()
        )
        output_BTC, _ = flash_rwkv2.statetune_cmix_bf16(
            x_BTC.contiguous(),
            shift_BC,
            self.x_k,
            self.key.weight,
            self.value.weight,
        )
        return output_BTC


__all__ = ["RWKV7StateTuningChannelMix", "RWKV7StateTuningTimeMix"]
