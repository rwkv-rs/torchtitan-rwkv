# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-tuning replacements for RWKV-7 TimeMix and ChannelMix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from torchtitan.components.lora import _get_lora_cls
from torchtitan.models.common.linear import Linear
from torchtitan.models.rwkv7.model import RWKVAttention, RWKVFeedForward, RWKVModel
from torchtitan.models.rwkv7.provider import (
    STATE_TUNING_ATTENTION_OPERATORS,
    STATE_TUNING_FEED_FORWARD_OPERATORS,
)
from torchtitan.protocols.module import Module


# Shape suffixes in this file:
# B: batch lanes, T: tokens per lane, N: total tokens, C: model channels,
# H: recurrent heads, K: recurrent head width, P: sequence boundaries,
# Q: chunks per sequence, R: total recurrent chunks.


class _WKVState(Module):
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


class RWKVStateTuningAttention(RWKVAttention):
    """Attention with trainable lane-broadcast shift and WKV state."""

    _provider_operators: ClassVar[tuple[str, ...]] = STATE_TUNING_ATTENTION_OPERATORS
    _provider_mode: ClassVar[str] = "attention state tuning"

    @dataclass(kw_only=True, slots=True)
    class Config(RWKVAttention.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.initial_attention_shift = nn.Parameter(
            torch.empty(config.dim, dtype=torch.bfloat16)
        )
        self._wkv_state = _WKVState(
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
    def training_metadata(
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
        chunk_token_starts_BQ = sequence_starts_B[:, None] + chunk_offsets_Q[None, :]
        chunk_token_starts_R = chunk_token_starts_BQ.flatten()
        chunk_token_ends_R = chunk_token_starts_R + 16
        return sequence_chunk_offsets_P, chunk_token_starts_R, chunk_token_ends_R

    def forward(
        self,
        hidden_states_BTC: torch.Tensor,
        v_first_BTC: torch.Tensor | None = None,
        attention_shift_BC: torch.Tensor | None = None,
        wkv_state_BHKK: torch.Tensor | None = None,
        training_metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del attention_shift_BC, wkv_state_BHKK, training_metadata
        batch_size, sequence_length, _ = hidden_states_BTC.shape
        attention_shift_BC = (
            self.initial_attention_shift.unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        wkv_state_BHKK = self._wkv_state(batch_size)
        (
            sequence_chunk_offsets_P,
            chunk_token_starts_R,
            chunk_token_ends_R,
        ) = self.training_metadata(
            batch_size,
            sequence_length,
            hidden_states_BTC.device,
        )
        return super().forward(
            hidden_states_BTC,
            v_first_BTC,
            attention_shift_BC,
            wkv_state_BHKK,
            (
                sequence_chunk_offsets_P,
                chunk_token_starts_R,
                chunk_token_ends_R,
            ),
        )


class RWKVStateTuningFeedForward(RWKVFeedForward):
    """Feed-forward with a trainable lane-broadcast shift."""

    _provider_operators: ClassVar[tuple[str, ...]] = STATE_TUNING_FEED_FORWARD_OPERATORS
    _provider_mode: ClassVar[str] = "feed-forward state tuning"

    @dataclass(kw_only=True, slots=True)
    class Config(RWKVFeedForward.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.initial_feed_forward_shift = nn.Parameter(
            torch.empty(config.dim, dtype=torch.bfloat16)
        )

    def forward(
        self,
        hidden_states_BTC: torch.Tensor,
        feed_forward_shift_BC: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del feed_forward_shift_BC
        feed_forward_shift_BC = (
            self.initial_feed_forward_shift.unsqueeze(0)
            .expand(hidden_states_BTC.shape[0], -1)
            .contiguous()
        )
        return super().forward(hidden_states_BTC, feed_forward_shift_BC)


class RWKVStateTuningModel(RWKVModel):
    """RWKV-7 model with only recurrent initial states trainable."""

    @dataclass(kw_only=True, slots=True)
    class Config(RWKVModel.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        if any(isinstance(module, _get_lora_cls(Linear)) for module in self.modules()):
            raise ValueError("RWKV-7 State Tuning and LoRA cannot be combined.")
        self.requires_grad_(False)
        for module in self.modules():
            if isinstance(module, RWKVStateTuningAttention):
                module.initial_attention_shift.requires_grad_(True)
                module.initial_wkv_state.requires_grad_(True)
            elif isinstance(module, RWKVStateTuningFeedForward):
                module.initial_feed_forward_shift.requires_grad_(True)


__all__ = [
    "RWKVStateTuningAttention",
    "RWKVStateTuningFeedForward",
    "RWKVStateTuningModel",
]
