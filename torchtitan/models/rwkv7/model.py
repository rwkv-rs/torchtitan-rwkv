# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RWKV7 training model backed exclusively by FlashRWKV2."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast, ClassVar

import torch
from torch import nn

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.common.nn_modules import GroupNorm, LayerNorm
from torchtitan.models.utils import get_nparams_and_active_nparams
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict

from .provider import (
    load_flash_rwkv2,
    preload_flash_rwkv2,
    PRETRAIN_CHANNELMIX_OPERATORS,
    PRETRAIN_TIMEMIX_OPERATORS,
)


# Shape suffixes in this file:
# B: batch lanes, T: tokens per lane, C: model channels, H: recurrent heads,
# K: recurrent head width, F: ChannelMix hidden width.


class RWKV7TimeMix(Module):
    """RWKV7 TimeMix using the fused FlashRWKV2 pretraining operators."""

    _provider_operators: ClassVar[tuple[str, ...]] = PRETRAIN_TIMEMIX_OPERATORS
    _provider_mode: ClassVar[str] = "TimeMix training"

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        dim: int
        head_size: int
        decay_low_rank_dim: int
        a_low_rank_dim: int
        v_low_rank_dim: int
        gate_low_rank_dim: int
        r_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        o_proj: Linear.Config
        g_norm: GroupNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.layer_id = config.layer_id
        self.dim = config.dim
        self.head_size = config.head_size
        self.num_heads = config.dim // config.head_size

        self.x_r = nn.Parameter(torch.empty(config.dim))
        self.x_w = nn.Parameter(torch.empty(config.dim))
        self.x_k = nn.Parameter(torch.empty(config.dim))
        self.x_v = nn.Parameter(torch.empty(config.dim))
        self.x_a = nn.Parameter(torch.empty(config.dim))
        self.x_g = nn.Parameter(torch.empty(config.dim))

        self.w0 = nn.Parameter(torch.empty(config.dim))
        self.w1 = nn.Parameter(torch.empty(config.dim, config.decay_low_rank_dim))
        self.w2 = nn.Parameter(torch.empty(config.decay_low_rank_dim, config.dim))
        self.a0 = nn.Parameter(torch.empty(config.dim))
        self.a1 = nn.Parameter(torch.empty(config.dim, config.a_low_rank_dim))
        self.a2 = nn.Parameter(torch.empty(config.a_low_rank_dim, config.dim))
        if config.layer_id != 0:
            self.v0 = nn.Parameter(torch.empty(config.dim))
            self.v1 = nn.Parameter(torch.empty(config.dim, config.v_low_rank_dim))
            self.v2 = nn.Parameter(torch.empty(config.v_low_rank_dim, config.dim))
        self.g1 = nn.Parameter(torch.empty(config.dim, config.gate_low_rank_dim))
        self.g2 = nn.Parameter(torch.empty(config.gate_low_rank_dim, config.dim))

        self.k_k = nn.Parameter(torch.empty(config.dim))
        self.k_a = nn.Parameter(torch.empty(config.dim))
        self.r_k = nn.Parameter(torch.empty(self.num_heads, config.head_size))

        self.r_proj = config.r_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.o_proj = config.o_proj.build()
        self.g_norm = config.g_norm.build()
        self._flash_rwkv2: ModuleType | None = None

    def preload_provider(self) -> None:
        self._flash_rwkv2 = preload_flash_rwkv2(
            self._provider_operators,
            self._provider_mode,
        )

    def _project_shifted(
        self,
        flash_rwkv2: ModuleType,
        xr_BTC: torch.Tensor,
        xw_BTC: torch.Tensor,
        xk_BTC: torch.Tensor,
        xv_BTC: torch.Tensor,
        xa_BTC: torch.Tensor,
        xg_BTC: torch.Tensor,
        v_first_BTC: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        r_BTC = self.r_proj(xr_BTC).contiguous()
        w_BTC = (self.w0 + torch.tanh(xw_BTC @ self.w1) @ self.w2).contiguous()
        k_BTC = self.k_proj(xk_BTC).contiguous()
        v_BTC = self.v_proj(xv_BTC).contiguous()
        if self.layer_id == 0:
            v_first_BTC = v_BTC
        else:
            assert v_first_BTC is not None
            v12_BTC = ((xv_BTC @ self.v1) @ self.v2).contiguous()
            v_BTC = flash_rwkv2.pretrain_tmix_vres_gate_bf16(
                v_BTC,
                v_first_BTC,
                self.v0,
                v12_BTC,
            )

        a_BTC = flash_rwkv2.pretrain_tmix_a_gate_bf16(
            self.a0,
            ((xa_BTC @ self.a1) @ self.a2).contiguous(),
        )
        g_BTC = (torch.sigmoid(xg_BTC @ self.g1) @ self.g2).contiguous()
        k_BTC, kk_BTC, ka_BTC = flash_rwkv2.pretrain_tmix_kk_pre_bf16(
            k_BTC,
            self.k_k,
            a_BTC,
            self.k_a,
            head_size=self.head_size,
        )
        return r_BTC, w_BTC, k_BTC, v_BTC, kk_BTC, ka_BTC, g_BTC

    def _readout(
        self,
        flash_rwkv2: ModuleType,
        recurrent_BTC: torch.Tensor,
        r_BTC: torch.Tensor,
        k_BTC: torch.Tensor,
        v_BTC: torch.Tensor,
        g_BTC: torch.Tensor,
    ) -> torch.Tensor:
        mixed_BTC = flash_rwkv2.pretrain_tmix_readout_bf16(
            recurrent_BTC.contiguous(),
            r_BTC,
            k_BTC,
            v_BTC,
            self.r_k,
            self.g_norm.weight,
            self.g_norm.bias,
            g_BTC,
            head_size=self.head_size,
        )
        return self.o_proj(mixed_BTC)

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
        x_BTC = x_BTC.contiguous()
        (
            xr_BTC,
            xw_BTC,
            xk_BTC,
            xv_BTC,
            xa_BTC,
            xg_BTC,
        ) = flash_rwkv2.pretrain_tmix_tokenshift_bf16(
            x_BTC,
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
        recurrent_BTC = flash_rwkv2.pretrain_tmix_wkv7_recurrent_bf16(
            r_BTC,
            w_BTC,
            k_BTC,
            v_BTC,
            kk_BTC,
            ka_BTC,
            head_size=self.head_size,
        )
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


class RWKV7ChannelMix(Module):
    """RWKV7 ChannelMix using the fused FlashRWKV2 pretraining operator."""

    _provider_operators: ClassVar[tuple[str, ...]] = PRETRAIN_CHANNELMIX_OPERATORS
    _provider_mode: ClassVar[str] = "ChannelMix training"

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        key: Linear.Config
        value: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.x_k = nn.Parameter(torch.empty(config.dim))
        self.key = config.key.build()
        self.value = config.value.build()
        self._flash_rwkv2: ModuleType | None = None

    def preload_provider(self) -> None:
        self._flash_rwkv2 = preload_flash_rwkv2(
            self._provider_operators,
            self._provider_mode,
        )

    def forward(self, x_BTC: torch.Tensor) -> torch.Tensor:
        flash_rwkv2 = load_flash_rwkv2(
            self._provider_operators,
            x_BTC,
            self._provider_mode,
            preloaded=self._flash_rwkv2,
        )
        return flash_rwkv2.pretrain_cmix_bf16(
            x_BTC.contiguous(),
            self.x_k,
            self.key.weight,
            self.value.weight,
        )


class RWKV7Block(Module):
    """One RWKV7 TimeMix and ChannelMix residual block."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        linear_attn: RWKV7TimeMix.Config
        mlp: RWKV7ChannelMix.Config
        input_layernorm: LayerNorm.Config
        post_attention_layernorm: LayerNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.linear_attn = config.linear_attn.build()
        self.mlp = config.mlp.build()
        self.input_layernorm = config.input_layernorm.build()
        self.post_attention_layernorm = config.post_attention_layernorm.build()

    def forward(
        self,
        x_BTC: torch.Tensor,
        v_first_BTC: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_BTC, v_first_BTC = self.linear_attn(
            self.input_layernorm(x_BTC).contiguous(),
            v_first_BTC,
        )
        x_BTC = x_BTC + attn_BTC
        x_BTC = x_BTC + self.mlp(self.post_attention_layernorm(x_BTC).contiguous())
        return x_BTC, v_first_BTC


class RWKV7Model(BaseModel):
    """RWKV7 causal language model for TorchTitan training."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        architecture_version: str
        vocab_size: int
        dim: int
        hidden_dim: int
        num_layers: int
        head_size: int
        decay_low_rank_dim: int
        a_low_rank_dim: int
        v_low_rank_dim: int
        gate_low_rank_dim: int
        layer_norm_epsilon: float
        group_norm_epsilon: float
        tok_embeddings: Embedding.Config
        embedding_norm: LayerNorm.Config
        layers: list[RWKV7Block.Config]
        norm: LayerNorm.Config
        lm_head: Linear.Config
        context_length: int = 4096

        def __post_init__(self) -> None:
            if self.architecture_version != "rwkv7":
                raise ValueError(
                    "RWKV7 architecture_version must be 'rwkv7', got "
                    f"{self.architecture_version!r}."
                )
            if self.head_size != 64:
                raise ValueError(f"RWKV7 requires head_size=64, got {self.head_size}.")
            if self.dim <= 0 or self.dim % self.head_size != 0:
                raise ValueError(
                    "RWKV7 dim must be positive and divisible by head_size, got "
                    f"dim={self.dim}, head_size={self.head_size}."
                )
            if self.hidden_dim != 4 * self.dim:
                raise ValueError(
                    f"RWKV7 hidden_dim must equal 4 * dim ({4 * self.dim}), "
                    f"got {self.hidden_dim}."
                )
            if self.num_layers <= 0 or len(self.layers) != self.num_layers:
                raise ValueError(
                    "RWKV7 layers must match num_layers, got "
                    f"num_layers={self.num_layers}, layers={len(self.layers)}."
                )
            ranks = {
                "decay_low_rank_dim": self.decay_low_rank_dim,
                "a_low_rank_dim": self.a_low_rank_dim,
                "v_low_rank_dim": self.v_low_rank_dim,
                "gate_low_rank_dim": self.gate_low_rank_dim,
            }
            invalid = {name: rank for name, rank in ranks.items() if rank <= 0}
            if invalid:
                raise ValueError(
                    f"RWKV7 low-rank dimensions must be positive: {invalid}."
                )
            if self.layer_norm_epsilon <= 0 or self.group_norm_epsilon <= 0:
                raise ValueError("RWKV7 normalization epsilons must be positive.")

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            training = config.training
            parallelism = config.parallelism
            if self.architecture_version != "rwkv7":
                raise ValueError(
                    "RWKV7Model requires architecture_version='rwkv7', got "
                    f"{self.architecture_version!r}."
                )
            if training.dtype != "bfloat16":
                raise ValueError(
                    "RWKV7 training requires training.dtype='bfloat16', got "
                    f"{training.dtype!r}."
                )
            if training.max_context_length % 16 != 0:
                raise ValueError(
                    "RWKV7 training.max_context_length must be a multiple of 16, "
                    f"got {training.max_context_length}."
                )
            if (
                training.num_tokens_per_microbatch_per_dp_rank
                % training.max_context_length
                != 0
            ):
                raise ValueError(
                    "RWKV7 num_tokens_per_microbatch_per_dp_rank must be divisible "
                    "by training.max_context_length."
                )
            unsupported = {
                "tensor_parallel_degree": parallelism.tensor_parallel_degree,
                "pipeline_parallel_degree": parallelism.pipeline_parallel_degree,
                "context_parallel_degree": parallelism.context_parallel_degree,
                "expert_parallel_degree": parallelism.expert_parallel_degree,
            }
            enabled = {
                name: degree for name, degree in unsupported.items() if degree != 1
            }
            if enabled:
                raise ValueError(
                    "RWKV7 currently supports only single-device, DP replicate, "
                    f"and FSDP shard; unsupported parallelism: {enabled}."
                )
            self.context_length = training.max_context_length

        def get_nparams_and_flops(
            self,
            model: nn.Module,
            seq_len: int,
        ) -> tuple[int, int]:
            del seq_len
            num_params, num_active_params = get_nparams_and_active_nparams(model)
            return num_params, 6 * num_active_params

    _skip_lm_head: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = config.tok_embeddings.build()
        self.embedding_norm = config.embedding_norm.build()
        self.layers = ModuleDict()
        for layer_id, layer_config in enumerate(config.layers):
            self.layers[str(layer_id)] = layer_config.build()
        self.norm = config.norm.build()
        self.lm_head = config.lm_head.build()
        self.enable_weight_tying = False

    def preload_provider(self) -> None:
        """Resolve FlashRWKV2 before block-level full-graph compilation."""
        for layer in self.layers.values():
            block = cast(RWKV7Block, layer)
            block.linear_attn.preload_provider()
            block.mlp.preload_provider()

    def forward(self, tokens_BT: torch.Tensor) -> torch.Tensor:
        x_BTC = self.tok_embeddings(tokens_BT)
        x_BTC = self.embedding_norm(x_BTC).contiguous()
        v_first_BTC = None
        for layer in self.layers.values():
            x_BTC, v_first_BTC = layer(x_BTC, v_first_BTC)
        x_BTC = self.norm(x_BTC)
        x_TC = x_BTC.reshape(-1, self.config.dim)
        if self._skip_lm_head:
            return x_TC
        return self.lm_head(x_TC)

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del parallel_dims, parallelism
        batch = dict(input_dict)
        tokens_T = batch.pop("input")
        labels_T = batch.pop("labels")
        batch.pop("positions", None)
        if batch:
            raise ValueError(
                f"RWKV7 received unsupported input fields: {sorted(batch)}."
            )
        tokens_BT = tokens_T.reshape(-1, self.config.context_length)
        return tokens_BT, labels_T.reshape(-1), {}


__all__ = [
    "RWKV7Block",
    "RWKV7ChannelMix",
    "RWKV7Model",
    "RWKV7TimeMix",
]
