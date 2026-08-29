# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RWKV-7 model registration and canonical architecture configurations."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import Replicate

from torchtitan.models.common import Embedding, Linear
from torchtitan.models.common.nn_modules import GroupNorm, LayerNorm
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import RWKVAttention, RWKVDecoderLayer, RWKVFeedForward, RWKVModel
from .parallelize import parallelize_rwkv
from .state_dict_adapter import RWKVStateDictAdapter


__all__ = [
    "model_registry",
    "parallelize_rwkv",
    "RWKVAttention",
    "RWKVDecoderLayer",
    "RWKVFeedForward",
    "RWKVModel",
    "rwkv_configs",
]


_VOCAB_SIZE = 65536
_HEAD_SIZE = 64
_LAYER_NORM_EPSILON = 1e-5
_GROUP_NORM_EPSILON = _HEAD_SIZE * 1e-5


def _replicate_for_parameter(
    tensor: torch.Tensor,
    parameter: nn.Parameter,
) -> torch.Tensor:
    if isinstance(parameter, DTensor):
        return distribute_tensor(
            tensor,
            parameter.device_mesh,
            [Replicate()] * parameter.device_mesh.ndim,
        )
    return tensor


def _copy_param_init(formula: Callable[[torch.device], torch.Tensor]) -> Callable:
    def init(parameter: nn.Parameter) -> None:
        initialized = formula(parameter.device).to(dtype=parameter.dtype)
        with torch.no_grad():
            parameter.copy_(_replicate_for_parameter(initialized, parameter))

    return init


def _orthogonal_param_init(gain: float = 1.0) -> Callable:
    def init(parameter: nn.Parameter) -> None:
        initialized = torch.empty(
            parameter.shape,
            device=parameter.device,
            dtype=torch.float32,
        )
        nn.init.orthogonal_(initialized, gain=gain)
        initialized = initialized.to(dtype=parameter.dtype)
        with torch.no_grad():
            parameter.copy_(_replicate_for_parameter(initialized, parameter))

    return init


def _attention_param_init(
    *,
    layer_idx: int,
    num_layers: int,
    dim: int,
    head_size: int,
) -> dict[str, Callable]:
    def position(device: torch.device) -> torch.Tensor:
        return torch.arange(dim, device=device, dtype=torch.float32)

    def time_mix(exponent: float) -> Callable:
        def formula(device: torch.device) -> torch.Tensor:
            position_C = position(device)
            ddd_C = position_C / dim
            ratio_1_to_almost0 = 1.0 - layer_idx / num_layers
            return 1.0 - ddd_C.pow(exponent * ratio_1_to_almost0)

        return _copy_param_init(formula)

    def linear(device: torch.device) -> torch.Tensor:
        position_C = position(device)
        return position_C / max(dim - 1, 1) - 0.5

    def zigzag(device: torch.device) -> torch.Tensor:
        position_C = position(device)
        value_C = (position_C.remainder(head_size) - (head_size - 1) / 2) / (
            (head_size - 1) / 2
        )
        return value_C * value_C.abs()

    def decay(device: torch.device) -> torch.Tensor:
        position_C = position(device)
        ratio_0_to_1 = layer_idx / max(num_layers - 1, 1)
        return -6.0 + 6.0 * (position_C / max(dim - 1, 1)).pow(
            1.0 + ratio_0_to_1**0.3
        )

    param_init: dict[str, Callable] = {
        "x_r": time_mix(0.2),
        "x_w": time_mix(0.9),
        "x_k": time_mix(0.7),
        "x_v": time_mix(0.7),
        "x_a": time_mix(0.9),
        "x_g": time_mix(0.2),
        "w0": _copy_param_init(
            lambda device: decay(device) + 0.5 + zigzag(device) * 2.5
        ),
        "a0": _copy_param_init(
            lambda device: -0.19 + zigzag(device) * 0.3 + linear(device) * 0.4
        ),
        "w1": nn.init.zeros_,
        "w2": _orthogonal_param_init(0.1),
        "a1": nn.init.zeros_,
        "a2": _orthogonal_param_init(0.1),
        "g1": nn.init.zeros_,
        "g2": _orthogonal_param_init(0.1),
        "k_k": _copy_param_init(lambda device: 0.71 - linear(device) * 0.1),
        "k_a": partial(nn.init.constant_, val=1.02),
        "r_k": partial(nn.init.constant_, val=-0.04),
    }
    if layer_idx != 0:
        param_init.update(
            {
                "v0": _copy_param_init(lambda device: 0.73 - linear(device) * 0.4),
                "v1": nn.init.zeros_,
                "v2": _orthogonal_param_init(0.1),
            }
        )
    return param_init


def _feed_forward_param_init(
    *,
    layer_idx: int,
    num_layers: int,
    dim: int,
) -> dict[str, Callable]:
    def x_k(device: torch.device) -> torch.Tensor:
        ddd_C = torch.arange(dim, device=device, dtype=torch.float32) / dim
        ratio = 1.0 - layer_idx / num_layers
        return 1.0 - ddd_C.pow(ratio**4)

    return {"x_k": _copy_param_init(x_k)}


def _linear(
    in_features: int,
    out_features: int,
    initializer: Callable,
) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        param_init={"weight": initializer},
    )


def _norm(dim: int) -> LayerNorm.Config:
    return LayerNorm.Config(
        normalized_shape=dim,
        eps=_LAYER_NORM_EPSILON,
        param_init={"weight": nn.init.ones_, "bias": nn.init.zeros_},
    )


def _rwkv_config(
    *,
    num_layers: int,
    dim: int,
    hidden_dim: int,
    decay_low_rank_dim: int,
    v_low_rank_dim: int,
    gate_low_rank_dim: int,
) -> RWKVModel.Config:
    layers = []
    for layer_idx in range(num_layers):
        layers.append(
            RWKVDecoderLayer.Config(
                linear_attn=RWKVAttention.Config(
                    layer_idx=layer_idx,
                    dim=dim,
                    head_size=_HEAD_SIZE,
                    decay_low_rank_dim=decay_low_rank_dim,
                    a_low_rank_dim=decay_low_rank_dim,
                    v_low_rank_dim=v_low_rank_dim,
                    gate_low_rank_dim=gate_low_rank_dim,
                    r_proj=_linear(dim, dim, _orthogonal_param_init()),
                    k_proj=_linear(dim, dim, _orthogonal_param_init(0.1)),
                    v_proj=_linear(dim, dim, _orthogonal_param_init()),
                    o_proj=_linear(dim, dim, nn.init.zeros_),
                    g_norm=GroupNorm.Config(
                        num_groups=dim // _HEAD_SIZE,
                        num_channels=dim,
                        eps=_GROUP_NORM_EPSILON,
                        param_init={
                            "weight": partial(
                                nn.init.constant_,
                                val=((layer_idx + 1) / num_layers) ** 0.7,
                            ),
                            "bias": nn.init.zeros_,
                        },
                    ),
                    param_init=_attention_param_init(
                        layer_idx=layer_idx,
                        num_layers=num_layers,
                        dim=dim,
                        head_size=_HEAD_SIZE,
                    ),
                ),
                mlp=RWKVFeedForward.Config(
                    dim=dim,
                    key=_linear(dim, hidden_dim, _orthogonal_param_init()),
                    value=_linear(hidden_dim, dim, nn.init.zeros_),
                    param_init=_feed_forward_param_init(
                        layer_idx=layer_idx,
                        num_layers=num_layers,
                        dim=dim,
                    ),
                ),
                input_layernorm=_norm(dim),
                post_attention_layernorm=_norm(dim),
            )
        )

    lm_head_gain = 0.5 * math.sqrt(_VOCAB_SIZE / dim) if _VOCAB_SIZE > dim else 0.5
    return RWKVModel.Config(
        architecture_version="rwkv7",
        vocab_size=_VOCAB_SIZE,
        dim=dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        head_size=_HEAD_SIZE,
        decay_low_rank_dim=decay_low_rank_dim,
        a_low_rank_dim=decay_low_rank_dim,
        v_low_rank_dim=v_low_rank_dim,
        gate_low_rank_dim=gate_low_rank_dim,
        layer_norm_epsilon=_LAYER_NORM_EPSILON,
        group_norm_epsilon=_GROUP_NORM_EPSILON,
        tok_embeddings=Embedding.Config(
            num_embeddings=_VOCAB_SIZE,
            embedding_dim=dim,
            param_init={
                "weight": partial(nn.init.uniform_, a=-1e-4, b=1e-4),
            },
        ),
        embedding_norm=_norm(dim),
        layers=layers,
        norm=_norm(dim),
        lm_head=_linear(
            dim,
            _VOCAB_SIZE,
            _orthogonal_param_init(lm_head_gain),
        ),
    )


def _debugmodel() -> RWKVModel.Config:
    return _rwkv_config(
        num_layers=2,
        dim=128,
        hidden_dim=512,
        decay_low_rank_dim=32,
        v_low_rank_dim=32,
        gate_low_rank_dim=32,
    )


def _0_1b() -> RWKVModel.Config:
    return _rwkv_config(
        num_layers=12,
        dim=768,
        hidden_dim=3072,
        decay_low_rank_dim=64,
        v_low_rank_dim=32,
        gate_low_rank_dim=128,
    )


def _0_4b() -> RWKVModel.Config:
    return _rwkv_config(
        num_layers=24,
        dim=1024,
        hidden_dim=4096,
        decay_low_rank_dim=64,
        v_low_rank_dim=32,
        gate_low_rank_dim=128,
    )


def _1_5b() -> RWKVModel.Config:
    return _rwkv_config(
        num_layers=24,
        dim=2048,
        hidden_dim=8192,
        decay_low_rank_dim=96,
        v_low_rank_dim=64,
        gate_low_rank_dim=256,
    )


def _2_9b() -> RWKVModel.Config:
    return _rwkv_config(
        num_layers=32,
        dim=2560,
        hidden_dim=10240,
        decay_low_rank_dim=96,
        v_low_rank_dim=64,
        gate_low_rank_dim=320,
    )


def _7_2b() -> RWKVModel.Config:
    return _rwkv_config(
        num_layers=32,
        dim=4096,
        hidden_dim=16384,
        decay_low_rank_dim=128,
        v_low_rank_dim=96,
        gate_low_rank_dim=480,
    )


def _13_3b() -> RWKVModel.Config:
    return _rwkv_config(
        num_layers=61,
        dim=4096,
        hidden_dim=16384,
        decay_low_rank_dim=192,
        v_low_rank_dim=128,
        gate_low_rank_dim=384,
    )


rwkv_configs = {
    "debugmodel": _debugmodel,
    "0.1b": _0_1b,
    "0.4b": _0_4b,
    "1.5b": _1_5b,
    "2.9b": _2_9b,
    "7.2b": _7_2b,
    "13.3b": _13_3b,
}


def model_registry(
    flavor: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = rwkv_configs[flavor]()
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            config = converter.build().convert(config)

    return ModelSpec(
        name="rwkv7",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_rwkv,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=RWKVStateDictAdapter,
    )
