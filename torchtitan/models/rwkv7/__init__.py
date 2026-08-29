# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RWKV7 model registration and canonical architecture configurations."""

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

from .model import RWKV7Block, RWKV7ChannelMix, RWKV7Model, RWKV7TimeMix
from .parallelize import parallelize_rwkv7
from .state_dict_adapter import RWKV7StateDictAdapter


__all__ = [
    "model_registry",
    "parallelize_rwkv7",
    "RWKV7Block",
    "RWKV7ChannelMix",
    "RWKV7Model",
    "RWKV7TimeMix",
    "rwkv7_configs",
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


def _copy_formula(formula: Callable[[torch.device], torch.Tensor]) -> Callable:
    def init(parameter: nn.Parameter) -> None:
        initialized = formula(parameter.device).to(dtype=parameter.dtype)
        with torch.no_grad():
            parameter.copy_(_replicate_for_parameter(initialized, parameter))

    return init


def _orthogonal(gain: float = 1.0) -> Callable:
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


def _time_mix_param_init(
    *,
    layer_id: int,
    num_layers: int,
    dim: int,
    head_size: int,
) -> dict[str, Callable]:
    def channel_position(device: torch.device) -> torch.Tensor:
        return torch.arange(dim, device=device, dtype=torch.float32)

    def time_mix(exponent: float) -> Callable:
        def formula(device: torch.device) -> torch.Tensor:
            position_C = channel_position(device)
            ddd_C = position_C / dim
            ratio = 1.0 - layer_id / num_layers
            return 1.0 - ddd_C.pow(exponent * ratio)

        return _copy_formula(formula)

    def linear(device: torch.device) -> torch.Tensor:
        position_C = channel_position(device)
        return position_C / max(dim - 1, 1) - 0.5

    def zigzag(device: torch.device) -> torch.Tensor:
        position_C = channel_position(device)
        value_C = (position_C.remainder(head_size) - (head_size - 1) / 2) / (
            (head_size - 1) / 2
        )
        return value_C * value_C.abs()

    def decay(device: torch.device) -> torch.Tensor:
        position_C = channel_position(device)
        ratio = layer_id / max(num_layers - 1, 1)
        return -6.0 + 6.0 * (position_C / max(dim - 1, 1)).pow(1.0 + ratio**0.3)

    param_init: dict[str, Callable] = {
        "x_r": time_mix(0.2),
        "x_w": time_mix(0.9),
        "x_k": time_mix(0.7),
        "x_v": time_mix(0.7),
        "x_a": time_mix(0.9),
        "x_g": time_mix(0.2),
        "w0": _copy_formula(lambda device: decay(device) + 0.5 + zigzag(device) * 2.5),
        "a0": _copy_formula(
            lambda device: -0.19 + zigzag(device) * 0.3 + linear(device) * 0.4
        ),
        "w1": nn.init.zeros_,
        "w2": _orthogonal(0.1),
        "a1": nn.init.zeros_,
        "a2": _orthogonal(0.1),
        "g1": nn.init.zeros_,
        "g2": _orthogonal(0.1),
        "k_k": _copy_formula(lambda device: 0.71 - linear(device) * 0.1),
        "k_a": partial(nn.init.constant_, val=1.02),
        "r_k": partial(nn.init.constant_, val=-0.04),
    }
    if layer_id != 0:
        param_init.update(
            {
                "v0": _copy_formula(lambda device: 0.73 - linear(device) * 0.4),
                "v1": nn.init.zeros_,
                "v2": _orthogonal(0.1),
            }
        )
    return param_init


def _channel_mix_param_init(
    *,
    layer_id: int,
    num_layers: int,
    dim: int,
) -> dict[str, Callable]:
    def x_k(device: torch.device) -> torch.Tensor:
        ddd_C = torch.arange(dim, device=device, dtype=torch.float32) / dim
        ratio = 1.0 - layer_id / num_layers
        return 1.0 - ddd_C.pow(ratio**4)

    return {"x_k": _copy_formula(x_k)}


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


def _layer_norm(dim: int) -> LayerNorm.Config:
    return LayerNorm.Config(
        normalized_shape=dim,
        eps=_LAYER_NORM_EPSILON,
        param_init={"weight": nn.init.ones_, "bias": nn.init.zeros_},
    )


def _build_config(
    *,
    num_layers: int,
    dim: int,
    hidden_dim: int,
    decay_low_rank_dim: int,
    v_low_rank_dim: int,
    gate_low_rank_dim: int,
) -> RWKV7Model.Config:
    layers = []
    for layer_id in range(num_layers):
        layers.append(
            RWKV7Block.Config(
                linear_attn=RWKV7TimeMix.Config(
                    layer_id=layer_id,
                    dim=dim,
                    head_size=_HEAD_SIZE,
                    decay_low_rank_dim=decay_low_rank_dim,
                    a_low_rank_dim=decay_low_rank_dim,
                    v_low_rank_dim=v_low_rank_dim,
                    gate_low_rank_dim=gate_low_rank_dim,
                    r_proj=_linear(dim, dim, _orthogonal()),
                    k_proj=_linear(dim, dim, _orthogonal(0.1)),
                    v_proj=_linear(dim, dim, _orthogonal()),
                    o_proj=_linear(dim, dim, nn.init.zeros_),
                    g_norm=GroupNorm.Config(
                        num_groups=dim // _HEAD_SIZE,
                        num_channels=dim,
                        eps=_GROUP_NORM_EPSILON,
                        param_init={
                            "weight": partial(
                                nn.init.constant_,
                                val=((layer_id + 1) / num_layers) ** 0.7,
                            ),
                            "bias": nn.init.zeros_,
                        },
                    ),
                    param_init=_time_mix_param_init(
                        layer_id=layer_id,
                        num_layers=num_layers,
                        dim=dim,
                        head_size=_HEAD_SIZE,
                    ),
                ),
                mlp=RWKV7ChannelMix.Config(
                    dim=dim,
                    key=_linear(dim, hidden_dim, _orthogonal()),
                    value=_linear(hidden_dim, dim, nn.init.zeros_),
                    param_init=_channel_mix_param_init(
                        layer_id=layer_id,
                        num_layers=num_layers,
                        dim=dim,
                    ),
                ),
                input_layernorm=_layer_norm(dim),
                post_attention_layernorm=_layer_norm(dim),
            )
        )

    lm_head_gain = 0.5 * math.sqrt(_VOCAB_SIZE / dim) if _VOCAB_SIZE > dim else 0.5
    return RWKV7Model.Config(
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
        embedding_norm=_layer_norm(dim),
        layers=layers,
        norm=_layer_norm(dim),
        lm_head=_linear(
            dim,
            _VOCAB_SIZE,
            _orthogonal(lm_head_gain),
        ),
    )


def _debugmodel() -> RWKV7Model.Config:
    return _build_config(
        num_layers=2,
        dim=128,
        hidden_dim=512,
        decay_low_rank_dim=32,
        v_low_rank_dim=32,
        gate_low_rank_dim=32,
    )


def _flavor(
    num_layers: int,
    dim: int,
    hidden_dim: int,
    decay_low_rank_dim: int,
    v_low_rank_dim: int,
    gate_low_rank_dim: int,
) -> Callable[[], RWKV7Model.Config]:
    return partial(
        _build_config,
        num_layers=num_layers,
        dim=dim,
        hidden_dim=hidden_dim,
        decay_low_rank_dim=decay_low_rank_dim,
        v_low_rank_dim=v_low_rank_dim,
        gate_low_rank_dim=gate_low_rank_dim,
    )


rwkv7_configs = {
    "debugmodel": _debugmodel,
    "0.1b": _flavor(12, 768, 3072, 64, 32, 128),
    "0.4b": _flavor(24, 1024, 4096, 64, 32, 128),
    "1.5b": _flavor(24, 2048, 8192, 96, 64, 256),
    "2.9b": _flavor(32, 2560, 10240, 96, 64, 320),
    "7.2b": _flavor(32, 4096, 16384, 128, 96, 480),
    "13.3b": _flavor(61, 4096, 16384, 192, 128, 384),
}


def model_registry(
    flavor: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = rwkv7_configs[flavor]()
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            config = converter.build().convert(config)

    return ModelSpec(
        name="rwkv7",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_rwkv7,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=RWKV7StateDictAdapter,
    )
