# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config converter that enables RWKV-7 initial-state tuning."""

from __future__ import annotations

from dataclasses import dataclass, fields

from torch import nn

from torchtitan.models.rwkv7.model import RWKVAttention, RWKVFeedForward, RWKVModel
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.module import Module

from .model import (
    RWKVStateTuningAttention,
    RWKVStateTuningFeedForward,
    RWKVStateTuningModel,
)


def _make_state_tuning_config(config: Module.Config) -> Module.Config:
    values = {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.init
    }
    param_init = dict(values.get("param_init") or {})
    if isinstance(config, RWKVAttention.Config):
        param_init["initial_attention_shift"] = nn.init.zeros_
        values["param_init"] = param_init
        return RWKVStateTuningAttention.Config(**values)
    if isinstance(config, RWKVFeedForward.Config):
        param_init["initial_feed_forward_shift"] = nn.init.zeros_
        values["param_init"] = param_init
        return RWKVStateTuningFeedForward.Config(**values)
    if isinstance(config, RWKVModel.Config):
        return RWKVStateTuningModel.Config(**values)
    raise TypeError(f"Unsupported RWKV-7 state-tuning config {type(config).__name__}.")


class RWKVStateTuningConverter(ModelConfigConverter):
    """Freeze the base model and replace only TimeMix/ChannelMix state owners."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config, **kwargs):
        del config, kwargs

    def convert(self, model_config: Module.Config) -> Module.Config:
        configs = list(model_config.traverse(Module.Config, recurse=True))
        converted_root = model_config
        for _, config, parent, attribute in reversed(configs):
            if not isinstance(
                config,
                (RWKVModel.Config, RWKVAttention.Config, RWKVFeedForward.Config),
            ):
                continue
            replacement = _make_state_tuning_config(config)

            if parent is None:
                converted_root = replacement
            elif isinstance(parent, list):
                assert isinstance(attribute, int)
                parent[attribute] = replacement
            else:
                assert isinstance(attribute, str)
                setattr(parent, attribute, replacement)
        return converted_root


__all__ = ["RWKVStateTuningConverter"]
