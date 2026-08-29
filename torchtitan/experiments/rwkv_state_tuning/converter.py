# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config converter that enables RWKV7 initial-state tuning."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import cast, ClassVar

from torch import nn

from torchtitan.components.lora import LoRAConverter
from torchtitan.models.rwkv7.model import RWKV7ChannelMix, RWKV7TimeMix
from torchtitan.protocols.model import make_frozen_config, ModelConfigConverter
from torchtitan.protocols.module import Module

from .model import RWKV7StateTuningChannelMix, RWKV7StateTuningTimeMix


def _state_config(config: Module.Config) -> Module.Config:
    values = {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.init
    }
    param_init = dict(values.get("param_init") or {})
    if isinstance(config, RWKV7TimeMix.Config):
        param_init["initial_shift"] = nn.init.zeros_
        values["param_init"] = param_init
        return RWKV7StateTuningTimeMix.Config(**values)
    if isinstance(config, RWKV7ChannelMix.Config):
        param_init["initial_shift"] = nn.init.zeros_
        values["param_init"] = param_init
        return RWKV7StateTuningChannelMix.Config(**values)
    raise TypeError(f"Unsupported RWKV7 state-tuning config {type(config).__name__}.")


class RWKV7StateTuningConverter(ModelConfigConverter):
    """Freeze the base model and replace only TimeMix/ChannelMix state owners."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        incompatible_converter_types: ClassVar[tuple[type, ...]] = (
            LoRAConverter.Config,
        )

    def __init__(self, config: Config, **kwargs):
        del config, kwargs

    def convert(self, model_config: Module.Config) -> Module.Config:
        configs = list(model_config.traverse(Module.Config, recurse=True))
        if any("LoRA" in type(config).__qualname__ for _, config, _, _ in configs):
            raise ValueError("RWKV7 State Tuning cannot be combined with LoRA.")

        converted_root = model_config
        for _, config, parent, attribute in reversed(configs):
            if isinstance(config, (RWKV7TimeMix.Config, RWKV7ChannelMix.Config)):
                replacement = _state_config(config)
            else:
                replacement = make_frozen_config(cast(Module.Config, config))

            if parent is None:
                converted_root = replacement
            elif isinstance(parent, list):
                assert isinstance(attribute, int)
                parent[attribute] = replacement
            else:
                assert isinstance(attribute, str)
                setattr(parent, attribute, replacement)
        return converted_root


__all__ = ["RWKV7StateTuningConverter"]
