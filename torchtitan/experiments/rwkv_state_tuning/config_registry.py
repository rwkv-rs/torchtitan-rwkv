# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RWKV-7 State Tuning recipes."""

from dataclasses import replace

from torchtitan.components.optimizer import default_adamw
from torchtitan.models.rwkv7 import model_registry
from torchtitan.models.rwkv7.config_registry import sft_debugmodel, sft_rwkv_1_5b
from torchtitan.trainer import Trainer

from .checkpoint import RWKVStateTuningCheckpointManager
from .converter import RWKVStateTuningConverter
from .parallelize import parallelize_rwkv_state_tuning


def state_tuning_debugmodel() -> Trainer.Config:
    config = sft_debugmodel()
    model_spec = model_registry(
        "debugmodel",
        converters=[RWKVStateTuningConverter.Config()],
    )
    model_spec = replace(
        model_spec,
        parallelize_fn=parallelize_rwkv_state_tuning,
    )
    config.model_spec = model_spec
    config.optimizer = default_adamw(lr=8e-4)
    config.checkpoint = RWKVStateTuningCheckpointManager.Config(
        interval=5,
        last_save_model_only=False,
    )
    return config


def state_tuning_rwkv_1_5b() -> Trainer.Config:
    config = sft_rwkv_1_5b()
    model_spec = model_registry(
        "1.5b",
        converters=[RWKVStateTuningConverter.Config()],
    )
    model_spec = replace(
        model_spec,
        parallelize_fn=parallelize_rwkv_state_tuning,
    )
    config.model_spec = model_spec
    config.optimizer = default_adamw(lr=8e-4)
    config.checkpoint = RWKVStateTuningCheckpointManager.Config(
        interval=5,
        last_save_model_only=False,
        initial_load_model_only=True,
        initial_load_in_hf=True,
    )
    return config
