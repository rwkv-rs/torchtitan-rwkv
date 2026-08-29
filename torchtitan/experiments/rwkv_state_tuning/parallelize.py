# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP policy that preserves RWKV7 State Tuning WKV states in FP32."""

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.models.rwkv7.parallelize import apply_rwkv7_fsdp, prepare_rwkv7_for_fsdp

from .model import RWKV7StateTuningTimeMix


def parallelize_rwkv7_state_tuning(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
    """Shard FP32 WKV state owners before applying the shared BF16 policy."""
    dp_mesh, dp_mesh_axes = prepare_rwkv7_for_fsdp(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
    fp32_policy = MixedPrecisionPolicy(
        param_dtype=None,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    fsdp_options: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": fp32_policy,
        "dp_mesh_dims": dp_mesh_axes,
    }
    if training.enable_cpu_offload:
        fsdp_options["offload_policy"] = CPUOffloadPolicy()
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        pp_enabled=False,
    )

    for module in model.modules():
        if not isinstance(module, RWKV7StateTuningTimeMix):
            continue
        fully_shard(
            module._wkv_state,
            reshard_after_forward=reshard_after_forward,
            **fsdp_options,
        )

    return apply_rwkv7_fsdp(
        model,
        dp_mesh,
        dp_mesh_axes,
        training=training,
        parallelism=parallelism,
    )


__all__ = ["parallelize_rwkv7_state_tuning"]
