# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization utilities for RWKV7."""

from typing import cast, TYPE_CHECKING

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, resolve_fsdp_mesh

if TYPE_CHECKING:
    from torchtitan.models.common.decoder import Decoder
    from torchtitan.models.rwkv7.model import RWKV7Model


def prepare_rwkv7_for_fsdp(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> tuple[DeviceMesh, DataParallelMeshDims | None]:
    """Validate RWKV7 parallelism, apply techniques, and resolve its DP mesh."""
    unsupported = {
        "tp": parallel_dims.tp,
        "pp": parallel_dims.pp,
        "cp": parallel_dims.cp,
        "ep": parallel_dims.ep,
    }
    enabled = {name: degree for name, degree in unsupported.items() if degree != 1}
    if enabled:
        raise ValueError(
            "RWKV7 currently supports only DP replicate and FSDP shard; "
            f"unsupported parallelism: {enabled}."
        )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if compile_config.enable and "model" in compile_config.components:
        cast("RWKV7Model", model).preload_provider()
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    if parallelism.spmd_backend == "spmd_types":
        dp_mesh, dp_mesh_axes = resolve_fsdp_mesh(parallel_dims)
    else:
        mesh_axis_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(mesh_axis_names)
        dp_mesh_axes = None
    return dp_mesh, dp_mesh_axes


def apply_rwkv7_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    dp_mesh_axes: DataParallelMeshDims | None,
    *,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
) -> nn.Module:
    """Apply the shared RWKV7 FSDP policy after any specialized child groups."""
    apply_fsdp_to_decoder(
        cast("Decoder", model),
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        dp_mesh_dims=dp_mesh_axes,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    return model


def parallelize_rwkv7(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> nn.Module:
    """Apply activation checkpointing, compile, and DP/FSDP to RWKV7."""
    dp_mesh, dp_mesh_axes = prepare_rwkv7_for_fsdp(
        model,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
    return apply_rwkv7_fsdp(
        model,
        dp_mesh,
        dp_mesh_axes,
        training=training,
        parallelism=parallelism,
    )


__all__ = ["apply_rwkv7_fsdp", "parallelize_rwkv7", "prepare_rwkv7_for_fsdp"]
