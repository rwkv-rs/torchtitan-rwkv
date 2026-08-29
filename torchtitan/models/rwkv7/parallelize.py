# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization utilities for RWKV-7."""

from typing import cast

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
from torchtitan.models.common.decoder import Decoder

from .model import RWKVModel


def parallelize_rwkv(
    model: RWKVModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> RWKVModel:
    """Apply activation checkpointing, compile, and DP/FSDP to RWKV-7."""
    unsupported = {
        "tp": parallel_dims.tp,
        "pp": parallel_dims.pp,
        "cp": parallel_dims.cp,
        "ep": parallel_dims.ep,
    }
    enabled = {name: degree for name, degree in unsupported.items() if degree != 1}
    if enabled:
        raise ValueError(
            "RWKV-7 currently supports only DP replicate and FSDP shard; "
            f"unsupported parallelism: {enabled}."
        )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if compile_config.enable and "model" in compile_config.components:
        model.preload_provider()
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


__all__ = ["parallelize_rwkv"]
