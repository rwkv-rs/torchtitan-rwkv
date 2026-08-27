# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import _StridedShard, Placement

from torchtitan.components.quantization._fsdp_weight import build_compute_weight
from torchtitan.protocols.module import Module

_active_parametrization = True


class ComputeWeightParametrization(nn.Module):
    """Derive a compute weight from the unsharded parameter, if it has one.

    Composes around a data-parallel parametrization rather than chaining after
    it. The representation to build is determined by the tensor subclass on
    the *sharded* parameter, and ReplicateComputation returns a plain local
    tensor, so that subclass is unreachable from its output -- this wrapper
    still sees the original parameter and passes both along.

    Parameters with no compute representation pass through, so this can be
    applied unconditionally.

    Used by passing the class itself as ``data_parallel``'s
    ``parametrization_transform``, which wraps it around the data-parallel
    parametrization before registering. The alternative was to do this inline
    in ``ReplicateComputation.forward``, which is simpler but puts quantization
    knowledge in the data-parallel core; going through the transform keeps that
    file free of it.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return build_compute_weight(self.inner(x), x)


@contextmanager
def disable_active_parametrization() -> Generator[None, None, None]:
    global _active_parametrization
    try:
        _active_parametrization = False
        yield
    finally:
        _active_parametrization = True


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


def _distribute_dtensor(
    tensor: DTensor,
    device_mesh: DeviceMesh,
    dp_placements: Sequence[Placement],
) -> DTensor:
    """
    Below are experimental enhancements to distribute a DTensor.
    This helps enable Simple FSDP + TP/EP, in which
        inner spec/mesh is TP/EP spec/mesh
        outer spec/mesh is FSDP/DDP/HSDP spec/mesh
    """
    inner_spec = tensor._spec
    outer_mesh, inner_mesh = device_mesh, inner_spec.mesh
    spanned_mesh = DeviceMesh._concatenate([outer_mesh, inner_mesh])

    if len(dp_placements) == 1:
        assert dp_placements[0].is_replicate() or dp_placements[0].is_shard()
        if dp_placements[0].is_shard():
            # For FSDP + EP/TP/EP+TP
            assert len(inner_spec.placements) == 2 or len(inner_spec.placements) == 1
            shard_dim = dp_placements[0].dim
            split_factor = inner_spec.num_shards_map[shard_dim]
            tensor_placement = (
                (
                    _StridedShard(shard_dim, split_factor=split_factor)
                    if split_factor > 1
                    else dp_placements[0]
                ),
            ) + inner_spec.placements
        else:
            # For DDP + TP/EP
            assert len(inner_spec.placements) == 1
            tensor_placement = (dp_placements[0], inner_spec.placements[0])
    elif len(dp_placements) == 2:
        assert dp_placements[0].is_replicate() and dp_placements[1].is_shard()
        # For HSDP + EP/TP/EP+TP
        assert len(inner_spec.placements) == 2 or len(inner_spec.placements) == 1
        shard_dim = dp_placements[1].dim
        split_factor = inner_spec.num_shards_map[shard_dim]
        tensor_placement = (
            dp_placements[0],
            (
                _StridedShard(shard_dim, split_factor=split_factor)
                if split_factor > 1
                else dp_placements[1]
            ),
        ) + inner_spec.placements
    else:
        raise ValueError(
            f"Unsupported placement {dp_placements} for distributing DTensor {tensor}"
        )

    # HSDP case needs 2 placements for 2D outer_mesh
    current_placements = (Replicate(),) * len(dp_placements)
    target_placements = tuple(dp_placements)

    current_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=current_placements,
        tensor_meta=inner_spec.tensor_meta,
    )
    target_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=target_placements,
        tensor_meta=inner_spec.tensor_meta,
    )
    result_tensor = redistribute_local_tensor(
        tensor._local_tensor,
        current_spec=current_spec,
        target_spec=target_spec,
    )
    return DTensor(
        result_tensor.requires_grad_(tensor.requires_grad),
        DTensorSpec(
            mesh=spanned_mesh,
            placements=tensor_placement,
            tensor_meta=inner_spec.tensor_meta,
        ),
        requires_grad=tensor.requires_grad,
    )


# Cache of (original_class, param_names) -> wrapper class, so all instances
# of the same module type share one SimpleFSDP class for torch.compile reuse.
_wrap_class_cache: dict[tuple[type, frozenset[str]], type] = {}


def _register_parametrization(
    module: nn.Module, param_names: list[str], parametrization: nn.Module
) -> None:
    """
    It works with state_dict without incurring parametrization calls because
    state_dict accesses parameters directly from self._parameters, not from getters
    https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L2141
    TODO: In checkpoint saving/loading, avoid parametrization calls when calling
    get_model_state_dict func in torchtitan/components/checkpointer/dcp.py.
    """
    param_name_to_property = {
        param_name: property(
            lambda self, pn=param_name: parametrization(self._parameters[pn])
        )
        for param_name in param_names
    }
    cache_key = (module.__class__, frozenset(param_names))
    if cache_key in _wrap_class_cache:
        module_cls = _wrap_class_cache[cache_key]
    else:
        module_cls = type(
            f"SimpleFSDP{module.__class__.__name__}",
            (module.__class__,),
            param_name_to_property,
        )
        # Expose the dynamically created class as a real, importable symbol
        # so that pickle/GraphPickler can resolve it during serialization.
        sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls
        _wrap_class_cache[cache_key] = module_cls
    module.__class__ = module_cls


class ReplicateComputation(Module):
    def __init__(
        self,
        device_mesh: DeviceMesh,
        param_sharding: tuple[Placement, ...],
        mode: str,
        mp_policy: MixedPrecisionPolicy | None,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.param_sharding = param_sharding
        self.mode = mode
        self.compute_placements: list[Placement] = [Replicate()] * self.device_mesh.ndim
        self.grad_placements: list[Placement] = [
            Partial(reduce_op="sum")
        ] * self.device_mesh.ndim
        mp_policy = mp_policy or MixedPrecisionPolicy()
        self.param_dtype: torch.dtype | None = mp_policy.param_dtype
        self.reduce_dtype: torch.dtype | None = mp_policy.reduce_dtype

    def replicate_compute(self, x: DTensor) -> torch.Tensor:
        # data parallel runtime replicate parameters and do local compute
        # the gradients are partial tensors that needs to perform reduction
        # (i.e. DDP: allreduce, FSDP: reduce_scatter, HSDP: mix of both)
        # support FSDP/DDP/HSDP + EP + TP (assuming TP shards the inner-most dim)
        non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
        assert non_dp_mesh_dims <= 2, "Only DP + EP/TP/EP+TP is supported"
        if non_dp_mesh_dims > 0:
            dp_mesh = self.device_mesh
            # re-wrap 2D DTensor to 1D DTensor on dp_mesh for efficient FSDP all-gather
            sharded_local_tensor = x.to_local()
            sharded_dtensor = DTensor.from_local(
                sharded_local_tensor, dp_mesh, self.param_sharding
            )

            # the actual FSDP's fwd all-gather & bwd reduce-scatter
            # DDP's bwd all-reduce on dp_mesh
            replicated_dtensor = sharded_dtensor.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.param_dtype,
                backward_dtype=self.reduce_dtype,
            )

            # re-wrap all-gathered DTensor on dp_mesh to be on non_dp_mesh
            # TODO: DTensor should support this mesh collapsing operation
            replicated_local_tensor = replicated_dtensor.to_local(
                grad_placements=self.grad_placements
            )

            non_dp_placements = tuple(x._spec.placements[-non_dp_mesh_dims:])
            non_dp_mesh_dim_names = tuple(
                x._spec.mesh.mesh_dim_names[-non_dp_mesh_dims:]
            )
            non_dp_mesh = x._spec.mesh[non_dp_mesh_dim_names]

            output = DTensor.from_local(
                replicated_local_tensor, non_dp_mesh, non_dp_placements
            )
        elif non_dp_mesh_dims == 0:
            output = x.redistribute(
                placements=self.compute_placements,
                forward_dtype=self.param_dtype,
                backward_dtype=self.reduce_dtype,
            )
            output = output.to_local(grad_placements=self.grad_placements)
        else:
            raise AssertionError(
                f"Unsupported replicate compute on placement {x._spec.placements} for DTensor {x}"
            )

        return output

    def forward(self, x: DTensor) -> torch.Tensor:
        global _active_parametrization
        # This should never be set to true during forward, only outside for model
        # inspection / debugging / initialization
        # model initialization can be done now through
        # with disable_active_parametrization():
        #     model.init_states()
        if not _active_parametrization:
            return x

        return self.replicate_compute(x)


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "replicate",
    mp_policy: MixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    parametrization_transform: Callable[[nn.Module], nn.Module] | None = None,
) -> nn.Module:
    """Shard ``model`` and install the data-parallel parametrization.

    ``parametrization_transform`` composes an outer parametrization around the
    data-parallel one, letting a caller derive something from the unsharded
    weight without this module knowing what. It receives the original
    parameter, which the data-parallel stage does not preserve: the stage
    returns a plain local tensor, so any tensor subclass on the sharded
    parameter is no longer reachable from its output.
    """
    param_sharding: tuple[Placement, ...]
    if mode == "replicate":
        param_sharding = (Replicate(),)
    elif mode == "fully_shard":
        param_sharding = (Shard(shard_dim),)
    elif mode == "hybrid_shard":
        # replicate inter-host, fully shard intra-host
        param_sharding = (Replicate(), Shard(shard_dim))
        assert (
            device_mesh.ndim == 2
        ), "hybrid sharded data parallel requires 2D DeviceMesh"
    else:
        raise ValueError(f"Unsupported mode {mode}")

    modules = list(model.modules())

    for mod in modules:
        params_dict = dict(mod.named_parameters(recurse=False))
        # we shouldn't apply data parallel to the modules that are already
        # sharded by data parallel
        if "SimpleFSDP" in mod.__class__.__name__:
            continue

        for p_name, p in params_dict.items():
            if p is not None and p.numel() > 0:
                distribute_tensor_func = (
                    _distribute_dtensor if isinstance(p, DTensor) else distribute_tensor
                )
                mod.register_parameter(
                    p_name,
                    nn.Parameter(
                        distribute_tensor_func(p, device_mesh, param_sharding)
                    ),
                )

                # to be compatible with DCP, we use a customized _register_parametrization
                # instead of nn.utils.parametrize.register_parametrization here
                # nn.utils.parametrize.register_parametrization(
                #     mod,
                #     p_name,
                #     ReplicateComputation(
                #         device_mesh,
                #         param_sharding,
                #         mode,
                #         mp_policy=mp_policy,
                #     ),
                #     unsafe=True,
                # )

        parametrization: nn.Module = ReplicateComputation(
            device_mesh,
            param_sharding,
            mode,
            mp_policy=mp_policy,
        )
        if parametrization_transform is not None:
            parametrization = parametrization_transform(parametrization)

        _register_parametrization(
            mod,
            list(params_dict.keys()),
            parametrization,
        )
    return model
