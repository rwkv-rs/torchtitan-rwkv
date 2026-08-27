# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private FSDP lifecycle support for compute weight representations.

The lifecycle uses two tensor subclasses, one per state:

``_ShardedFSDPWeight``
    The persistent parameter. Holds the high-precision shard, owns the FSDP
    pre/post-all-gather hooks, and knows how to build a format's compute
    representation. A format subclasses this and implements one method.

``_ComputeFSDPWeight``
    The unsharded weight for one unshard lifetime. Holds only the format's
    operands and no high-precision storage, which is what lets FSDP release
    the all-gather output. Generic: it derives its FSDP-managed inner tensors
    from the representation's dataclass fields, so formats do not subclass it.

Both present the logical high-precision metadata of the model parameter, so
autograd returns a high-precision parameter gradient in either state.

Instance timeline, from module construction to reshard::

    module __init__      S = _ShardedFSDPWeight(bf16_shard)
                         |     one instance, lives for the whole run;
                         |     it is the nn.Parameter and the checkpoint
                         v
    ---- unshard ----------------------------------------------------------
    fsdp_pre_all_gather  S._tensor.to(param_dtype)  ->  BF16 comm tensor
                         |
                         v
    (all-gather)         replicated BF16 weight, temporary
                         |
    fsdp_post_all_gather |  out is None: first unshard
      S builds ------->  C = _ComputeFSDPWeight(representation)
                         |     new instance; holds only qdata/scales.
                         |     Returned to FSDP with the representation's
                         |     tensors so FSDP can manage their storage.
                         v
                         BF16 comm tensor released -- C never referenced it
                         |
    forward/backward     compute reads C.compute_representation
                         |
    ---- reshard ----------------------------------------------------------
                         FSDP frees the storage of C's inner tensors.
                         C itself stays alive: autograd and the module may
                         still hold it, and its addresses must not move.
                         |
    ---- unshard again --------------------------------------------------->
    fsdp_post_all_gather |  out is C: refill
      S refills ------>  C's existing tensors are written in place
                         |     no new instance; _validate_refilled_tensor_
                         |     identity() enforces that
                         v
                         (repeats until the final reshard)

So S is created once per parameter and C once per *distinct* unshard
lifetime -- not once per unshard. RAF=False keeps a single C alive across
forward, backward, recomputation, and pipeline microbatches; RAF=True
reuses that same C object, refilling its storage before backward.

GraphTrainer's SimpleFSDP reaches the same place by a different route: it
reconstructs the replicated BF16 weight itself and calls
:func:`build_compute_weight`, which constructs C directly.

Terminology
-----------

Several names here are a word apart, so:

compute representation
    The frozen dataclass of a format's operands -- qdata, scales, any
    workspace. Plain data; not a tensor subclass.

compute weight
    The ``_ComputeFSDPWeight`` tensor that *holds* a compute representation
    and presents the parameter's logical high-precision metadata. "C" below.

``_build_compute_representation(logical_weight, out=None)``
    The format's quantizer, and the only method a format implements. Returns
    a compute *representation*. With ``out`` set it refills that
    representation's existing tensors in place instead of allocating.

``_build_compute_weight(weight)``
    Both steps: quantize an unsharded weight, then wrap the result. Used by
    SimpleFSDP, which reconstructs the unsharded weight itself instead of
    going through the FSDP hooks.

``_BuildComputeWeightFunction``
    The ``autograd.Function`` wrapping ``_build_compute_weight``, so the
    compute weight's gradient reaches the parameter.

``build_compute_weight(unsharded_weight, sharded_parameter)``
    The entry point GraphTrainer's SimpleFSDP calls from inside the
    parametrization it composes -- its only caller, since FSDP2 goes through
    the hooks instead. The only public name here.

logical weight
    The unsharded high-precision weight handed to the quantizer. "Logical"
    because it excludes any padding the all-gather added.

managed tensors
    The representation's tensors, whose storage FSDP allocates, frees, and
    refills across the unshard lifecycle. One per dataclass field.

metadata source
    The tensor a compute weight copies its logical shape, dtype, device, and
    layout from -- normally the unsharded high-precision weight it was built
    from, since a compute weight has no storage of its own to describe.

Adding a format
---------------

Subclass ``_ShardedFSDPWeight``; do not subclass ``_ComputeFSDPWeight``.

A format differs from every other format in exactly one way: how it turns a
high-precision weight into its operands. That belongs on the sharded class,
because the sharded weight is the parameter FSDP calls hooks on, so it is
what *produces* the operands. ``_ComputeFSDPWeight`` only *holds* them, and
holding is format-independent -- it reads the operand tensors off the
representation's dataclass fields, which works for any format. Subclassing
it would add a type that overrides nothing.

So a format supplies a frozen dataclass of the tensors one unshard lifetime
owns, and one method, ``_build_compute_representation``. Everything else --
flattening, refill, reshard, the SimpleFSDP bridge -- comes from here.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


# The compute-weight machinery is internal; ``build_compute_weight`` is the
# one entry point a data-parallel implementation needs.
__all__ = ["build_compute_weight"]

# Ops FSDP performs on the sharded parameter's real storage. The wrapper must
# survive them so the parameter keeps its identity across FSDP bookkeeping.
_FSDP_SHARDED_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.c10d.scatter_.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.alias.default,
}

# The compute weight has no high-precision storage, so only metadata-level ops
# are answerable. Views re-wrap; factories allocate fresh plain tensors.
_FSDP_COMPUTE_VIEW_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
}

_FSDP_COMPUTE_FACTORY_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.zeros_like.default,
}


def _representation_field_names(representation_cls: type) -> tuple[str, ...]:
    """Return the field names naming a representation's owned allocations."""
    if not is_dataclass(representation_cls):
        raise TypeError(
            "A compute representation must be a dataclass of tensors; got "
            f"{representation_cls.__name__}."
        )
    return tuple(field.name for field in fields(representation_cls))


def _representation_tensors(representation: Any) -> tuple[torch.Tensor, ...]:
    """Return the independently allocated tensors a representation owns.

    Field order defines the FSDP-managed tensor order, so no format restates
    it. Every field must be a distinct allocation: FSDP takes ownership of
    each one's storage, so listing two views of the same storage would make it
    free the same memory twice and leave ``__tensor_unflatten__`` unable to
    tell which field was the derived one. Derived views belong in properties,
    which ``fields()`` skips.
    """
    tensors = tuple(
        getattr(representation, name)
        for name in _representation_field_names(type(representation))
    )
    storages = {tensor.untyped_storage()._cdata for tensor in tensors}
    if len(storages) != len(tensors):
        raise ValueError(
            f"{type(representation).__name__} fields must be distinct "
            "allocations; a field aliasing another field's storage should be "
            "a property instead."
        )
    return tensors


def _validate_refilled_tensor_identity(
    managed_tensors: tuple[torch.Tensor, ...],
    refilled_tensors: tuple[torch.Tensor, ...],
) -> None:
    """Require a refill to preserve every tensor object managed by FSDP."""
    if len(managed_tensors) != len(refilled_tensors) or any(
        previous is not current
        for previous, current in zip(managed_tensors, refilled_tensors, strict=True)
    ):
        raise RuntimeError(
            "FSDP compute-representation refill replaced managed storage"
        )


class _FSDPWeightBase(torch.Tensor):
    """Logical high-precision metadata shared by both lifecycle states."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, *args: Any, **kwargs: Any):
        del args
        return torch.Tensor._make_wrapper_subclass(
            cls,
            kwargs.get("_logical_size", tensor.size()),
            strides=kwargs.get("_logical_stride", tensor.stride()),
            storage_offset=kwargs.get(
                "_logical_storage_offset", tensor.storage_offset()
            ),
            dtype=kwargs.get("_logical_dtype", tensor.dtype),
            layout=tensor.layout,
            device=kwargs.get("_logical_device", tensor.device),
            pin_memory=tensor.is_pinned(),
            requires_grad=kwargs.get("_logical_requires_grad", tensor.requires_grad),
        )


class _ShardedFSDPWeight(_FSDPWeightBase):
    """Persistent high-precision parameter that owns the FSDP hooks.

    This is the sharded half of the lifecycle. It is the ``nn.Parameter`` the
    optimizer updates and the checkpoint stores, it holds the high-precision
    shard in ``_tensor``, and it lives for the whole run. It is *not* what
    compute sees under FSDP: the post-all-gather hook hands back a
    :class:`_ComputeFSDPWeight` holding the quantized operands, and that is
    what forward and backward read for the duration of one unshard. See the
    instance timeline at the top of this module for how the two hand off.

    **This is the class a format subclasses**, because a format differs only
    in how it turns a high-precision weight into operands, and this is the
    side that produces them. ``_ComputeFSDPWeight`` only holds them, which is
    format-independent, so it is generic and is never subclassed.

    A subclass supplies:

    * a frozen dataclass of the tensors one unshard lifetime owns, whose
      fields are distinct allocations -- derived views belong in properties;
    * ``_build_compute_representation(logical_weight, out=None)``, which
      allocates a new representation when ``out`` is None and otherwise
      refills ``out``'s existing tensors in place.

    Everything else -- flattening, refill, reshard, and the SimpleFSDP bridge
    -- is inherited.
    """

    def __init__(self, tensor: torch.Tensor, **logical_metadata: Any) -> None:
        del logical_metadata
        self._tensor = tensor

    def __tensor_flatten__(self):
        return ["_tensor"], (self.dtype,)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        del metadata, outer_size, outer_stride
        return cls(inner_tensors["_tensor"])

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        del types
        template = None
        preserve_wrapper = func in _FSDP_SHARDED_OPS

        def unwrap(tensor: _ShardedFSDPWeight) -> torch.Tensor:
            nonlocal template
            if template is None:
                template = tensor
            elif preserve_wrapper and type(tensor) is not type(template):
                raise RuntimeError("FSDP operation mixed sharded weight types")
            return tensor._tensor

        output = func(
            *pytree.tree_map_only(cls, unwrap, args or ()),
            **pytree.tree_map_only(cls, unwrap, kwargs or {}),
        )
        if not preserve_wrapper:
            return output
        assert template is not None
        return pytree.tree_map_only(torch.Tensor, type(template), output)

    def _build_compute_representation(
        self,
        logical_weight: torch.Tensor,
        out: Any = None,
    ) -> Any:
        """Quantize ``logical_weight``, into ``out``'s tensors when refilling."""
        raise NotImplementedError

    def _build_compute_weight(self, unsharded_weight: torch.Tensor):
        """Build a storage-free compute parameter from an unsharded weight.

        FSDP2 reaches its compute weight through ``fsdp_post_all_gather``.
        GraphTrainer's SimpleFSDP reconstructs the unsharded weight itself and
        calls this directly, wrapped in an autograd function that owns the
        gradient edge FSDP2 would otherwise create.

        Unlike the sharded parameter, which is always a DTensor, the unsharded
        weight reaches here both bare and wrapped: SimpleFSDP returns a plain
        local tensor on a pure data-parallel mesh and re-wraps on the
        non-data-parallel mesh when composed with TP or EP. Mirror whichever
        it was, so the compute weight is substitutable for it.
        """
        if not isinstance(unsharded_weight, DTensor):
            with torch.no_grad():
                return _ComputeFSDPWeight(
                    unsharded_weight,
                    self._build_compute_representation(unsharded_weight),
                )
        local_weight = unsharded_weight._local_tensor
        with torch.no_grad():
            representation = self._build_compute_representation(local_weight)
        return DTensor.from_local(
            _ComputeFSDPWeight(local_weight, representation),
            unsharded_weight.device_mesh,
            unsharded_weight.placements,
            run_check=False,
            shape=unsharded_weight.shape,
            stride=unsharded_weight.stride(),
        )

    def fsdp_should_release_all_gather_outputs_after_post_all_gather(self) -> bool:
        """Release the high-precision all-gather output after state construction."""
        return True

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        """Return the high-precision communication tensor."""
        del outer_stride, module
        if outer_size[0] % mesh.size() != 0:
            raise ValueError(
                "FSDP compute weights require dimension 0 to be evenly divisible "
                "by the FSDP shard mesh size"
            )
        dtype = mp_policy.param_dtype or self._tensor.dtype
        return (self._tensor.to(dtype),), None

    def fsdp_post_all_gather(
        self, all_gather_outputs, metadata, param_dtype, *, out=None
    ):
        """Create or refill the compute weight representation after all-gather."""
        del metadata, param_dtype
        (gathered_weight,) = all_gather_outputs

        # On the first unshard, FSDP has no compute-weight container or managed
        # tensors yet. Build both and return them to FSDP. With RAF=False, FSDP
        # keeps this representation alive through forward and backward.
        if out is None:
            with torch.no_grad():
                representation = self._build_compute_representation(gathered_weight)
            return (
                _ComputeFSDPWeight(gathered_weight, representation),
                _representation_tensors(representation),
            )

        # After FSDP releases and later unshards the weight again, ``out`` is
        # the same compute-weight object returned above. This occurs between
        # forward and backward with RAF=True, or after a later reshard. Refill
        # the same managed tensor objects so existing module and autograd
        # references remain valid.
        target = out._local_tensor if isinstance(out, DTensor) else out
        if not isinstance(target, _ComputeFSDPWeight):
            raise RuntimeError("FSDP output does not own a compute representation")
        existing = target.compute_representation
        managed_tensors = _representation_tensors(existing)
        with (
            torch.no_grad(),
            # Refilling lifecycle-managed storage is not a user-visible tensor
            # mutation and must not invalidate saved-tensor version checks.
            torch.autograd._unsafe_preserve_version_counter(managed_tensors),
        ):
            refilled = self._build_compute_representation(gathered_weight, out=existing)
        _validate_refilled_tensor_identity(
            managed_tensors, _representation_tensors(refilled)
        )
        target._compute_representation = refilled


class _ComputeFSDPWeight(_FSDPWeightBase):
    """Unsharded weight holding one unshard lifetime's format operands.

    The unsharded half of the lifecycle, built by
    :class:`_ShardedFSDPWeight`'s post-all-gather hook and alive until the
    final reshard. Carries no high-precision storage -- that is what lets FSDP
    release the all-gather output -- so reading it as a high-precision tensor
    is an error; only format-aware consumers may read
    ``compute_representation``. See the instance timeline at the top of this
    module for how the two classes hand off.

    Generic by design: the FSDP-managed inner tensors come from the
    representation's dataclass fields, which works for any format. Do not
    subclass it; formats subclass :class:`_ShardedFSDPWeight` instead.
    """

    def __init__(
        self,
        metadata_source: torch.Tensor,
        compute_representation: Any,
        **logical_metadata: Any,
    ) -> None:
        del metadata_source, logical_metadata
        self._compute_representation = compute_representation
        # __tensor_flatten__ reports inner tensors by attribute name and the
        # subclass machinery fetches them with a plain getattr, so each one has
        # to exist as an attribute here -- reaching into the representation is
        # not an option. Mirror rather than copy: these are the same tensor
        # objects, so an in-place refill updates both views, and a refill that
        # substituted objects is rejected by
        # _validate_refilled_tensor_identity.
        for name in _representation_field_names(type(compute_representation)):
            setattr(self, f"_{name}", getattr(compute_representation, name))

    def __tensor_flatten__(self):
        representation_cls = type(self._compute_representation)
        names = [f"_{name}" for name in _representation_field_names(representation_cls)]
        return names, (representation_cls, self.dtype)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        representation_cls, dtype = metadata
        operands = [
            inner_tensors[f"_{name}"]
            for name in _representation_field_names(representation_cls)
        ]
        representation = representation_cls(*operands)
        # FSDP supplies the logical shape; any managed tensor can stand in for
        # the rest, since they share the compute weight's device and layout.
        return _ComputeFSDPWeight(
            operands[0],
            representation,
            _logical_size=outer_size,
            _logical_stride=outer_stride,
            _logical_dtype=dtype,
        )

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        del types
        template = None

        def unwrap(tensor: _ComputeFSDPWeight) -> torch.Tensor:
            nonlocal template
            if template is None:
                template = tensor
            elif tensor._compute_representation is not template._compute_representation:
                raise RuntimeError(
                    "FSDP operation mixed compute weight representations"
                )
            # There is no high-precision storage to hand the op; a meta tensor
            # carries the logical metadata that view ops need.
            return torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device="meta",
                requires_grad=tensor.requires_grad,
            )

        def wrap_view(tensor: torch.Tensor):
            assert template is not None
            representation = template._compute_representation
            # __new__ reads layout and pinning off a real tensor, and the
            # template has no storage to answer with, so borrow a managed
            # tensor for those two and give the view's logical metadata for
            # everything else. Which managed tensor does not matter: they
            # share the compute weight's device, layout, and pinning.
            layout_source = _representation_tensors(representation)[0]
            return _ComputeFSDPWeight(
                layout_source,
                representation,
                _logical_size=tensor.size(),
                _logical_stride=tensor.stride(),
                _logical_storage_offset=tensor.storage_offset(),
                _logical_dtype=template.dtype,
                _logical_device=template.device,
                _logical_requires_grad=tensor.requires_grad,
            )

        original_args, original_kwargs = args, kwargs or {}
        args, kwargs = pytree.tree_map_only(
            cls, unwrap, (original_args, original_kwargs)
        )
        assert template is not None
        if func in _FSDP_COMPUTE_FACTORY_OPS:
            kwargs["device"] = template.device
            return func(*args, **kwargs)
        if func not in _FSDP_COMPUTE_VIEW_OPS:
            raise RuntimeError(
                f"{func} attempted to read a storage-free FSDP compute weight"
            )
        wrapped = pytree.tree_map_only(torch.Tensor, wrap_view, func(*args, **kwargs))
        return return_and_correct_aliasing(
            func, original_args, original_kwargs, wrapped
        )

    @property
    def compute_representation(self) -> Any:
        """Return the representation for the current unshard lifetime."""
        return self._compute_representation


class _BuildComputeWeightFunction(torch.autograd.Function):
    """Own the gradient edge for a compute weight built outside FSDP2.

    FSDP2 creates this edge internally for its post-all-gather output.
    GraphTrainer's SimpleFSDP reconstructs the unsharded weight itself and so
    has no such edge, and this routes its logical weight gradient straight
    back through the high-precision gather.

    Stays here rather than under GraphTrainer, its only user: this module
    already owns the equivalent FSDP2 edge, and keeping both in one place is
    what makes them comparable. Reached through :func:`build_compute_weight`.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, weight: torch.Tensor, wrapper: _ShardedFSDPWeight):
        del ctx
        return wrapper._build_compute_weight(weight)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_weight: torch.Tensor):
        del ctx
        return grad_weight, None


def build_compute_weight(
    unsharded_weight: torch.Tensor, sharded_parameter: torch.Tensor
) -> torch.Tensor:
    """Derive the compute weight a data parallel implementation should use.

    Both halves of the lifecycle are arguments because neither alone
    suffices: ``sharded_parameter`` is the persistent parameter, whose tensor
    subclass determines which representation to build, and
    ``unsharded_weight`` is what the data-parallel stage produced from it,
    which is the value to quantize. That stage returns a plain local tensor,
    so the subclass is unreachable from its output.

    A parameter carrying no compute representation passes through unchanged,
    so a caller may apply this unconditionally.

    GraphTrainer's SimpleFSDP is the only caller today, so this supports what
    SimpleFSDP does and nothing more: ``sharded_parameter`` must be the
    distributed parameter. FSDP2 never reaches here -- it drives the same
    construction through ``fsdp_post_all_gather`` and creates the gradient
    edge itself.
    """
    # SimpleFSDP distributes every parameter before installing its
    # parametrization, so the sharded parameter is always a DTensor. Require
    # that rather than accepting a plain tensor no caller passes.
    if not isinstance(sharded_parameter, DTensor):
        raise RuntimeError(
            "build_compute_weight() expects the distributed sharded "
            f"parameter; got a plain {type(sharded_parameter).__name__}."
        )
    source = sharded_parameter._local_tensor
    if isinstance(source, _ComputeFSDPWeight):
        raise RuntimeError(
            "build_compute_weight() received an already-unsharded weight. "
            "FSDP2 builds the compute weight in fsdp_post_all_gather and owns "
            "its gradient edge, so calling this as well would quantize the "
            "weight a second time."
        )
    if not isinstance(source, _ShardedFSDPWeight):
        return unsharded_weight
    return _BuildComputeWeightFunction.apply(unsharded_weight, source)
