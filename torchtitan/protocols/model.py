# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import torch

from torchtitan.config import Configurable, ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims

from .module import Module


_frozen_config_class_cache: dict[
    type[Module.Config],
    type[Module.Config],
] = {}


def make_frozen_config(config: Module.Config) -> Module.Config:
    """Create a config that freezes the module parameters it directly owns."""
    config_cls = type(config)
    frozen_cls = _frozen_config_class_cache.get(config_cls)
    if frozen_cls is None:

        class FrozenConfig(config_cls):  # type: ignore[valid-type, misc]
            def build(self, **kwargs):
                instance = config_cls.build(self, **kwargs)
                for parameter in instance.parameters(recurse=False):
                    parameter.requires_grad_(False)
                return instance

        FrozenConfig.__name__ = f"Frozen{config_cls.__name__}"
        FrozenConfig.__qualname__ = f"Frozen{config_cls.__qualname__}"
        frozen_cls = FrozenConfig
        _frozen_config_class_cache[config_cls] = frozen_cls

    return frozen_cls(
        **{
            field.name: getattr(config, field.name)
            for field in fields(config)
            if field.init
        }
    )


class ModelConfigConverter(Configurable):
    """Base class for converters that transform the model config tree.

    Subclasses implement ``convert()`` to modify configs before model build
    (e.g. quantization, LoRA).  Converters may return a replacement root
    config when the transform needs to wrap the model config itself.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        incompatible_converter_types: ClassVar[tuple[type, ...]] = ()
        """Converter config types that cannot be combined with this converter."""

    @abstractmethod
    def convert(self, model_config: Module.Config) -> Module.Config:
        raise NotImplementedError


class BaseModel(Module):
    """Base class for all model classes.

    Models inherit from BaseModel (which is Module = nn.Module + Configurable).
    Each model defines a nested Config(BaseModel.Config) with model hyperparameters.
    The model is constructed via ``config.build()``.

    ``init_states`` (from Module) auto-recurses; override only for custom
    ordering (e.g., weight tying before init).
    """

    def init_weights(self, **kwargs) -> None:
        """Backward-compatible alias for ``init_states``.

        External tools (e.g., AutoParallel) wrap ``init_weights`` with
        DTensor-aware interception. This alias ensures they can find it.
        """
        # TODO: remove this once autoparallel has wrap_init_states
        buffer_device = kwargs.get("buffer_device")
        self.init_states(buffer_device=buffer_device)

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Prepare the forward inputs from a dataloader batch.

        Models driven by the standard trainer/validator forward path implement
        this to build any attention masks, apply context-parallel sharding and
        SPMD annotation as needed, and split ``input``/``labels`` out of the
        batch. ``input_dict`` is the batch with ``labels`` folded in; return
        ``(inputs, labels, extra_kwargs)``.

        The trainer calls this via ``cast(BaseModel, model).preprocess_inputs``,
        so the declaration lives here for typing. There is no meaningful default:
        models with a bespoke pipeline (e.g. Flux) never call it, and every model
        that does must override it -- hence the ``NotImplementedError`` below.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement preprocess_inputs()."
        )

    def verify_module_protocol(self) -> None:
        """Verify all submodules satisfy the ``Module`` protocol.

        Catches non-``Module`` submodules early with a clear error message,
        preventing obscure failures when the ``Module`` protocol is being
        used later.

        Override in models where some internal ``nn.Module`` submodules
        cannot conform to the ``Module`` protocol.
        """
        failures: list[tuple[str, str]] = []
        for fqn, mod in self.named_modules():
            if not isinstance(mod, Module):
                failures.append((fqn, type(mod).__name__))
        if failures:
            details = ", ".join(f"'{fqn}' ({cls})" for fqn, cls in failures)
            raise RuntimeError(
                f"The following modules do not satisfy the Module protocol: {details}"
            )

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Base config for all models.

        Subclasses define model-specific hyperparameters.
        """

        # TODO: This function violates encapsulation;
        # maybe replace it with config passes from outside.
        @abstractmethod
        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            pass

        @abstractmethod
        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            pass
