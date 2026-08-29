# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fail-closed access to the pinned FlashRWKV2 training API."""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType

import torch


FLASH_RWKV2_VERSION = "0.1.0a11"

PRETRAIN_ATTENTION_OPERATORS = (
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_readout_bf16",
    "pretrain_tmix_tokenshift_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "pretrain_tmix_wkv7_recurrent_bf16",
)
PRETRAIN_FEED_FORWARD_OPERATORS = ("pretrain_cmix_bf16",)
STATE_TUNING_ATTENTION_OPERATORS = (
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_readout_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "statetune_tmix_tokenshift_bf16",
    "statetune_tmix_wkv7_recurrent_fp32io16",
)
STATE_TUNING_FEED_FORWARD_OPERATORS = ("statetune_cmix_bf16",)


@lru_cache(maxsize=None)
def _load_and_validate(
    operators: tuple[str, ...],
    mode: str,
) -> ModuleType:
    try:
        module = importlib.import_module("flashrwkv2")
    except ImportError as error:
        raise RuntimeError(
            f"RWKV-7 {mode} requires FlashRWKV2=={FLASH_RWKV2_VERSION}; "
            f"import failed: {error}"
        ) from error

    version = getattr(module, "__version__", "unknown")
    source = getattr(module, "__file__", "unknown")
    if version != FLASH_RWKV2_VERSION:
        raise RuntimeError(
            f"RWKV-7 {mode} requires FlashRWKV2=={FLASH_RWKV2_VERSION}; "
            f"installed version={version}, source={source}."
        )

    missing = [name for name in operators if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            f"RWKV-7 {mode} requires FlashRWKV2 public operators {missing}; "
            f"installed version={version}, source={source}."
        )
    return module


def load_flash_rwkv2(
    operators: tuple[str, ...],
    tensor: torch.Tensor,
    mode: str,
    *,
    preloaded: ModuleType | None = None,
) -> ModuleType:
    """Load the pinned provider after validating the execution boundary."""
    if not tensor.is_cuda:
        raise RuntimeError(
            f"RWKV-7 {mode} requires CUDA tensors; got device={tensor.device}, "
            f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}."
        )
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"RWKV-7 {mode} requires bfloat16 tensors, got {tensor.dtype}.")
    return preloaded if preloaded is not None else _load_and_validate(operators, mode)


def preload_flash_rwkv2(
    operators: tuple[str, ...],
    mode: str,
) -> ModuleType:
    """Validate and cache one operator group before full-graph compilation."""
    return _load_and_validate(operators, mode)


__all__ = [
    "FLASH_RWKV2_VERSION",
    "PRETRAIN_ATTENTION_OPERATORS",
    "PRETRAIN_FEED_FORWARD_OPERATORS",
    "STATE_TUNING_ATTENTION_OPERATORS",
    "STATE_TUNING_FEED_FORWARD_OPERATORS",
    "load_flash_rwkv2",
    "preload_flash_rwkv2",
]
