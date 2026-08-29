# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Versioned RWKV7 State Tuning artifacts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from torchtitan.models.rwkv7.adapter import base_config_sha256


STATE_ARTIFACT_FORMAT = "torchtitan_rwkv_state"
STATE_ARTIFACT_VERSION = 1

_STATE_KEY_PATTERN = re.compile(
    r"^layers\.(?P<layer>\d+)\."
    r"(?:(?:linear_attn\.(?P<time>initial_shift|initial_wkv_state))|"
    r"(?:mlp\.(?P<channel>initial_shift)))$"
)


def state_tuning_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state_dict.items()
        if _STATE_KEY_PATTERN.fullmatch(key)
    }


def build_state_artifact_config(
    state_dict: Mapping[str, torch.Tensor],
    *,
    base_model_path: str | os.PathLike[str],
) -> dict[str, Any]:
    if not state_dict:
        raise ValueError("RWKV7 State Tuning artifact state is empty.")
    with open(Path(base_model_path) / "config.json") as file:
        base_config = json.load(file)
    return {
        "format": STATE_ARTIFACT_FORMAT,
        "version": STATE_ARTIFACT_VERSION,
        "architecture": "rwkv7",
        "base_model": Path(base_model_path).name,
        "base_config_sha256": base_config_sha256(base_model_path),
        "num_layers": base_config["num_hidden_layers"],
        "hidden_size": base_config["hidden_size"],
        "head_size": base_config["head_size"],
        "shift_dtype": "bfloat16",
        "wkv_state_dtype": "float32",
    }


def validate_state_artifact(
    artifact_path: str | os.PathLike[str],
    *,
    base_model_path: str | os.PathLike[str],
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    artifact_path = Path(artifact_path)
    with open(artifact_path / "state_config.json") as file:
        metadata = json.load(file)
    with open(Path(base_model_path) / "config.json") as file:
        base_config = json.load(file)
    expected_metadata = {
        "format": STATE_ARTIFACT_FORMAT,
        "version": STATE_ARTIFACT_VERSION,
        "architecture": "rwkv7",
        "base_model": Path(base_model_path).name,
        "base_config_sha256": base_config_sha256(base_model_path),
        "num_layers": base_config["num_hidden_layers"],
        "hidden_size": base_config["hidden_size"],
        "head_size": base_config["head_size"],
        "shift_dtype": "bfloat16",
        "wkv_state_dtype": "float32",
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"RWKV7 State Tuning metadata mismatch: {mismatches}.")

    tensor_path = artifact_path / "state_model.safetensors"
    if not tensor_path.is_file():
        raise ValueError(f"RWKV7 State Tuning artifact is missing {tensor_path}.")
    actual_shapes = {}
    actual_dtypes = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            if _STATE_KEY_PATTERN.fullmatch(key) is None:
                raise ValueError(f"Invalid RWKV7 State Tuning tensor key {key!r}.")
            actual_shapes[key] = tuple(tensors.get_slice(key).get_shape())
            actual_dtypes[key] = tensors.get_slice(key).get_dtype()

    num_layers = base_config["num_hidden_layers"]
    hidden_size = base_config["hidden_size"]
    head_size = base_config["head_size"]
    num_heads = hidden_size // head_size
    expected = {}
    for layer_id in range(num_layers):
        expected.update(
            {
                f"layers.{layer_id}.linear_attn.initial_shift": (hidden_size,),
                f"layers.{layer_id}.linear_attn.initial_wkv_state": (
                    num_heads,
                    head_size,
                    head_size,
                ),
                f"layers.{layer_id}.mlp.initial_shift": (hidden_size,),
            }
        )
    if set(actual_shapes) != set(expected):
        raise ValueError(
            "RWKV7 State Tuning tensor keys do not match the base model: "
            f"missing={sorted(set(expected) - set(actual_shapes))}, "
            f"unexpected={sorted(set(actual_shapes) - set(expected))}."
        )
    bad_shapes = {
        key: {"expected": expected[key], "actual": actual_shapes[key]}
        for key in expected
        if expected[key] != actual_shapes[key]
    }
    if bad_shapes:
        raise ValueError(f"RWKV7 State Tuning tensor shape mismatch: {bad_shapes}.")
    bad_dtypes = {}
    for key, dtype in actual_dtypes.items():
        expected_dtype = "F32" if key.endswith("initial_wkv_state") else "BF16"
        if dtype != expected_dtype:
            bad_dtypes[key] = {"expected": expected_dtype, "actual": dtype}
    if bad_dtypes:
        raise ValueError(f"RWKV7 State Tuning tensor dtype mismatch: {bad_dtypes}.")

    if expected_shapes is not None:
        if dict(expected_shapes) != actual_shapes:
            raise ValueError(
                "RWKV7 State Tuning artifact does not match the configured model."
            )
    return metadata, actual_shapes


def save_state_artifact(
    artifact_path: str | os.PathLike[str],
    state_dict: Mapping[str, torch.Tensor],
    *,
    base_model_path: str | os.PathLike[str],
) -> None:
    artifact_path = Path(artifact_path)
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise ValueError(
            f"State artifact output directory is not empty: {artifact_path}."
        )
    artifact_path.mkdir(parents=True, exist_ok=True)
    tensors = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in state_tuning_state_dict(state_dict).items()
    }
    config = build_state_artifact_config(
        tensors,
        base_model_path=base_model_path,
    )
    with open(artifact_path / "state_config.json", "w") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")
    save_file(tensors, artifact_path / "state_model.safetensors")


def load_state_artifact(
    artifact_path: str | os.PathLike[str],
    *,
    base_model_path: str | os.PathLike[str],
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, torch.Tensor]:
    validate_state_artifact(
        artifact_path,
        base_model_path=base_model_path,
        expected_shapes=expected_shapes,
    )
    return load_file(Path(artifact_path) / "state_model.safetensors", device="cpu")


__all__ = [
    "build_state_artifact_config",
    "load_state_artifact",
    "save_state_artifact",
    "STATE_ARTIFACT_FORMAT",
    "STATE_ARTIFACT_VERSION",
    "state_tuning_state_dict",
    "validate_state_artifact",
]
