# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RWKV7 native LoRA artifacts, PEFT conversion, and base-weight merging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


NATIVE_ADAPTER_FORMAT = "torchtitan_rwkv_lora"
NATIVE_ADAPTER_VERSION = 1
RWKV7_LORA_TARGETS = ("r_proj", "k_proj", "v_proj", "o_proj")

_NATIVE_KEY_PATTERN = re.compile(
    r"^layers\.(?P<layer>\d+)\.linear_attn\."
    r"(?P<target>r_proj|k_proj|v_proj|o_proj)\."
    r"lora_(?P<side>[ab])\.weight$"
)
_PEFT_KEY_PATTERN = re.compile(
    r"^base_model\.model\.model\.layers\.(?P<layer>\d+)\.linear_attn\."
    r"(?P<target>r_proj|k_proj|v_proj|o_proj)\."
    r"lora_(?P<side>[AB])\.weight$"
)


def _native_key_match(key: str) -> re.Match[str]:
    match = _NATIVE_KEY_PATTERN.fullmatch(key)
    if match is None:
        raise ValueError(f"Invalid RWKV7 native adapter tensor key {key!r}.")
    return match


def _read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path) as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    with open(path, "w") as file:
        json.dump(dict(value), file, indent=2, sort_keys=True)
        file.write("\n")


def base_config_sha256(base_model_path: str | os.PathLike[str]) -> str:
    config_path = Path(base_model_path) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Base model is missing {config_path}.")
    digest = hashlib.sha256()
    with open(config_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in state_dict.items()
        if _NATIVE_KEY_PATTERN.fullmatch(key)
    }


def build_native_adapter_config(
    state_dict: Mapping[str, torch.Tensor],
    *,
    base_model_path: str | os.PathLike[str],
    alpha: float,
    architecture: str = "rwkv7",
) -> dict[str, Any]:
    if not state_dict:
        raise ValueError("RWKV7 native adapter state is empty.")

    ranks = set()
    targets = set()
    dtypes = set()
    for key, tensor in state_dict.items():
        match = _native_key_match(key)
        targets.add(match.group("target"))
        dtypes.add(str(tensor.dtype).removeprefix("torch."))
        if match.group("side") == "a":
            ranks.add(tensor.shape[0])
        else:
            ranks.add(tensor.shape[1])

    if len(ranks) != 1:
        raise ValueError(f"RWKV7 adapter tensors have inconsistent ranks: {ranks}.")
    if len(dtypes) != 1:
        raise ValueError(f"RWKV7 adapter tensors have inconsistent dtypes: {dtypes}.")
    if targets != set(RWKV7_LORA_TARGETS):
        raise ValueError(
            "RWKV7 native adapter targets must be exactly "
            f"{list(RWKV7_LORA_TARGETS)}, got {sorted(targets)}."
        )

    rank = ranks.pop()
    base_model_path = Path(base_model_path)
    return {
        "format": NATIVE_ADAPTER_FORMAT,
        "version": NATIVE_ADAPTER_VERSION,
        "architecture": architecture,
        "base_model": base_model_path.name,
        "base_config_sha256": base_config_sha256(base_model_path),
        "rank": rank,
        "alpha": alpha,
        "targets": list(RWKV7_LORA_TARGETS),
        "dtype": dtypes.pop(),
    }


def _validate_native_metadata(
    metadata: Mapping[str, Any],
    *,
    base_model_path: str | os.PathLike[str],
) -> None:
    expected = {
        "format": NATIVE_ADAPTER_FORMAT,
        "version": NATIVE_ADAPTER_VERSION,
        "architecture": "rwkv7",
        "base_model": Path(base_model_path).name,
        "base_config_sha256": base_config_sha256(base_model_path),
        "targets": list(RWKV7_LORA_TARGETS),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"RWKV7 native adapter metadata mismatch: {mismatches}.")
    if not isinstance(metadata.get("rank"), int) or metadata["rank"] <= 0:
        raise ValueError("RWKV7 native adapter rank must be a positive integer.")
    if not isinstance(metadata.get("alpha"), (int, float)):
        raise ValueError("RWKV7 native adapter alpha must be numeric.")
    if metadata.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError(
            f"RWKV7 native adapter has unsupported dtype {metadata.get('dtype')!r}."
        )


def validate_native_adapter(
    adapter_path: str | os.PathLike[str],
    *,
    base_model_path: str | os.PathLike[str],
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    adapter_path = Path(adapter_path)
    metadata = _read_json(adapter_path / "adapter_config.json")
    _validate_native_metadata(metadata, base_model_path=base_model_path)

    tensor_path = adapter_path / "adapter_model.safetensors"
    if not tensor_path.is_file():
        raise ValueError(f"RWKV7 native adapter is missing {tensor_path}.")
    actual_shapes = {}
    actual_dtypes = {}
    layers = set()
    targets_by_layer: dict[int, set[str]] = {}
    sides_by_module: dict[tuple[int, str], set[str]] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            match = _NATIVE_KEY_PATTERN.fullmatch(key)
            if match is None:
                raise ValueError(f"Invalid RWKV7 native adapter tensor key {key!r}.")
            tensor_slice = tensors.get_slice(key)
            shape = tuple(tensor_slice.get_shape())
            actual_shapes[key] = shape
            actual_dtypes[key] = tensor_slice.get_dtype()
            layer_id = int(match.group("layer"))
            target = match.group("target")
            side = match.group("side")
            layers.add(layer_id)
            targets_by_layer.setdefault(layer_id, set()).add(target)
            sides_by_module.setdefault((layer_id, target), set()).add(side)

    base_config = _read_json(Path(base_model_path) / "config.json")
    num_layers = base_config.get("num_hidden_layers")
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("RWKV7 base config requires positive num_hidden_layers.")
    if layers != set(range(num_layers)):
        raise ValueError(
            "RWKV7 native adapter layer set does not match the base model: "
            f"expected={list(range(num_layers))}, actual={sorted(layers)}."
        )
    for layer_id in layers:
        if targets_by_layer[layer_id] != set(RWKV7_LORA_TARGETS):
            raise ValueError(
                f"RWKV7 adapter layer {layer_id} has targets "
                f"{sorted(targets_by_layer[layer_id])}."
            )
        for target in RWKV7_LORA_TARGETS:
            if sides_by_module[(layer_id, target)] != {"a", "b"}:
                raise ValueError(
                    f"RWKV7 adapter layer {layer_id} target {target} requires A and B."
                )

    rank = metadata["rank"]
    hidden_size = base_config.get("hidden_size")
    for key, shape in actual_shapes.items():
        side = _native_key_match(key).group("side")
        expected_shape = (rank, hidden_size) if side == "a" else (hidden_size, rank)
        if shape != expected_shape:
            raise ValueError(
                f"RWKV7 adapter tensor {key} has shape {shape}, expected {expected_shape}."
            )

    expected_dtype = {
        "bfloat16": "BF16",
        "float16": "F16",
        "float32": "F32",
    }[metadata["dtype"]]
    bad_dtypes = {
        key: {"expected": expected_dtype, "actual": dtype}
        for key, dtype in actual_dtypes.items()
        if dtype != expected_dtype
    }
    if bad_dtypes:
        raise ValueError(f"RWKV7 adapter tensor dtype mismatch: {bad_dtypes}.")

    if expected_shapes is not None:
        if set(actual_shapes) != set(expected_shapes):
            raise ValueError(
                "RWKV7 adapter keys do not match the configured LoRA model: "
                f"missing={sorted(set(expected_shapes) - set(actual_shapes))}, "
                f"unexpected={sorted(set(actual_shapes) - set(expected_shapes))}."
            )
        bad_shapes = {
            key: {"expected": expected_shapes[key], "actual": actual_shapes[key]}
            for key in expected_shapes
            if expected_shapes[key] != actual_shapes[key]
        }
        if bad_shapes:
            raise ValueError(
                f"RWKV7 adapter shapes do not match the configured model: {bad_shapes}."
            )
    return metadata, actual_shapes


def save_native_adapter(
    adapter_path: str | os.PathLike[str],
    state_dict: Mapping[str, torch.Tensor],
    *,
    base_model_path: str | os.PathLike[str],
    alpha: float,
) -> None:
    adapter_path = Path(adapter_path)
    if adapter_path.exists() and any(adapter_path.iterdir()):
        raise ValueError(f"Adapter output directory is not empty: {adapter_path}.")
    adapter_path.mkdir(parents=True, exist_ok=True)
    tensors = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in adapter_state_dict(state_dict).items()
    }
    metadata = build_native_adapter_config(
        tensors,
        base_model_path=base_model_path,
        alpha=alpha,
    )
    _write_json(adapter_path / "adapter_config.json", metadata)
    save_file(tensors, adapter_path / "adapter_model.safetensors")


def load_native_adapter(
    adapter_path: str | os.PathLike[str],
    *,
    base_model_path: str | os.PathLike[str],
    expected_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, torch.Tensor]:
    validate_native_adapter(
        adapter_path,
        base_model_path=base_model_path,
        expected_shapes=expected_shapes,
    )
    return load_file(Path(adapter_path) / "adapter_model.safetensors", device="cpu")


def _native_to_peft_key(key: str) -> str:
    match = _native_key_match(key)
    return (
        "base_model.model.model.layers."
        f"{match.group('layer')}.linear_attn.{match.group('target')}."
        f"lora_{match.group('side').upper()}.weight"
    )


def _peft_to_native_key(key: str) -> str:
    match = _PEFT_KEY_PATTERN.fullmatch(key)
    if match is None:
        raise ValueError(f"Invalid RWKV7 PEFT adapter tensor key {key!r}.")
    return (
        f"layers.{match.group('layer')}.linear_attn.{match.group('target')}."
        f"lora_{match.group('side').lower()}.weight"
    )


def native_to_peft(
    native_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    base_model_path: str | os.PathLike[str],
) -> None:
    metadata, _ = validate_native_adapter(
        native_path,
        base_model_path=base_model_path,
    )
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"PEFT output directory is not empty: {output_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    native_tensors = load_file(
        Path(native_path) / "adapter_model.safetensors",
        device="cpu",
    )
    peft_tensors = {
        _native_to_peft_key(key): value.contiguous()
        for key, value in native_tensors.items()
    }
    peft_config = {
        "base_model_name_or_path": metadata["base_model"],
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": metadata["alpha"],
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": metadata["rank"],
        "target_modules": metadata["targets"],
        "task_type": "CAUSAL_LM",
    }
    _write_json(output_path / "adapter_config.json", peft_config)
    save_file(peft_tensors, output_path / "adapter_model.safetensors")


def peft_to_native(
    peft_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    base_model_path: str | os.PathLike[str],
) -> None:
    peft_path = Path(peft_path)
    peft_config = _read_json(peft_path / "adapter_config.json")
    expected_config = {
        "peft_type": "LORA",
        "bias": "none",
        "fan_in_fan_out": False,
        "lora_dropout": 0.0,
    }
    mismatches = {
        key: {"expected": value, "actual": peft_config.get(key)}
        for key, value in expected_config.items()
        if peft_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"RWKV7 PEFT adapter config mismatch: {mismatches}.")
    if set(peft_config.get("target_modules", [])) != set(RWKV7_LORA_TARGETS):
        raise ValueError(
            "RWKV7 PEFT adapter target_modules must be exactly "
            f"{list(RWKV7_LORA_TARGETS)}."
        )
    rank = peft_config.get("r")
    alpha = peft_config.get("lora_alpha")
    if not isinstance(rank, int) or rank <= 0 or not isinstance(alpha, (int, float)):
        raise ValueError(
            "RWKV7 PEFT adapter requires positive r and numeric lora_alpha."
        )

    peft_tensors = load_file(peft_path / "adapter_model.safetensors", device="cpu")
    native_tensors = {
        _peft_to_native_key(key): value.contiguous()
        for key, value in peft_tensors.items()
    }
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"Native output directory is not empty: {output_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = build_native_adapter_config(
        native_tensors,
        base_model_path=base_model_path,
        alpha=float(alpha),
    )
    if metadata["rank"] != rank:
        raise ValueError(
            "RWKV7 PEFT adapter tensor rank does not match adapter_config.json: "
            f"tensors={metadata['rank']}, config={rank}."
        )
    _write_json(output_path / "adapter_config.json", metadata)
    save_file(native_tensors, output_path / "adapter_model.safetensors")
    validate_native_adapter(output_path, base_model_path=base_model_path)


def _hf_base_weight_key(native_key: str) -> str:
    match = _native_key_match(native_key)
    return (
        f"model.layers.{match.group('layer')}.linear_attn."
        f"{match.group('target')}.weight"
    )


def merge_native_adapter(
    base_model_path: str | os.PathLike[str],
    native_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> None:
    metadata, _ = validate_native_adapter(
        native_path,
        base_model_path=base_model_path,
    )
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"Merged output directory is not empty: {output_path}.")
    output_path.mkdir(parents=True, exist_ok=True)

    base_model_path = Path(base_model_path)
    for source in base_model_path.iterdir():
        destination = output_path / source.name
        if source.suffix == ".safetensors":
            continue
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    adapter_tensors = load_file(
        Path(native_path) / "adapter_model.safetensors",
        device="cpu",
    )
    deltas: dict[
        str,
        tuple[torch.Tensor | None, torch.Tensor | None],
    ] = {}
    for key, tensor in adapter_tensors.items():
        base_key = _hf_base_weight_key(key)
        side = _native_key_match(key).group("side")
        adapter_a, adapter_b = deltas.get(base_key, (None, None))
        if side == "a":
            adapter_a = tensor
        else:
            adapter_b = tensor
        deltas[base_key] = (adapter_a, adapter_b)

    merged_keys = set()
    scaling = float(metadata["alpha"]) / metadata["rank"]
    for source in sorted(base_model_path.glob("*.safetensors")):
        with safe_open(source, framework="pt", device="cpu") as tensors:
            shard_metadata = tensors.metadata()
            shard = {key: tensors.get_tensor(key) for key in tensors.keys()}
        for key in set(shard).intersection(deltas):
            adapter_a, adapter_b = deltas[key]
            if adapter_a is None or adapter_b is None:
                raise ValueError(f"RWKV7 adapter is missing an A/B tensor for {key}.")
            base_weight = shard[key]
            delta = adapter_b.float() @ adapter_a.float()
            shard[key] = (base_weight.float() + scaling * delta).to(base_weight.dtype)
            merged_keys.add(key)
        save_file(
            {key: value.contiguous() for key, value in shard.items()},
            output_path / source.name,
            metadata=shard_metadata,
        )

    if merged_keys != set(deltas):
        raise ValueError(
            "RWKV7 base checkpoint is missing adapter targets: "
            f"{sorted(set(deltas) - merged_keys)}."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("native-to-peft", "peft-to-native"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("input")
        subparser.add_argument("output")
        subparser.add_argument("--base-model", required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("base_model")
    merge_parser.add_argument("adapter")
    merge_parser.add_argument("output")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "native-to-peft":
        native_to_peft(args.input, args.output, base_model_path=args.base_model)
    elif args.command == "peft-to-native":
        peft_to_native(args.input, args.output, base_model_path=args.base_model)
    else:
        merge_native_adapter(args.base_model, args.adapter, args.output)


if __name__ == "__main__":
    main()


__all__ = [
    "adapter_state_dict",
    "base_config_sha256",
    "build_native_adapter_config",
    "load_native_adapter",
    "merge_native_adapter",
    "native_to_peft",
    "NATIVE_ADAPTER_FORMAT",
    "NATIVE_ADAPTER_VERSION",
    "peft_to_native",
    "RWKV7_LORA_TARGETS",
    "save_native_adapter",
    "validate_native_adapter",
]
