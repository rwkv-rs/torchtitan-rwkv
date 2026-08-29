# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict conversion between TorchTitan and Hugging Face RWKV7."""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any

from safetensors import safe_open
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import RWKV7Model


_LAYER_HF_TO_TT = {
    "input_layernorm.weight": "input_layernorm.weight",
    "input_layernorm.bias": "input_layernorm.bias",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "post_attention_layernorm.bias": "post_attention_layernorm.bias",
    "linear_attn.x_r": "linear_attn.x_r",
    "linear_attn.x_w": "linear_attn.x_w",
    "linear_attn.x_k": "linear_attn.x_k",
    "linear_attn.x_v": "linear_attn.x_v",
    "linear_attn.x_a": "linear_attn.x_a",
    "linear_attn.x_g": "linear_attn.x_g",
    "linear_attn.w0": "linear_attn.w0",
    "linear_attn.w1": "linear_attn.w1",
    "linear_attn.w2": "linear_attn.w2",
    "linear_attn.a0": "linear_attn.a0",
    "linear_attn.a1": "linear_attn.a1",
    "linear_attn.a2": "linear_attn.a2",
    "linear_attn.v0": "linear_attn.v0",
    "linear_attn.v1": "linear_attn.v1",
    "linear_attn.v2": "linear_attn.v2",
    "linear_attn.g1": "linear_attn.g1",
    "linear_attn.g2": "linear_attn.g2",
    "linear_attn.k_k": "linear_attn.k_k",
    "linear_attn.k_a": "linear_attn.k_a",
    "linear_attn.r_k": "linear_attn.r_k",
    "linear_attn.r_proj.weight": "linear_attn.r_proj.weight",
    "linear_attn.k_proj.weight": "linear_attn.k_proj.weight",
    "linear_attn.v_proj.weight": "linear_attn.v_proj.weight",
    "linear_attn.o_proj.weight": "linear_attn.o_proj.weight",
    "linear_attn.g_norm.weight": "linear_attn.g_norm.weight",
    "linear_attn.g_norm.bias": "linear_attn.g_norm.bias",
    "mlp.x_k": "mlp.x_k",
    "mlp.key.weight": "mlp.key.weight",
    "mlp.value.weight": "mlp.value.weight",
}

_TRAINING_ARTIFACT_KEY_PATTERN = re.compile(
    r"^layers\.\d+\."
    r"(?:(?:linear_attn\.(?:initial_shift|initial_wkv_state))|"
    r"(?:mlp\.initial_shift)|"
    r"(?:linear_attn\.(?:r_proj|k_proj|v_proj|o_proj)\.lora_[ab]\.weight))$"
)


class RWKV7StateDictAdapter(StateDictAdapter):
    """Map the complete RWKV7 base-model state without layout transforms."""

    def __init__(
        self,
        model_config: RWKV7Model.Config,
        hf_assets_path: str | None,
    ) -> None:
        super().__init__(model_config, hf_assets_path)
        self.rwkv_config = model_config

    def _expected_mapping(self) -> dict[str, str]:
        mapping = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.embedding_norm.weight": "embedding_norm.weight",
            "model.embedding_norm.bias": "embedding_norm.bias",
            "model.norm.weight": "norm.weight",
            "model.norm.bias": "norm.bias",
            "lm_head.weight": "lm_head.weight",
        }
        for layer_id in range(self.rwkv_config.num_layers):
            for hf_suffix, tt_suffix in _LAYER_HF_TO_TT.items():
                if layer_id == 0 and hf_suffix in {
                    "linear_attn.v0",
                    "linear_attn.v1",
                    "linear_attn.v2",
                }:
                    continue
                mapping[
                    f"model.layers.{layer_id}.{hf_suffix}"
                ] = f"layers.{layer_id}.{tt_suffix}"
        return mapping

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        tt_to_hf = {
            tt_key: hf_key for hf_key, tt_key in self._expected_mapping().items()
        }
        hf_state_dict = {}
        for tt_key, value in state_dict.items():
            hf_key = tt_to_hf.get(tt_key)
            if hf_key is not None:
                hf_state_dict[hf_key] = value
            elif _TRAINING_ARTIFACT_KEY_PATTERN.fullmatch(tt_key) is None:
                raise ValueError(f"Unexpected RWKV7 TorchTitan state key {tt_key!r}.")
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        mapping = self._expected_mapping()
        expected = set(mapping)
        actual = set(hf_state_dict)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "RWKV7 HF state keys do not match the model contract: "
                f"missing={missing}, unexpected={unexpected}."
            )
        return {mapping[hf_key]: value for hf_key, value in hf_state_dict.items()}

    def _expected_hf_shapes(self) -> dict[str, tuple[int, ...]]:
        dim = self.rwkv_config.dim
        hidden_dim = self.rwkv_config.hidden_dim
        num_heads = dim // self.rwkv_config.head_size
        shapes: dict[str, tuple[int, ...]] = {
            "model.embed_tokens.weight": (self.rwkv_config.vocab_size, dim),
            "model.embedding_norm.weight": (dim,),
            "model.embedding_norm.bias": (dim,),
            "model.norm.weight": (dim,),
            "model.norm.bias": (dim,),
            "lm_head.weight": (self.rwkv_config.vocab_size, dim),
        }
        for layer_id in range(self.rwkv_config.num_layers):
            prefix = f"model.layers.{layer_id}."
            layer_shapes: dict[str, tuple[int, ...]] = {
                "input_layernorm.weight": (dim,),
                "input_layernorm.bias": (dim,),
                "post_attention_layernorm.weight": (dim,),
                "post_attention_layernorm.bias": (dim,),
                "linear_attn.x_r": (dim,),
                "linear_attn.x_w": (dim,),
                "linear_attn.x_k": (dim,),
                "linear_attn.x_v": (dim,),
                "linear_attn.x_a": (dim,),
                "linear_attn.x_g": (dim,),
                "linear_attn.w0": (dim,),
                "linear_attn.w1": (dim, self.rwkv_config.decay_low_rank_dim),
                "linear_attn.w2": (self.rwkv_config.decay_low_rank_dim, dim),
                "linear_attn.a0": (dim,),
                "linear_attn.a1": (dim, self.rwkv_config.a_low_rank_dim),
                "linear_attn.a2": (self.rwkv_config.a_low_rank_dim, dim),
                "linear_attn.g1": (dim, self.rwkv_config.gate_low_rank_dim),
                "linear_attn.g2": (self.rwkv_config.gate_low_rank_dim, dim),
                "linear_attn.k_k": (dim,),
                "linear_attn.k_a": (dim,),
                "linear_attn.r_k": (num_heads, self.rwkv_config.head_size),
                "linear_attn.r_proj.weight": (dim, dim),
                "linear_attn.k_proj.weight": (dim, dim),
                "linear_attn.v_proj.weight": (dim, dim),
                "linear_attn.o_proj.weight": (dim, dim),
                "linear_attn.g_norm.weight": (dim,),
                "linear_attn.g_norm.bias": (dim,),
                "mlp.x_k": (dim,),
                "mlp.key.weight": (hidden_dim, dim),
                "mlp.value.weight": (dim, hidden_dim),
            }
            if layer_id != 0:
                layer_shapes.update(
                    {
                        "linear_attn.v0": (dim,),
                        "linear_attn.v1": (dim, self.rwkv_config.v_low_rank_dim),
                        "linear_attn.v2": (self.rwkv_config.v_low_rank_dim, dim),
                    }
                )
            shapes.update({prefix + key: shape for key, shape in layer_shapes.items()})
        return shapes

    def _validate_hf_config(self, path: str) -> None:
        config_path = os.path.join(path, "config.json")
        try:
            with open(config_path) as config_file:
                hf_config = json.load(config_file)
        except FileNotFoundError as error:
            raise ValueError(
                f"RWKV7 HF checkpoint is missing {config_path}."
            ) from error

        expected = {
            "architecture_version": self.rwkv_config.architecture_version,
            "vocab_size": self.rwkv_config.vocab_size,
            "hidden_size": self.rwkv_config.dim,
            "num_hidden_layers": self.rwkv_config.num_layers,
            "intermediate_size": self.rwkv_config.hidden_dim,
            "head_size": self.rwkv_config.head_size,
            "num_attention_heads": self.rwkv_config.dim // self.rwkv_config.head_size,
            "layer_norm_epsilon": self.rwkv_config.layer_norm_epsilon,
            "group_norm_epsilon": self.rwkv_config.group_norm_epsilon,
            "decay_low_rank_dim": self.rwkv_config.decay_low_rank_dim,
            "a_low_rank_dim": self.rwkv_config.a_low_rank_dim,
            "v_low_rank_dim": self.rwkv_config.v_low_rank_dim,
            "gate_low_rank_dim": self.rwkv_config.gate_low_rank_dim,
            "tie_word_embeddings": False,
        }
        mismatches = {
            key: {"expected": value, "actual": hf_config.get(key)}
            for key, value in expected.items()
            if hf_config.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"RWKV7 HF config does not match the selected flavor: {mismatches}."
            )

    def _validate_hf_tensors(self, path: str) -> None:
        expected = self._expected_hf_shapes()
        actual: dict[str, tuple[int, ...]] = {}
        safetensor_paths = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not safetensor_paths:
            raise ValueError(f"RWKV7 HF checkpoint has no safetensors files in {path}.")
        for safetensor_path in safetensor_paths:
            with safe_open(safetensor_path, framework="pt", device="cpu") as tensors:
                for key in tensors.keys():
                    actual[key] = tuple(tensors.get_slice(key).get_shape())

        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            raise ValueError(
                "RWKV7 HF tensor keys do not match the model contract: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}."
            )
        bad_shapes = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if actual[key] != expected[key]
        }
        if bad_shapes:
            raise ValueError(f"RWKV7 HF tensor shapes do not match: {bad_shapes}.")

    def get_hf_storage_reader(
        self,
        path: str,
        from_quantized: bool = False,
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            raise ValueError("RWKV7 does not support quantized HF checkpoint loading.")
        self._validate_hf_config(path)
        self._validate_hf_tensors(path)
        return HuggingFaceStorageReader(path)


__all__ = ["RWKV7StateDictAdapter"]
