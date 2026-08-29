# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RWKV7 checkpoint manager with native LoRA artifact handling."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Annotated, Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint import (
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.checkpointer.base import MODEL
from torchtitan.tools.logging import logger

from .adapter import (
    adapter_state_dict,
    build_native_adapter_config,
    load_native_adapter,
)


def finalize_hf_artifact(artifact_path: str, tensor_filename: str) -> None:
    """Turn DCP's consolidated single shard into a native artifact directory."""
    source_path = os.path.join(artifact_path, "model-00001-of-00001.safetensors")
    if not os.path.isfile(source_path):
        raise RuntimeError(
            "HuggingFaceStorageWriter did not produce its consolidated single "
            f"shard in {artifact_path}."
        )
    os.replace(source_path, os.path.join(artifact_path, tensor_filename))

    index_path = os.path.join(artifact_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        os.remove(index_path)
    sharded_path = os.path.join(artifact_path, "sharded")
    if os.path.isdir(sharded_path):
        shutil.rmtree(sharded_path)


class RWKV7CheckpointManager(CheckpointManager):
    """Preserve ordinary DCP behavior and add final native LoRA artifacts."""

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        initial_load_adapter_path: Annotated[str, tyro.conf.Suppress] = ""
        """Optional native adapter loaded after a fresh HF base load."""

        adapter_folder: Annotated[str, tyro.conf.Suppress] = "adapter"
        """Final adapter artifact folder under the checkpoint root."""

        def __post_init__(self) -> None:
            CheckpointManager.Config.__post_init__(self)
            if not self.adapter_folder:
                raise ValueError("RWKV7 checkpoint adapter_folder must not be empty.")

    def __init__(self, config: Config, **kwargs: Any) -> None:
        self.initial_load_adapter_path = config.initial_load_adapter_path
        self.adapter_folder = config.adapter_folder
        super().__init__(config, **kwargs)

    def _base_model_path(self) -> str:
        if self.initial_load_path:
            return self.initial_load_path
        if self.sd_adapter is not None and self.sd_adapter.hf_assets_path:
            return self.sd_adapter.hf_assets_path
        raise ValueError(
            "RWKV7 adapter handling requires an HF base model path with config.json."
        )

    def _adapter_states(self) -> dict[str, Any]:
        return adapter_state_dict(self.states[MODEL].state_dict())

    def _adapter_alpha(self, states: dict[str, Any]) -> float:
        ranks = {
            tensor.shape[0]
            for key, tensor in states.items()
            if key.endswith(".lora_a.weight")
        }
        scalings = {
            float(module._lora_scaling)
            for model in self.states[MODEL].model
            for module in model.modules()
            if hasattr(module, "_lora_scaling")
        }
        if len(ranks) != 1 or len(scalings) != 1:
            raise ValueError(
                "RWKV7 LoRA artifact requires one consistent rank and scaling, got "
                f"ranks={ranks}, scalings={scalings}."
            )
        return ranks.pop() * scalings.pop()

    def _load_native_adapter(self) -> None:
        expected_states = self._adapter_states()
        if not expected_states:
            raise ValueError(
                "checkpoint.initial_load_adapter_path requires an RWKV7 LoRA model."
            )
        expected_shapes = {
            key: tuple(tensor.shape) for key, tensor in expected_states.items()
        }
        load_native_adapter(
            self.initial_load_adapter_path,
            base_model_path=self._base_model_path(),
            expected_shapes=expected_shapes,
        )
        dcp.load(
            expected_states,
            storage_reader=HuggingFaceStorageReader(
                self.initial_load_adapter_path,
            ),
        )
        self.states[MODEL].load_state_dict(expected_states)

    def _load(self, step: int = -1) -> bool:
        has_checkpoint_folder = self._storage.isdir(self.folder)
        resume_step = -1
        if has_checkpoint_folder:
            resume_step = self._find_load_step() if step == -1 else step
        resuming = resume_step != -1

        loaded = super()._load(step)
        if self.initial_load_adapter_path and not resuming:
            self._load_native_adapter()
            return True
        return loaded

    def _save_adapter_artifact(self, curr_step: int) -> None:
        states = self._adapter_states()
        if not states:
            return
        if not self.initial_load_in_hf:
            logger.warning(
                "Skipping the RWKV7 adapter-only artifact because this run did "
                "not load an HF base model; the full DCP checkpoint remains saved."
            )
            return

        artifact_path = os.path.join(
            self.folder,
            f"{self.adapter_folder}-step-{curr_step}",
        )
        writer = HuggingFaceStorageWriter(
            path=artifact_path,
            save_distributed=True,
            enable_consolidation=True,
        )
        dcp.save(states, storage_writer=writer)
        if dist.is_initialized():
            dist.barrier()

        if not dist.is_initialized() or dist.get_rank() == 0:
            finalize_hf_artifact(artifact_path, "adapter_model.safetensors")
            metadata = build_native_adapter_config(
                states,
                base_model_path=self._base_model_path(),
                alpha=self._adapter_alpha(states),
            )
            with open(os.path.join(artifact_path, "adapter_config.json"), "w") as file:
                json.dump(metadata, file, indent=2, sort_keys=True)
                file.write("\n")

        if dist.is_initialized():
            dist.barrier()

    def _save_last_step(self, curr_step: int) -> None:
        super()._save_last_step(curr_step)
        self._save_adapter_artifact(curr_step)


__all__ = ["finalize_hf_artifact", "RWKV7CheckpointManager"]
