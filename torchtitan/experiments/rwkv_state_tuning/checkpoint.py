# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DCP and release-artifact handling for RWKV7 State Tuning."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Annotated, Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import tyro
from torch.distributed.checkpoint import (
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)

from torchtitan.components.checkpointer.base import MODEL
from torchtitan.models.rwkv7.checkpoint import (
    finalize_hf_artifact,
    RWKV7CheckpointManager,
)
from torchtitan.tools.logging import logger

from .artifact import (
    build_state_artifact_config,
    load_state_artifact,
    state_tuning_state_dict,
)


class RWKV7StateTuningCheckpointManager(RWKV7CheckpointManager):
    """Add native initial-state loading and final state-only artifacts."""

    @dataclass(kw_only=True, slots=True)
    class Config(RWKV7CheckpointManager.Config):
        initial_load_state_path: Annotated[str, tyro.conf.Suppress] = ""
        state_artifact_folder: Annotated[str, tyro.conf.Suppress] = "state"

        def __post_init__(self) -> None:
            RWKV7CheckpointManager.Config.__post_init__(self)
            if not self.state_artifact_folder:
                raise ValueError("state_artifact_folder must not be empty.")

    def __init__(self, config: Config, **kwargs: Any) -> None:
        self.initial_load_state_path = config.initial_load_state_path
        self.state_artifact_folder = config.state_artifact_folder
        super().__init__(config, **kwargs)

    def _state_states(self) -> dict[str, Any]:
        return state_tuning_state_dict(self.states[MODEL].state_dict())

    def _load_state_artifact(self) -> None:
        states = self._state_states()
        if not states:
            raise ValueError(
                "checkpoint.initial_load_state_path requires a State Tuning model."
            )
        expected_shapes = {key: tuple(value.shape) for key, value in states.items()}
        load_state_artifact(
            self.initial_load_state_path,
            base_model_path=self._base_model_path(),
            expected_shapes=expected_shapes,
        )
        dcp.load(
            states,
            storage_reader=HuggingFaceStorageReader(self.initial_load_state_path),
        )
        self.states[MODEL].load_state_dict(states)

    def _load(self, step: int = -1) -> bool:
        has_checkpoint_folder = self._storage.isdir(self.folder)
        resume_step = -1
        if has_checkpoint_folder:
            resume_step = self._find_load_step() if step == -1 else step
        resuming = resume_step != -1

        loaded = super()._load(step)
        if self.initial_load_state_path and not resuming:
            self._load_state_artifact()
            return True
        return loaded

    def _save_state_artifact(self, curr_step: int) -> None:
        states = self._state_states()
        if not states:
            return
        if not self.initial_load_in_hf:
            logger.warning(
                "Skipping the RWKV7 state-only artifact because this run did not "
                "load an HF base model; the full DCP checkpoint remains saved."
            )
            return
        artifact_path = os.path.join(
            self.folder,
            f"{self.state_artifact_folder}-step-{curr_step}",
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
            finalize_hf_artifact(artifact_path, "state_model.safetensors")
            config = build_state_artifact_config(
                states,
                base_model_path=self._base_model_path(),
            )
            with open(os.path.join(artifact_path, "state_config.json"), "w") as file:
                json.dump(config, file, indent=2, sort_keys=True)
                file.write("\n")
        if dist.is_initialized():
            dist.barrier()

    def _save_last_step(self, curr_step: int) -> None:
        super()._save_last_step(curr_step)
        self._save_state_artifact(curr_step)


__all__ = ["RWKV7StateTuningCheckpointManager"]
