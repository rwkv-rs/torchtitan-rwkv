# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RWKV7 State Tuning recipes."""

from dataclasses import replace
from typing import cast

from torchtitan.components.data import (
    FixedRowTextCollator,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import ChatProcessor
from torchtitan.models.rwkv7 import model_registry, RWKV7Model
from torchtitan.trainer import Trainer

from .checkpoint import RWKV7StateTuningCheckpointManager
from .converter import RWKV7StateTuningConverter
from .parallelize import parallelize_rwkv7_state_tuning


def _messages(sample):
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _state_tuning_config(
    flavor: str,
    *,
    hf_assets_path: str,
    load_base: bool,
) -> Trainer.Config:
    model_spec = model_registry(
        flavor,
        converters=[RWKV7StateTuningConverter.Config()],
    )
    model_spec = replace(
        model_spec,
        parallelize_fn=parallelize_rwkv7_state_tuning,
    )
    model_config = cast(RWKV7Model.Config, model_spec.model)
    dataset = SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={
                "data_files": "tests/assets/sft_test/data.json",
            },
        ),
        processor=ChatProcessor.Config(messages_fn=_messages),
        post_filters=(lambda sample: sample is not None,),
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=model_config.vocab_size,
            ),
        ),
        hf_assets_path=hf_assets_path,
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * 1024,
            max_context_length=1024,
            steps=10,
            dtype="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            dataset=dataset,
            collator=FixedRowTextCollator.Config(),
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=True,
        ),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
        checkpoint=RWKV7StateTuningCheckpointManager.Config(
            interval=5,
            last_save_model_only=False,
            initial_load_model_only=load_base,
            initial_load_in_hf=load_base,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True),
    )


def rwkv7_debugmodel_state_tuning() -> Trainer.Config:
    return _state_tuning_config(
        "debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
        load_base=False,
    )


def rwkv7_1_5b_state_tuning() -> Trainer.Config:
    return _state_tuning_config(
        "1.5b",
        hf_assets_path="./assets/hf/rwkv7-1.5b",
        load_base=True,
    )
