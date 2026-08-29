# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training recipes for RWKV-7 pretraining, full SFT, and LoRA SFT."""

from __future__ import annotations

from typing import cast

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FixedRowTextCollator,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.lora import LoRAConverter
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import ChatProcessor, DATASETS
from torchtitan.trainer import Trainer

from . import model_registry
from .adapter import RWKV_LORA_TARGETS
from .checkpoint import RWKVCheckpointManager
from .model import RWKVModel


def rwkv7_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=cast(RWKVModel.Config, model_spec.model).vocab_size,
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * 128,
            max_context_length=128,
            steps=10,
            dtype="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
        checkpoint=RWKVCheckpointManager.Config(interval=10),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True),
    )


def sft_debugmodel() -> Trainer.Config:
    def process_sample(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=cast(RWKVModel.Config, model_spec.model).vocab_size,
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=default_adamw(lr=3e-4),
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
            dataset=SingleDatasetConfig(
                source=HuggingFaceRandomAccessSource.Config(
                    path="json",
                    split="train",
                    load_dataset_kwargs={
                        "data_files": "tests/assets/sft_test/data.json",
                    },
                ),
                processor=ChatProcessor.Config(messages_fn=process_sample),
                post_filters=(lambda sample: sample is not None,),
            ),
            collator=FixedRowTextCollator.Config(),
        ),
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
        checkpoint=RWKVCheckpointManager.Config(
            interval=5,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True),
    )


def sft_debugmodel_lora() -> Trainer.Config:
    config = sft_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                target_modules=list(RWKV_LORA_TARGETS),
            )
        ],
    )
    config.optimizer = default_adamw(lr=8e-4)
    return config


def rwkv7_0_1b() -> Trainer.Config:
    model_spec = model_registry("0.1b")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=cast(RWKVModel.Config, model_spec.model).vocab_size,
            ),
        ),
        hf_assets_path="./assets/hf/rwkv7-0.1b",
        model_spec=model_spec,
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4096,
            max_context_length=4096,
            steps=1000,
            dtype="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        metrics=MetricsProcessor.Config(enable_tensorboard=True),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
        checkpoint=RWKVCheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True),
    )


def rwkv7_0_4b() -> Trainer.Config:
    config = rwkv7_0_1b()
    config.model_spec = model_registry("0.4b")
    config.hf_assets_path = "./assets/hf/rwkv7-0.4b"
    return config


def rwkv7_1_5b() -> Trainer.Config:
    config = rwkv7_0_1b()
    config.model_spec = model_registry("1.5b")
    config.hf_assets_path = "./assets/hf/rwkv7-1.5b"
    return config


def rwkv7_2_9b() -> Trainer.Config:
    config = rwkv7_0_1b()
    config.model_spec = model_registry("2.9b")
    config.hf_assets_path = "./assets/hf/rwkv7-2.9b"
    return config


def rwkv7_7_2b() -> Trainer.Config:
    config = rwkv7_0_1b()
    config.model_spec = model_registry("7.2b")
    config.hf_assets_path = "./assets/hf/rwkv7-7.2b"
    return config


def rwkv7_13_3b() -> Trainer.Config:
    config = rwkv7_0_1b()
    config.model_spec = model_registry("13.3b")
    config.hf_assets_path = "./assets/hf/rwkv7-13.3b"
    return config


def sft_rwkv_1_5b() -> Trainer.Config:
    config = sft_debugmodel()
    config.model_spec = model_registry("1.5b")
    config.hf_assets_path = "./assets/hf/rwkv7-1.5b"
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.initial_load_in_hf = True
    return config


def sft_rwkv_1_5b_lora() -> Trainer.Config:
    config = sft_rwkv_1_5b()
    config.model_spec = model_registry(
        "1.5b",
        converters=[
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                target_modules=list(RWKV_LORA_TARGETS),
            )
        ],
    )
    config.optimizer = default_adamw(lr=8e-4)
    return config
