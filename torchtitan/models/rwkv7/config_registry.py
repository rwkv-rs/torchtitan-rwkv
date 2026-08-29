# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training recipes for RWKV7 pretraining, full SFT, and LoRA SFT."""

from __future__ import annotations

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
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.trainer import Trainer

from . import model_registry
from .adapter import RWKV7_LORA_TARGETS
from .checkpoint import RWKV7CheckpointManager


def _sft_messages(sample):
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _loss(model_spec) -> ChunkedLossWrapper.Config:
    return ChunkedLossWrapper.Config(
        loss_fn=CrossEntropyLoss.Config(
            global_vocab_size=model_spec.model.vocab_size,
        ),
    )


def _pretrain_config(
    flavor: str,
    *,
    max_context_length: int,
    num_tokens_per_microbatch: int,
    hf_assets_path: str,
) -> Trainer.Config:
    model_spec = model_registry(flavor)
    return Trainer.Config(
        loss=_loss(model_spec),
        hf_assets_path=hf_assets_path,
        model_spec=model_spec,
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch,
            max_context_length=max_context_length,
            steps=1000,
            dtype="bfloat16",
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        metrics=MetricsProcessor.Config(enable_tensorboard=True),
        parallelism=ParallelismConfig(data_parallel_shard_degree=-1),
        checkpoint=RWKV7CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True),
    )


def _sft_config(
    flavor: str,
    *,
    hf_assets_path: str,
    lora: bool,
    load_base: bool,
) -> Trainer.Config:
    converters: list[ModelConfigConverter.Config] | None = None
    if lora:
        converters = [
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                target_modules=list(RWKV7_LORA_TARGETS),
            )
        ]
    model_spec = model_registry(flavor, converters=converters)
    dataset = SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={
                "data_files": "tests/assets/sft_test/data.json",
            },
        ),
        processor=ChatProcessor.Config(messages_fn=_sft_messages),
        post_filters=(lambda sample: sample is not None,),
    )
    return Trainer.Config(
        loss=_loss(model_spec),
        hf_assets_path=hf_assets_path,
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-4 if lora else 3e-4),
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
        checkpoint=RWKV7CheckpointManager.Config(
            interval=5,
            last_save_model_only=False,
            initial_load_model_only=load_base,
            initial_load_in_hf=load_base,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True),
    )


def rwkv7_debugmodel() -> Trainer.Config:
    config = _pretrain_config(
        "debugmodel",
        max_context_length=128,
        num_tokens_per_microbatch=2 * 128,
        hf_assets_path="./tests/assets/tokenizer",
    )
    config.training.steps = 10
    config.lr_scheduler.warmup_steps = 2
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        shuffle=False,
    )
    config.metrics.log_freq = 1
    config.checkpoint.interval = 10
    return config


def rwkv7_debugmodel_sft() -> Trainer.Config:
    return _sft_config(
        "debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
        lora=False,
        load_base=False,
    )


def rwkv7_debugmodel_lora_sft() -> Trainer.Config:
    return _sft_config(
        "debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
        lora=True,
        load_base=False,
    )


def rwkv7_0_1b() -> Trainer.Config:
    return _pretrain_config(
        "0.1b",
        max_context_length=4096,
        num_tokens_per_microbatch=4096,
        hf_assets_path="./assets/hf/rwkv7-0.1b",
    )


def rwkv7_0_4b() -> Trainer.Config:
    return _pretrain_config(
        "0.4b",
        max_context_length=4096,
        num_tokens_per_microbatch=4096,
        hf_assets_path="./assets/hf/rwkv7-0.4b",
    )


def rwkv7_1_5b() -> Trainer.Config:
    return _pretrain_config(
        "1.5b",
        max_context_length=4096,
        num_tokens_per_microbatch=4096,
        hf_assets_path="./assets/hf/rwkv7-1.5b",
    )


def rwkv7_2_9b() -> Trainer.Config:
    return _pretrain_config(
        "2.9b",
        max_context_length=4096,
        num_tokens_per_microbatch=4096,
        hf_assets_path="./assets/hf/rwkv7-2.9b",
    )


def rwkv7_7_2b() -> Trainer.Config:
    return _pretrain_config(
        "7.2b",
        max_context_length=4096,
        num_tokens_per_microbatch=4096,
        hf_assets_path="./assets/hf/rwkv7-7.2b",
    )


def rwkv7_13_3b() -> Trainer.Config:
    return _pretrain_config(
        "13.3b",
        max_context_length=4096,
        num_tokens_per_microbatch=4096,
        hf_assets_path="./assets/hf/rwkv7-13.3b",
    )


def rwkv7_1_5b_sft() -> Trainer.Config:
    return _sft_config(
        "1.5b",
        hf_assets_path="./assets/hf/rwkv7-1.5b",
        lora=False,
        load_base=True,
    )


def rwkv7_1_5b_lora_sft() -> Trainer.Config:
    return _sft_config(
        "1.5b",
        hf_assets_path="./assets/hf/rwkv7-1.5b",
        lora=True,
        load_base=True,
    )
