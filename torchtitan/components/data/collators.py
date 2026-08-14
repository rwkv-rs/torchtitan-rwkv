# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configured conversion from dataset rows to trainer batches."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import Configurable


TrainerBatch: TypeAlias = tuple[dict[str, Any], torch.Tensor]


class Collator(Configurable, ABC):
    """Configured row-to-batch conversion."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(self, rows: Sequence[Any]) -> TrainerBatch:
        ...

    def num_rows_per_batch(self) -> int:
        """Return the number of dataset rows consumed by one trainer batch."""
        return 1


class TextCollator(Collator):
    """Concatenates text rows and pads only the final token-batch tail."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config
        self._num_tokens_per_batch = context.num_tokens_per_batch
        self._max_context_length = context.max_context_length
        self._max_num_documents = context.max_num_documents

    def __call__(self, rows: Sequence[TextSequence]) -> TrainerBatch:
        num_tokens = sum(len(row.input_ids) for row in rows)
        if num_tokens > self._num_tokens_per_batch:
            raise ValueError("text rows exceed the configured token batch")

        input_ids = torch.cat([torch.as_tensor(row.input_ids) for row in rows])
        labels = torch.cat([torch.as_tensor(row.labels) for row in rows])
        positions = torch.cat(
            [
                torch.arange(len(row.input_ids))
                if row.positions is None
                else torch.as_tensor(row.positions)
                for row in rows
            ]
        )

        pad_len = self._num_tokens_per_batch - num_tokens
        if pad_len:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len))
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=IGNORE_INDEX)
            positions = torch.cat(
                [positions, torch.zeros(pad_len, dtype=positions.dtype)]
            )

        if self._max_num_documents is not None:
            input_ids, labels, positions = _cap_document_count(
                input_ids,
                labels,
                positions,
                max_num_documents=self._max_num_documents,
                max_context_length=self._max_context_length,
            )

        return {
            "input": input_ids,
            "positions": positions,
        }, labels


def _cap_document_count(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    *,
    max_num_documents: int,
    max_context_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace excess document suffixes with context-sized padding segments."""
    document_starts = (positions == 0).nonzero(as_tuple=True)[0]
    if document_starts.numel() <= max_num_documents:
        return input_ids, labels, positions

    num_tokens = positions.numel()
    cutoff = 0
    for num_kept_documents in range(max_num_documents, -1, -1):
        candidate_cutoff = (
            num_tokens
            if num_kept_documents == document_starts.numel()
            else int(document_starts[num_kept_documents])
        )
        num_padding_documents = (
            num_tokens - candidate_cutoff + max_context_length - 1
        ) // max_context_length
        if num_kept_documents + num_padding_documents <= max_num_documents:
            cutoff = candidate_cutoff
            break

    input_ids = input_ids.clone()
    labels = labels.clone()
    positions = positions.clone()
    input_ids[cutoff:] = 0
    labels[cutoff:] = IGNORE_INDEX
    positions[cutoff:] = (
        torch.arange(
            num_tokens - cutoff, dtype=positions.dtype, device=positions.device
        )
        % max_context_length
    )
    return input_ids, labels, positions
