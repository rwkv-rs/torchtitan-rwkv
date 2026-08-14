# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import TextSequence
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.text_datasets import DATASETS

_TOKENIZER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "tokenizer"
)


def _build_dataloader(max_context_length: int) -> GrainDataLoader:
    return GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        shuffle=False,
        num_prefetch_batches=0,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH),
        max_context_length=max_context_length,
        num_tokens_per_batch=max_context_length,
    )


class TestTextDatasetPacking(unittest.TestCase):
    """Packing must emit the configured token count with in-range positions.

    Inputs and labels are shifted per document at tokenization time, so a
    packed sample contains exactly ``max_context_length`` tokens in this test.
    Emitting one extra token would push the largest position one past the final
    RoPE cache entry and only surface as an asynchronous device-side assertion.
    """

    def test_positions_are_contiguous_per_document_runs(self):
        dataloader = _build_dataloader(256)
        try:
            iterator = iter(dataloader)
            for _ in range(100):
                input_dict, _labels = next(iterator)
                positions = input_dict["positions"]
                steps = positions[1:] - positions[:-1]
                # Each position either continues the current document (+1) or
                # restarts a new one (back to 0).
                self.assertTrue(bool(torch.all((steps == 1) | (positions[1:] == 0))))
        finally:
            dataloader.close()

    def test_no_cross_document_targets(self):
        """The last token of a document must never predict the next document."""
        tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
        dataloader = _build_dataloader(256)
        interior_doc_starts = 0
        try:
            iterator = iter(dataloader)
            for _ in range(100):
                input_dict, labels = next(iterator)
                input_ids = input_dict["input"]
                positions = input_dict["positions"]

                # EOS closes a document and is never fed back in; BOS opens one and
                # is never a target.
                self.assertFalse(bool(torch.any(input_ids == tokenizer.eos_id)))
                self.assertFalse(bool(torch.any(labels == tokenizer.bos_id)))

                starts = (input_ids == tokenizer.bos_id).nonzero().flatten()
                starts = starts[starts > 0]
                interior_doc_starts += len(starts)
                self.assertTrue(bool(torch.all(positions[starts] == 0)))
                # The token right before a document start predicts that document's
                # own EOS, not the next document's first token.
                self.assertTrue(bool(torch.all(labels[starts - 1] == tokenizer.eos_id)))
        finally:
            dataloader.close()

        # Guard against the assertions above passing vacuously.
        self.assertGreater(interior_doc_starts, 0)

    def test_document_cap_pads_with_context_sized_segments(self):
        collator = TextCollator.Config().build(
            context=SimpleNamespace(
                num_tokens_per_batch=12,
                max_context_length=4,
                max_num_documents=4,
            )
        )
        row = TextSequence(
            input_ids=np.arange(12),
            labels=np.arange(12),
            positions=np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        )

        input_dict, labels = collator([row])
        input_ids = input_dict["input"]
        positions = input_dict["positions"]

        document_starts = (positions == 0).nonzero(as_tuple=True)[0]
        self.assertEqual(document_starts.tolist(), [0, 2, 4, 8])
        boundaries = torch.cat([document_starts, torch.tensor([positions.numel()])])
        self.assertLessEqual(int(torch.diff(boundaries).max()), 4)
        torch.testing.assert_close(input_ids[4:], torch.zeros(8, dtype=torch.int64))
        torch.testing.assert_close(
            labels[4:], torch.full((8,), IGNORE_INDEX, dtype=torch.int64)
        )


class TestTextDatasetBufferCheckpointing(unittest.TestCase):
    def test_packing_state_round_trips(self):
        dataloader = _build_dataloader(256)
        try:
            iterator = iter(dataloader)
            for _ in range(5):
                next(iterator)
            state = dataloader.state_dict()
            expected = [next(iterator) for _ in range(5)]
        finally:
            dataloader.close()

        resumed = _build_dataloader(256)
        try:
            resumed.load_state_dict(state)
            resumed_iterator = iter(resumed)
            actual = [next(resumed_iterator) for _ in range(5)]
        finally:
            resumed.close()

        for (expected_inputs, expected_labels), (actual_inputs, actual_labels) in zip(
            expected, actual, strict=True
        ):
            self.assertTrue(
                torch.equal(expected_inputs["input"], actual_inputs["input"])
            )
            self.assertTrue(
                torch.equal(expected_inputs["positions"], actual_inputs["positions"])
            )
            self.assertTrue(torch.equal(expected_labels, actual_labels))


if __name__ == "__main__":
    unittest.main()
