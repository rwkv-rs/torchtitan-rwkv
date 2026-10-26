# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare RWKV training dispatches during the first forward/backward step."""

import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import torch
import tyro

from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer

from .provider import preload_flash_rwkv2


class RWKVTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        tuning_mode: Annotated[Literal["tune", "require"], tyro.conf.Suppress] = "tune"
        """Tune missing dispatches, or require existing verified profiles."""

        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            if self.tuning_mode not in ("tune", "require"):
                raise ValueError("RWKV tuning_mode must be 'tune' or 'require'.")

    def __init__(self, config: Config):
        super().__init__(config)
        self._tuning_mode = config.tuning_mode
        self._tuning_report: dict[str, Any] | None = None
        self._fwd_bwd_fn = self.fwd_bwd_fn
        self.fwd_bwd_fn = self._prepare_forward_backward

    def _prepare_forward_backward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        if self._tuning_report is not None:
            return self._fwd_bwd_fn(inputs, labels, global_valid_tokens, extra_kwargs)

        provider = preload_flash_rwkv2(("prepare_tuning",), "training preparation")
        losses: list[torch.Tensor] = []
        # Preserve the first step's loss and gradients while preparing eager dispatches.
        with torch.compiler.set_stance("force_eager"):
            self._tuning_report = provider.prepare_tuning(
                [
                    lambda: losses.append(
                        super(RWKVTrainer, self)._forward_backward_body(
                            inputs, labels, global_valid_tokens, extra_kwargs
                        )
                    )
                ],
                mode=self._tuning_mode,
            )
        logger.info(
            "FlashRWKV2 training tuning: %s",
            json.dumps(self._tuning_report, sort_keys=True),
        )
        return losses[0]
