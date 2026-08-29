# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torchtitan.components.checkpointer.base import MODEL

from torchtitan.components.data import (
    FixedRowTextCollator,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.components.lora import LoRAConverter
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import ParallelismConfig
from torchtitan.experiments.rwkv_state_tuning import (
    parallelize as state_parallelize,
    RWKVStateTuningAttention,
    RWKVStateTuningConverter,
    RWKVStateTuningFeedForward,
    RWKVStateTuningModel,
)
from torchtitan.experiments.rwkv_state_tuning.config_registry import (
    state_tuning_debugmodel,
)
from torchtitan.hf_datasets.text_datasets import ChatProcessor
from torchtitan.models.rwkv7 import (
    model as rwkv7_model,
    model_registry,
    rwkv_configs,
    RWKVAttention,
    RWKVDecoderLayer,
    RWKVFeedForward,
    RWKVModel,
)
from torchtitan.models.rwkv7.adapter import adapter_state_dict, RWKV_LORA_TARGETS
from torchtitan.models.rwkv7.checkpoint import RWKVCheckpointManager
from torchtitan.models.rwkv7.provider import (
    load_flash_rwkv2,
    PRETRAIN_ATTENTION_OPERATORS,
    PRETRAIN_FEED_FORWARD_OPERATORS,
    STATE_TUNING_ATTENTION_OPERATORS,
    STATE_TUNING_FEED_FORWARD_OPERATORS,
)
from torchtitan.models.rwkv7.state_dict_adapter import RWKVStateDictAdapter
from torchtitan.tools.utils import set_default_dtype


FLAVORS = {
    "debugmodel": (2, 128, 512, 32, 32, 32, 17_233_024),
    "0.1b": (12, 768, 3072, 64, 32, 128, 191_034_624),
    "0.4b": (24, 1024, 4096, 64, 32, 128, 450_767_872),
    "1.5b": (24, 2048, 8192, 96, 64, 256, 1_527_404_544),
    "2.9b": (32, 2560, 10240, 96, 64, 320, 2_947_735_040),
    "7.2b": (32, 4096, 16384, 128, 96, 480, 7_199_141_888),
    "13.3b": (61, 4096, 16384, 192, 128, 384, 13_269_245_952),
}


@pytest.mark.parametrize("flavor", FLAVORS)
def test_rwkv7_flavor_contract(flavor):
    (
        num_layers,
        dim,
        hidden_dim,
        decay_rank,
        value_rank,
        gate_rank,
        num_params,
    ) = FLAVORS[flavor]
    config = rwkv_configs[flavor]()
    assert config.architecture_version == "rwkv7"
    assert config.vocab_size == 65536
    assert config.num_layers == num_layers
    assert config.dim == dim
    assert config.hidden_dim == hidden_dim
    assert config.decay_low_rank_dim == decay_rank
    assert config.a_low_rank_dim == decay_rank
    assert config.v_low_rank_dim == value_rank
    assert config.gate_low_rank_dim == gate_rank
    assert config.head_size == 64
    assert config.layer_norm_epsilon == 1e-5
    assert config.group_norm_epsilon == 64e-5

    with torch.device("meta"):
        model = config.build()
    assert isinstance(model, RWKVModel)
    assert all(isinstance(layer, RWKVDecoderLayer) for layer in model.layers.values())
    assert all(
        isinstance(layer.linear_attn, RWKVAttention)
        and isinstance(layer.mlp, RWKVFeedForward)
        for layer in model.layers.values()
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == num_params


def test_rwkv7_initialization_matches_reference_formulas():
    model = model_registry("debugmodel").model.build()
    model.init_states()
    layer_0 = model.layers["0"]
    layer_1 = model.layers["1"]
    assert layer_0.linear_attn.w0[0].item() == pytest.approx(-8.0)
    assert layer_0.linear_attn.r_k.unique().item() == pytest.approx(-0.04)
    assert torch.count_nonzero(layer_0.linear_attn.w1) == 0
    assert torch.count_nonzero(layer_0.linear_attn.o_proj.weight) == 0
    assert layer_0.linear_attn.g_norm.weight[0].item() == pytest.approx((1 / 2) ** 0.7)
    assert layer_1.linear_attn.g_norm.weight[0].item() == pytest.approx(1.0)
    assert torch.count_nonzero(layer_0.mlp.value.weight) == 0


def test_rwkv7_bfloat16_materialization_initializes_orthogonal_weights():
    with torch.device("meta"), set_default_dtype(torch.bfloat16):
        model = model_registry("debugmodel").model.build()
    model.to_empty(device="cpu")
    model.init_states()
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_rwkv7_runtime_config_boundaries():
    model_config = rwkv_configs["debugmodel"]()
    training = SimpleNamespace(
        dtype="bfloat16",
        max_context_length=127,
        num_tokens_per_microbatch_per_dp_rank=256,
    )
    runtime = SimpleNamespace(
        training=training,
        parallelism=ParallelismConfig(),
    )
    with pytest.raises(ValueError, match="multiple of 16"):
        model_config.update_from_config(config=runtime)

    training.max_context_length = 128
    training.num_tokens_per_microbatch_per_dp_rank = 255
    with pytest.raises(ValueError, match="must be divisible"):
        model_config.update_from_config(config=runtime)

    training.num_tokens_per_microbatch_per_dp_rank = 256
    runtime.parallelism.tensor_parallel_degree = 2
    with pytest.raises(ValueError, match="unsupported parallelism"):
        model_config.update_from_config(config=runtime)


def test_rwkv7_has_no_cpu_provider_fallback():
    model = model_registry("debugmodel").model.build()
    tokens = torch.zeros(1, 16, dtype=torch.long)
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        model(tokens)


@pytest.mark.parametrize(
    ("converters", "expected_operators"),
    [
        (
            None,
            (PRETRAIN_ATTENTION_OPERATORS, PRETRAIN_FEED_FORWARD_OPERATORS),
        ),
        (
            [RWKVStateTuningConverter.Config()],
            (
                STATE_TUNING_ATTENTION_OPERATORS,
                STATE_TUNING_FEED_FORWARD_OPERATORS,
            ),
        ),
    ],
)
def test_rwkv7_preloads_exact_provider_groups_before_compile(
    monkeypatch,
    converters,
    expected_operators,
):
    calls = []
    preloaded = SimpleNamespace()

    def preload(operators, mode):
        calls.append((operators, mode))
        return preloaded

    monkeypatch.setattr(rwkv7_model, "preload_flash_rwkv2", preload)
    model = model_registry("debugmodel", converters=converters).model.build()
    model.preload_provider()

    assert [operators for operators, _ in calls] == [
        expected_operators[0],
        expected_operators[1],
        expected_operators[0],
        expected_operators[1],
    ]
    for layer in model.layers.values():
        assert layer.linear_attn._flash_rwkv2 is preloaded
        assert layer.mlp._flash_rwkv2 is preloaded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_preloaded_provider_supports_fullgraph_compile():
    provider = ModuleType("fake_flash_rwkv2")
    provider.identity = torch.neg

    class ProviderBoundary(torch.nn.Module):
        def forward(self, tensor):
            loaded = load_flash_rwkv2(
                ("identity",),
                tensor,
                "compile test",
                preloaded=provider,
            )
            return loaded.identity(tensor)

    compiled = torch.compile(ProviderBoundary().cuda(), fullgraph=True)
    tensor = torch.ones(16, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(compiled(tensor), -tensor)


def test_rwkv7_state_dict_mapping_covers_every_base_parameter():
    config = rwkv_configs["debugmodel"]()
    with torch.device("meta"):
        model = config.build()
    state_dict = model.state_dict()
    adapter = RWKVStateDictAdapter(config, None)
    hf_state_dict = adapter.to_hf(state_dict)
    round_trip = adapter.from_hf(hf_state_dict)
    assert set(round_trip) == set(state_dict)
    for key in state_dict:
        assert round_trip[key] is state_dict[key]


def test_rwkv7_state_dict_adapter_ignores_state_tuning_parameters():
    config = rwkv_configs["debugmodel"]()
    spec = model_registry(
        "debugmodel",
        converters=[RWKVStateTuningConverter.Config()],
    )
    with torch.device("meta"):
        state_dict = spec.model.build().state_dict()
    adapter = RWKVStateDictAdapter(config, None)
    hf_state_dict = adapter.to_hf(state_dict)

    assert set(adapter.from_hf(hf_state_dict)) == set(config.build().state_dict())


def test_rwkv7_lora_targets_and_initialization():
    spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=8,
                alpha=16.0,
                target_modules=list(RWKV_LORA_TARGETS),
            )
        ],
    )
    model = spec.model.build()
    model.init_states()
    trainable = dict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert len(trainable) == 2 * 4 * 2
    for layer_idx in range(2):
        for target in RWKV_LORA_TARGETS:
            prefix = f"layers.{layer_idx}.linear_attn.{target}"
            adapter_a = trainable[f"{prefix}.lora_a.weight"]
            adapter_b = trainable[f"{prefix}.lora_b.weight"]
            assert adapter_a.shape == (8, 128)
            assert adapter_b.shape == (128, 8)
            assert torch.count_nonzero(adapter_a) > 0
            assert torch.count_nonzero(adapter_b) == 0
    assert all(
        parameter.requires_grad == (name in trainable)
        for name, parameter in model.named_parameters()
    )

    manager = object.__new__(RWKVCheckpointManager)
    manager.states = {MODEL: SimpleNamespace(model=[model])}
    states = adapter_state_dict(model.state_dict())
    assert manager._adapter_alpha(states) == 16.0


def test_rwkv7_state_tuning_trainable_parameter_set():
    spec = model_registry(
        "debugmodel",
        converters=[RWKVStateTuningConverter.Config()],
    )
    model = spec.model.build()
    assert isinstance(model, RWKVStateTuningModel)
    assert all(
        isinstance(layer.linear_attn, RWKVStateTuningAttention)
        and isinstance(layer.mlp, RWKVStateTuningFeedForward)
        for layer in model.layers.values()
    )
    model.init_states()
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert set(trainable) == {
        "layers.0.linear_attn.initial_attention_shift",
        "layers.0.linear_attn._wkv_state.initial_wkv_state",
        "layers.0.mlp.initial_feed_forward_shift",
        "layers.1.linear_attn.initial_attention_shift",
        "layers.1.linear_attn._wkv_state.initial_wkv_state",
        "layers.1.mlp.initial_feed_forward_shift",
    }
    for name, parameter in trainable.items():
        if name.endswith("initial_wkv_state"):
            assert parameter.shape == (2, 64, 64)
            assert parameter.dtype == torch.float32
        else:
            assert parameter.shape == (128,)
            assert parameter.dtype == torch.bfloat16

    state_dict = model.state_dict()
    canonical_wkv_keys = {
        f"layers.{layer_idx}.linear_attn.initial_wkv_state" for layer_idx in range(2)
    }
    assert canonical_wkv_keys <= set(state_dict)
    assert not any("._wkv_state." in key for key in state_dict)

    wkv_key = "layers.0.linear_attn.initial_wkv_state"
    loaded_wkv_state = torch.ones_like(state_dict[wkv_key])
    model.load_state_dict({wkv_key: loaded_wkv_state}, strict=False)
    torch.testing.assert_close(
        model.layers["0"].linear_attn.initial_wkv_state,
        loaded_wkv_state,
    )


@pytest.mark.parametrize(
    "converters",
    [
        [
            LoRAConverter.Config(target_modules=list(RWKV_LORA_TARGETS)),
            RWKVStateTuningConverter.Config(),
        ],
        [
            RWKVStateTuningConverter.Config(),
            LoRAConverter.Config(target_modules=list(RWKV_LORA_TARGETS)),
        ],
    ],
)
def test_rwkv7_state_tuning_rejects_lora(converters):
    with pytest.raises(ValueError, match="cannot be combined"):
        with torch.device("meta"):
            model_registry("debugmodel", converters=converters).model.build()


def test_rwkv7_state_tuning_preserves_fp32_wkv_state_under_fsdp(monkeypatch):
    model_spec = model_registry(
        "debugmodel",
        converters=[RWKVStateTuningConverter.Config()],
    )
    model = model_spec.model.build()
    sharded = []

    monkeypatch.setattr(
        state_parallelize,
        "resolve_fsdp_mesh",
        lambda *args, **kwargs: (object(), None),
    )

    def record_fully_shard(module, **kwargs):
        sharded.append((module, kwargs))
        return module

    monkeypatch.setattr(state_parallelize, "fully_shard", record_fully_shard)
    monkeypatch.setattr(
        state_parallelize,
        "apply_fsdp_to_decoder",
        lambda model, *args, **kwargs: model,
    )
    parallelism = SimpleNamespace(
        spmd_backend="spmd_types",
        fsdp_reshard_after_forward="default",
        enable_fsdp_symm_mem=False,
    )
    result = state_parallelize.parallelize_rwkv_state_tuning(
        model,
        parallel_dims=SimpleNamespace(tp=1, pp=1, cp=1, ep=1),
        training=SimpleNamespace(
            enable_cpu_offload=False,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=parallelism,
        compile_config=SimpleNamespace(enable=False, components=[]),
        ac_config=None,
        dump_folder=".",
    )

    assert result is model
    assert len(sharded) == 2
    for module, kwargs in sharded:
        assert set(module.parameters()) == {module.initial_wkv_state}
        assert module.initial_wkv_state.dtype == torch.float32
        assert kwargs["mp_policy"].param_dtype is None
        assert kwargs["mp_policy"].reduce_dtype == torch.float32


def test_rwkv7_state_tuning_recipe_uses_fp32_aware_parallelism():
    config = state_tuning_debugmodel()
    assert (
        config.model_spec.parallelize_fn
        is state_parallelize.parallelize_rwkv_state_tuning
    )


def test_fixed_row_text_collator_keeps_lanes_independent():
    context = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_id=0),
        max_context_length=4,
        num_tokens_per_batch=8,
    )
    collator = FixedRowTextCollator.Config().build(context=context)
    rows = [
        TextSequence(
            input_ids=np.asarray([1, 2]),
            labels=np.asarray([-100, 3]),
        ),
        TextSequence(
            input_ids=np.asarray([4, 5, 6]),
            labels=np.asarray([-100, 6, 7]),
        ),
    ]
    input_dict, labels = collator(rows)
    assert collator.num_rows_per_batch() == 2
    assert input_dict["input"].reshape(2, 4).tolist() == [
        [1, 2, 0, 0],
        [4, 5, 6, 0],
    ]
    assert labels.reshape(2, 4).tolist() == [
        [-100, 3, -100, -100],
        [-100, 6, 7, -100],
    ]
    assert input_dict["positions"].reshape(2, 4).tolist() == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]


def question_answer_to_messages(sample):
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def _build_fixed_row_dataloader():
    return GrainDataLoader.Config(
        dataset=SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="json",
                split="train",
                load_dataset_kwargs={
                    "data_files": "tests/assets/sft_test/data.json",
                },
            ),
            processor=ChatProcessor.Config(messages_fn=question_answer_to_messages),
            post_filters=(lambda sample: sample is not None,),
        ),
        collator=FixedRowTextCollator.Config(),
        seed=42,
        shuffle=True,
        repeat=True,
        num_prefetch_batches=0,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer"),
        max_context_length=128,
        num_tokens_per_batch=256,
    )


def test_fixed_row_grain_dataloader_resumes_exactly():
    dataloader = _build_fixed_row_dataloader()
    try:
        iterator = iter(dataloader)
        for _ in range(3):
            next(iterator)
        state = dataloader.state_dict()
        expected = [next(iterator) for _ in range(3)]
    finally:
        dataloader.close()

    resumed = _build_fixed_row_dataloader()
    try:
        resumed.load_state_dict(state)
        resumed_iterator = iter(resumed)
        actual = [next(resumed_iterator) for _ in range(3)]
    finally:
        resumed.close()

    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        expected_inputs, expected_labels = expected_batch
        actual_inputs, actual_labels = actual_batch
        assert torch.equal(expected_inputs["input"], actual_inputs["input"])
        assert torch.equal(expected_inputs["positions"], actual_inputs["positions"])
        assert torch.equal(expected_labels, actual_labels)
