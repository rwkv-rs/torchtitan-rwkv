# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint import (
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)

from torchtitan.experiments.rwkv_state_tuning.artifact import (
    load_state_artifact,
    save_state_artifact,
)
from torchtitan.experiments.rwkv_state_tuning.checkpoint import (
    RWKV7StateTuningCheckpointManager,
)
from torchtitan.models.rwkv7.adapter import (
    load_native_adapter,
    merge_native_adapter,
    native_to_peft,
    peft_to_native,
    RWKV7_LORA_TARGETS,
    save_native_adapter,
)
from torchtitan.models.rwkv7.checkpoint import (
    finalize_hf_artifact,
    RWKV7CheckpointManager,
)


def _write_base_config(path, *, num_layers=2, hidden_size=4, head_size=2):
    path.mkdir()
    config = {
        "architecture_version": "rwkv7",
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "head_size": head_size,
    }
    (path / "config.json").write_text(json.dumps(config))


def _adapter_tensors(*, num_layers=2, hidden_size=4, rank=2):
    tensors = {}
    for layer_id in range(num_layers):
        for target_id, target in enumerate(RWKV7_LORA_TARGETS):
            prefix = f"layers.{layer_id}.linear_attn.{target}"
            offset = layer_id * 10 + target_id
            tensors[f"{prefix}.lora_a.weight"] = (
                torch.arange(rank * hidden_size, dtype=torch.bfloat16).reshape(
                    rank, hidden_size
                )
                + offset
            )
            tensors[f"{prefix}.lora_b.weight"] = (
                torch.arange(hidden_size * rank, dtype=torch.bfloat16).reshape(
                    hidden_size, rank
                )
                - offset
            )
    return tensors


def test_native_peft_round_trip(tmp_path):
    base_path = tmp_path / "base"
    native_path = tmp_path / "native"
    peft_path = tmp_path / "peft"
    round_trip_path = tmp_path / "round-trip"
    _write_base_config(base_path)
    tensors = _adapter_tensors()
    save_native_adapter(native_path, tensors, base_model_path=base_path, alpha=4.0)
    native_config = json.loads((native_path / "adapter_config.json").read_text())
    assert native_config["rank"] == 2
    assert native_config["alpha"] == 4.0

    loaded = load_native_adapter(native_path, base_model_path=base_path)
    assert loaded.keys() == tensors.keys()
    for key in tensors:
        torch.testing.assert_close(loaded[key], tensors[key])

    native_to_peft(native_path, peft_path, base_model_path=base_path)
    peft_config = json.loads((peft_path / "adapter_config.json").read_text())
    assert peft_config["r"] == 2
    assert peft_config["lora_alpha"] == 4.0
    peft_to_native(peft_path, round_trip_path, base_model_path=base_path)
    round_trip = load_native_adapter(round_trip_path, base_model_path=base_path)
    assert round_trip.keys() == tensors.keys()
    for key in tensors:
        torch.testing.assert_close(round_trip[key], tensors[key])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "invalid", "metadata mismatch"),
        ("version", 2, "metadata mismatch"),
        ("architecture", "rwkv7a", "metadata mismatch"),
        ("base_config_sha256", "invalid", "metadata mismatch"),
        ("targets", ["r_proj"], "metadata mismatch"),
        ("rank", 0, "rank must be a positive integer"),
        ("dtype", "int8", "unsupported dtype"),
    ],
)
def test_native_adapter_rejects_invalid_metadata(tmp_path, field, value, message):
    base_path = tmp_path / "base"
    native_path = tmp_path / "native"
    _write_base_config(base_path)
    save_native_adapter(
        native_path,
        _adapter_tensors(),
        base_model_path=base_path,
        alpha=4.0,
    )
    config_path = native_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        load_native_adapter(native_path, base_model_path=base_path)


def test_native_adapter_rejects_tensor_dtype_mismatch(tmp_path):
    base_path = tmp_path / "base"
    native_path = tmp_path / "native"
    _write_base_config(base_path)
    save_native_adapter(
        native_path,
        _adapter_tensors(),
        base_model_path=base_path,
        alpha=4.0,
    )
    tensor_path = native_path / "adapter_model.safetensors"
    tensors = load_file(tensor_path)
    first_key = next(iter(tensors))
    tensors[first_key] = tensors[first_key].float()
    save_file(tensors, tensor_path)

    with pytest.raises(ValueError, match="tensor dtype mismatch"):
        load_native_adapter(native_path, base_model_path=base_path)


def test_peft_load_and_save_round_trip(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers.RwkvConfig(), "architecture_version"):
        pytest.skip("requires the authoritative transformers-rwkv fork")

    base_path = tmp_path / "base"
    native_path = tmp_path / "native"
    peft_path = tmp_path / "peft"
    peft_saved_path = tmp_path / "peft-saved"
    round_trip_path = tmp_path / "round-trip"
    _write_base_config(base_path, hidden_size=128, head_size=64)
    tensors = _adapter_tensors(hidden_size=128)
    save_native_adapter(native_path, tensors, base_model_path=base_path, alpha=4.0)
    native_to_peft(native_path, peft_path, base_model_path=base_path)

    config = transformers.RwkvConfig(
        architecture_version="rwkv7",
        vocab_size=256,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        head_size=64,
        num_attention_heads=2,
        decay_low_rank_dim=32,
        a_low_rank_dim=32,
        v_low_rank_dim=32,
        gate_low_rank_dim=32,
        layer_norm_epsilon=1e-5,
        group_norm_epsilon=64e-5,
        tie_word_embeddings=False,
    )
    base_model = transformers.RwkvForCausalLM(config).to(torch.bfloat16)
    peft_model = peft.PeftModel.from_pretrained(
        base_model,
        peft_path,
        is_trainable=True,
        autocast_adapter_dtype=False,
    )
    peft_model.save_pretrained(
        peft_saved_path,
        safe_serialization=True,
    )
    peft_to_native(
        peft_saved_path,
        round_trip_path,
        base_model_path=base_path,
    )
    round_trip = load_native_adapter(
        round_trip_path,
        base_model_path=base_path,
    )
    assert round_trip.keys() == tensors.keys()
    for key in tensors:
        torch.testing.assert_close(round_trip[key], tensors[key])


def test_merge_native_adapter_streams_base_shards(tmp_path):
    base_path = tmp_path / "base"
    native_path = tmp_path / "native"
    merged_path = tmp_path / "merged"
    _write_base_config(base_path)
    adapter = _adapter_tensors()
    save_native_adapter(native_path, adapter, base_model_path=base_path, alpha=4.0)

    base_tensors = {"model.embed_tokens.weight": torch.ones(2, 4)}
    for layer_id in range(2):
        for target in RWKV7_LORA_TARGETS:
            key = f"model.layers.{layer_id}.linear_attn.{target}.weight"
            base_tensors[key] = torch.full((4, 4), float(layer_id + 1))
    save_file(base_tensors, base_path / "model.safetensors")

    merge_native_adapter(base_path, native_path, merged_path)
    merged = load_file(merged_path / "model.safetensors")
    assert torch.equal(
        merged["model.embed_tokens.weight"],
        base_tensors["model.embed_tokens.weight"],
    )
    for layer_id in range(2):
        for target in RWKV7_LORA_TARGETS:
            base_key = f"model.layers.{layer_id}.linear_attn.{target}.weight"
            native_prefix = f"layers.{layer_id}.linear_attn.{target}"
            expected = base_tensors[base_key] + 2.0 * (
                adapter[f"{native_prefix}.lora_b.weight"].float()
                @ adapter[f"{native_prefix}.lora_a.weight"].float()
            )
            torch.testing.assert_close(merged[base_key], expected)


def test_state_tuning_artifact_round_trip(tmp_path):
    base_path = tmp_path / "base"
    artifact_path = tmp_path / "state"
    _write_base_config(base_path, num_layers=2, hidden_size=128, head_size=64)
    states = {}
    for layer_id in range(2):
        states[f"layers.{layer_id}.linear_attn.initial_shift"] = torch.zeros(
            128, dtype=torch.bfloat16
        )
        states[f"layers.{layer_id}.linear_attn.initial_wkv_state"] = torch.zeros(
            2, 64, 64, dtype=torch.float32
        )
        states[f"layers.{layer_id}.mlp.initial_shift"] = torch.zeros(
            128, dtype=torch.bfloat16
        )
    save_state_artifact(artifact_path, states, base_model_path=base_path)
    loaded = load_state_artifact(artifact_path, base_model_path=base_path)
    assert loaded.keys() == states.keys()
    for key in states:
        torch.testing.assert_close(loaded[key], states[key])


def test_state_tuning_artifact_rejects_invalid_metadata(tmp_path):
    base_path = tmp_path / "base"
    artifact_path = tmp_path / "state"
    _write_base_config(base_path, num_layers=2, hidden_size=128, head_size=64)
    states = {}
    for layer_id in range(2):
        states[f"layers.{layer_id}.linear_attn.initial_shift"] = torch.zeros(
            128, dtype=torch.bfloat16
        )
        states[f"layers.{layer_id}.linear_attn.initial_wkv_state"] = torch.zeros(
            2, 64, 64, dtype=torch.float32
        )
        states[f"layers.{layer_id}.mlp.initial_shift"] = torch.zeros(
            128, dtype=torch.bfloat16
        )
    save_state_artifact(artifact_path, states, base_model_path=base_path)
    config_path = artifact_path / "state_config.json"
    config = json.loads(config_path.read_text())
    config["version"] = 2
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_state_artifact(artifact_path, base_model_path=base_path)


def test_dcp_hf_artifact_finalization_and_reload(tmp_path):
    artifact_path = tmp_path / "artifact"
    expected = {"layers.0.adapter": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)}
    writer = HuggingFaceStorageWriter(
        path=str(artifact_path),
        save_distributed=True,
        enable_consolidation=True,
    )
    dcp.save(expected, storage_writer=writer)
    finalize_hf_artifact(str(artifact_path), "adapter_model.safetensors")

    assert sorted(path.name for path in artifact_path.iterdir()) == [
        "adapter_model.safetensors"
    ]
    actual = {"layers.0.adapter": torch.empty_like(expected["layers.0.adapter"])}
    dcp.load(actual, storage_reader=HuggingFaceStorageReader(str(artifact_path)))
    torch.testing.assert_close(actual["layers.0.adapter"], expected["layers.0.adapter"])


def test_from_scratch_parameter_efficient_runs_keep_dcp_without_artifact(caplog):
    adapter_manager = SimpleNamespace(
        initial_load_in_hf=False,
        _adapter_states=lambda: {"adapter": torch.ones(1)},
    )
    RWKV7CheckpointManager._save_adapter_artifact(adapter_manager, 10)

    state_manager = SimpleNamespace(
        initial_load_in_hf=False,
        _state_states=lambda: {"state": torch.ones(1)},
    )
    RWKV7StateTuningCheckpointManager._save_state_artifact(state_manager, 10)

    assert "adapter-only artifact" in caplog.text
    assert "state-only artifact" in caplog.text
