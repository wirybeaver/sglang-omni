# SPDX-License-Identifier: Apache-2.0
"""Pipeline registration and backbone configuration contracts."""

from unittest.mock import patch

import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.ming3_asr import stages
from sglang_omni.models.ming3_asr.config import (
    Ming3ASRFactoryArgs,
    Ming3ASRPipelineConfig,
)
from sglang_omni.models.ming3_asr.stages import Ming3ASRConfig, create_thinker_executor
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

CHECKPOINT_ARCHITECTURE = "MingV3ASRForConditionalGeneration"
HIDDEN_SIZE = 64
NATIVE_CLIP_SECONDS = 30.0


def test_checkpoint_architecture_resolves_to_ming3_asr_pipeline() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config(CHECKPOINT_ARCHITECTURE)
        is Ming3ASRPipelineConfig
    )
    capabilities = get_model_capabilities(CHECKPOINT_ARCHITECTURE)
    assert capabilities is not None
    assert capabilities.supports_reference_audio is False


def test_default_pipeline_ends_at_thinker() -> None:
    config = Ming3ASRPipelineConfig(model_path="inclusionAI/Ming-3.0-asr-tiny")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "thinker",
    ]
    assert config.terminal_stages == ["thinker"]
    thinker = config.stages[1]
    assert isinstance(thinker.factory, Ming3ASRFactoryArgs)
    assert thinker.factory.pre_lm_max_batch_size == 8
    assert thinker.factory.pre_lm_max_batch_wait_ms == 2
    factory_arguments = resolve_stage_factory_args(thinker, config)
    assert factory_arguments["enable_async_decode"] is True


def two_stage_document(
    *,
    pre_lm_max_batch_size: int = 4,
    pre_lm_max_batch_wait_ms: int = 0,
) -> dict[str, object]:
    return {
        "model_path": "inclusionAI/Ming-3.0-asr-tiny",
        "stages": [
            {
                "name": "preprocessing",
                "process": "preprocessing",
                "factory_path": (
                    "sglang_omni.models.ming3_asr.stages.create_preprocessing_executor"
                ),
                "next": "thinker",
            },
            {
                "name": "thinker",
                "process": "thinker",
                "gpu": 0,
                "terminal": True,
                "factory_path": (
                    "sglang_omni.models.ming3_asr.stages.create_thinker_executor"
                ),
                "factory": {
                    "pre_lm_max_batch_size": pre_lm_max_batch_size,
                    "pre_lm_max_batch_wait_ms": pre_lm_max_batch_wait_ms,
                },
            },
        ],
    }


def test_two_stage_configuration_resolves_thinker_overrides() -> None:
    config = Ming3ASRPipelineConfig.model_validate(two_stage_document())
    config_manager = ConfigManager(config)
    config = config_manager.merge_config(
        config_manager.parse_extra_args(
            [
                "--thinker.factory.enable_async_decode",
                "false",
                "--thinker.factory.async_decode_min_batch_size",
                "2",
            ]
        )
    )
    assert [stage.name for stage in config.stages] == ["preprocessing", "thinker"]
    thinker = config.stages[1]
    assert thinker.factory.pre_lm_max_batch_size == 4
    assert thinker.factory.pre_lm_max_batch_wait_ms == 0
    factory_arguments = resolve_stage_factory_args(thinker, config)
    assert factory_arguments["enable_async_decode"] is False
    assert factory_arguments["async_decode_min_batch_size"] == 2


@pytest.mark.parametrize(
    ("batch_size", "wait_ms", "match"),
    [
        (0, 2, "pre_lm_max_batch_size"),
        (8, -1, "pre_lm_max_batch_wait_ms"),
    ],
)
def test_configuration_rejects_invalid_pre_lm_knobs(
    batch_size: int,
    wait_ms: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        Ming3ASRPipelineConfig.model_validate(
            two_stage_document(
                pre_lm_max_batch_size=batch_size,
                pre_lm_max_batch_wait_ms=wait_ms,
            )
        )


@pytest.mark.parametrize(
    ("batch_size", "wait_ms", "match"),
    [
        (0, 2, "pre_lm_max_batch_size"),
        (8, -1, "pre_lm_max_batch_wait_ms"),
    ],
)
def test_thinker_factory_rejects_invalid_pre_lm_knobs_before_load(
    batch_size: int,
    wait_ms: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        create_thinker_executor(
            "missing-checkpoint",
            max_seq_len=8,
            max_new_tokens=4,
            enable_async_decode=False,
            async_decode_min_batch_size=1,
            stream_emit_interval_seconds=0.05,
            pre_lm_max_batch_size=batch_size,
            pre_lm_max_batch_wait_ms=wait_ms,
            request_build_max_workers=1,
            request_build_max_pending=1,
        )


def test_thinker_factory_rejects_tensor_parallelism_before_load() -> None:
    with pytest.raises(ValueError, match="tp_size"):
        create_thinker_executor(
            "missing-checkpoint",
            max_seq_len=8,
            max_new_tokens=4,
            enable_async_decode=False,
            async_decode_min_batch_size=1,
            stream_emit_interval_seconds=0.05,
            pre_lm_max_batch_size=8,
            pre_lm_max_batch_wait_ms=2,
            request_build_max_workers=1,
            request_build_max_pending=1,
            tp_size=2,
        )


def test_long_uploads_split_into_native_clips() -> None:
    config = Ming3ASRPipelineConfig(model_path="inclusionAI/Ming-3.0-asr-tiny")
    chunking = config.resolved_audio_chunking
    assert chunking.allow_audio_chunking is True
    assert chunking.max_native_clip_s == NATIVE_CLIP_SECONDS


def test_backbone_config_flattens_text_config() -> None:
    config = Ming3ASRConfig(
        text_config={
            "hidden_size": HIDDEN_SIZE,
            "model_type": "bailing_moe_v3",
        },
        audio_token_id=7,
    )
    assert config.model_type == "ming_v3_asr"
    assert config.hidden_size == HIDDEN_SIZE


@pytest.mark.parametrize(
    ("arguments", "disabled"),
    [([], False), (["--thinker.engine.disable_router_gate_matvec", "true"], True)],
)
def test_router_engine_option_is_consumed_before_native_args(
    arguments: list[str], disabled: bool
) -> None:
    config = Ming3ASRPipelineConfig(model_path="missing-checkpoint")
    manager = ConfigManager(config)
    config = manager.merge_config(manager.parse_extra_args(arguments))
    factory_arguments = resolve_stage_factory_args(config.stages[1], config)
    configuration = {
        "audio_config": {
            "n_mels": 4,
            "n_audio_ctx": 8,
            "n_audio_state": 4,
            "n_audio_head": 1,
            "n_audio_layer": 1,
            "downsampler_kernel_size_list": [3],
            "downsampler_stride_list": [2],
        },
        "text_config": {"hidden_size": 4},
    }
    with (
        patch.object(
            stages, "resolve_concrete_device", return_value=torch.device("cuda:0")
        ),
        patch.object(stages, "patch_kda_diagonal_anchor"),
        patch.object(stages, "patch_bailing_router_gate") as router_patch,
        patch.object(stages, "load_checkpoint_config", return_value=configuration),
        patch.object(stages, "Ming3AudioEncoder"),
        patch.object(stages.AutoConfig, "register"),
        patch.dict(stages.ModelRegistry.models),
        patch.object(
            stages,
            "build_sglang_server_args",
            side_effect=RuntimeError("stop before native engine startup"),
        ) as builder,
        pytest.raises(RuntimeError, match="stop before native engine startup"),
    ):
        create_thinker_executor(**factory_arguments)
    router_patch.assert_called_once_with(disabled)
    assert "disable_router_gate_matvec" not in builder.call_args.kwargs
