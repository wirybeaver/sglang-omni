# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import sglang_omni.model_runner.base as model_runner_base
import sglang_omni.models.whisper_asr.engine_builder as whisper_engine_builder
import sglang_omni.models.whisper_asr.stages as whisper_asr_stages
import sglang_omni.scheduling.bootstrap as bootstrap
import sglang_omni.scheduling.omni_scheduler as omni_scheduler
import sglang_omni.scheduling.sglang_backend as sglang_backend
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_static_factory_args
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.models.whisper_asr import request_builders as whisper_request_builders
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig
from tests.unit_test.fakes import FakeServerArgs


def test_whisper_encoder_cuda_graph_is_opt_in() -> None:
    signature = inspect.signature(whisper_asr_stages.create_sglang_whisper_asr_executor)

    assert signature.parameters["enable_encoder_cuda_graph"].default is False
    assert signature.parameters["enable_torch_compile"].default is True
    assert signature.parameters["encoder_graph_batch_buckets"].default is None
    assert signature.parameters["request_build_max_workers"].default == 2
    assert signature.parameters["request_build_max_pending"].default == 16
    assert signature.parameters["prefill_coalesce_requests"].default == 2
    assert signature.parameters["prefill_coalesce_wait_ms"].default == 6.0
    assert signature.parameters["prefill_coalesce_when_idle"].default is True
    assert (
        signature.parameters["prefill_coalesce_requires_pending_builds"].default is True
    )
    assert (
        signature.parameters["prefill_coalesce_after_builds_during_decode"].default
        is False
    )


def test_whisper_encoder_cuda_graph_setup_is_ordered_after_generation_graphs() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    calls: list[tuple[list[int], int]] = []
    builder = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
        enable_encoder_cuda_graph=True,
    )
    builder.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(nb_max_frames=3000)
    )
    builder.encoder_token_count = 1500
    assert builder.encoder_graph_batch_buckets == (1, 2, 4, 8, 12, 16)
    model = SimpleNamespace(
        init_encoder_graphs=lambda buckets, feature_len: calls.append(
            (list(buckets), feature_len)
        )
    )

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=4096),
        generation_cuda_graph_enabled=True,
    )
    assert calls == [([1, 2], 3000)]

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=4096),
        generation_cuda_graph_enabled=False,
    )
    assert calls == [([1, 2], 3000)]


def test_whisper_encoder_cuda_graph_buckets_follow_final_prefill_budget() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    calls: list[list[int]] = []
    builder = WhisperASREngineBuilder(
        max_running_requests=16,
        max_new_tokens=32,
        mem_fraction_static=0.2,
        enable_encoder_cuda_graph=True,
        encoder_graph_batch_buckets=[8, 1, 4, 4, 16],
    )
    builder.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(nb_max_frames=3000)
    )
    builder.encoder_token_count = 1500
    model = SimpleNamespace(
        init_encoder_graphs=lambda buckets, feature_len: calls.append(list(buckets))
    )

    builder.setup_model_resources(
        model,
        server_args=SimpleNamespace(max_prefill_tokens=8192),
        generation_cuda_graph_enabled=True,
    )

    assert calls == [[1, 4]]


def test_whisper_disables_chunked_prefill_for_atomic_encoder_prefix() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    builder = WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )
    defaults = builder.generation_defaults(dtype="float16")

    assert defaults["max_prefill_tokens"] == 4096
    assert defaults["chunked_prefill_size"] == 0

    overrides = {"chunked_prefill_size": 0}
    builder.adjust_overrides(overrides)
    assert overrides["chunked_prefill_size"] == 0

    with pytest.raises(ValueError, match="encoder prefix must be admitted atomically"):
        builder.adjust_overrides({"chunked_prefill_size": 4096})


def test_whisper_prefill_coalescing_defaults_are_forwarded() -> None:
    from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder

    builder = WhisperASREngineBuilder(
        max_running_requests=16,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )

    assert builder.extra_scheduler_kwargs() == {
        "request_build_max_workers": 2,
        "request_build_max_pending": 16,
        "prefill_coalesce_requests": 2,
        "prefill_coalesce_wait_ms": 6.0,
        "prefill_coalesce_when_idle": True,
        "prefill_coalesce_requires_pending_builds": True,
        "prefill_coalesce_after_builds_during_decode": False,
    }


def test_whisper_asr_config_uses_single_batched_stage() -> None:
    config = WhisperASRPipelineConfig(model_path="openai/whisper-large-v3")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_whisper_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert config.stages[0].factory_args["enable_encoder_cuda_graph"] is True
    assert config.stages[0].factory_args["request_build_max_workers"] == 2
    assert config.stages[0].factory_args["request_build_max_pending"] == 16
    assert config.stages[0].factory_args["prefill_coalesce_requests"] == 2
    assert config.stages[0].factory_args["prefill_coalesce_wait_ms"] == 6.0
    assert config.stages[0].factory_args["prefill_coalesce_when_idle"] is True
    assert WhisperASRPipelineConfig.mem_fraction_role_to_stage() == {"asr": "asr"}
    assert WhisperASRPipelineConfig.generation_sglang_role_to_stage() == {
        "generation": "asr"
    }
    assert (
        config.stages[0].factory_args["prefill_coalesce_requires_pending_builds"]
        is True
    )
    assert (
        config.stages[0].factory_args["prefill_coalesce_after_builds_during_decode"]
        is False
    )
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("WhisperForConditionalGeneration")
        is WhisperASRPipelineConfig
    )


def test_whisper_asr_rtx4090_profile_is_bf16_and_bounded() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = ConfigManager.from_file(
        str(repo_root / "examples/configs/whisper_asr_rtx4090.yaml")
    ).config
    stage = config.stages[0]

    factory_args = resolve_stage_static_factory_args(stage, config)

    assert factory_args["dtype"] == "bfloat16"
    assert factory_args["max_running_requests"] == 16
    assert factory_args["enable_torch_compile"] is False
    assert factory_args["server_args_overrides"]["mem_fraction_static"] == 0.65


def test_whisper_asr_threads_explicit_cuda_graph_bs(monkeypatch) -> None:
    build_kwargs: dict[str, object] = {}
    runtime_logs: list[tuple[str, tuple[object, ...]]] = []
    fake_processor = SimpleNamespace(
        tokenizer=object(),
        feature_extractor=SimpleNamespace(nb_max_frames=3000),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                    max_target_positions=448
                )
            ),
            AutoProcessor=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: fake_processor
            ),
            GenerationConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: object()
            ),
        ),
    )
    monkeypatch.setattr(
        whisper_request_builders,
        "make_whisper_scheduler_adapters",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        whisper_engine_builder,
        "get_visible_gpu_sm_version",
        lambda gpu_id: 89,
    )
    monkeypatch.setattr(
        whisper_engine_builder,
        "get_process_gpu_memory_bytes",
        lambda gpu_id: 0,
    )
    monkeypatch.setattr(
        whisper_engine_builder.logger,
        "info",
        lambda message, *args: runtime_logs.append((message, args)),
    )
    monkeypatch.setattr(
        model_runner_base,
        "ModelRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        omni_scheduler,
        "OmniScheduler",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def _fake_server_args_builder(model_path, context_length, **overrides):
        build_kwargs["context_length"] = context_length
        build_kwargs.update(overrides)
        server_args = FakeServerArgs(**overrides)
        server_args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(
                max_bs=overrides["cuda_graph_max_bs"],
                bs=overrides["cuda_graph_bs"],
            ),
            prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
        )
        return server_args

    def _fake_create_infrastructure(server_args, gpu_id, **kwargs):
        model_worker = SimpleNamespace(model_runner=SimpleNamespace(model=object()))
        return False, (
            model_worker,
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        _fake_server_args_builder,
    )
    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        _fake_create_infrastructure,
    )

    whisper_asr_stages.create_sglang_whisper_asr_executor("dummy")

    assert build_kwargs["cuda_graph_max_bs"] == 16
    assert build_kwargs["cuda_graph_bs"] == [1, 2, 4, 8, 12, 16]
    assert build_kwargs["enable_torch_compile"] is True
    runtime_log = next(
        entry for entry in runtime_logs if entry[0].startswith("Whisper ASR runtime")
    )
    assert runtime_log[1][4] == [1, 2, 4, 8, 12, 16]
    checkpoints = [
        entry[1][0]
        for entry in runtime_logs
        if entry[0].startswith("Whisper ASR memory checkpoint")
    ]
    assert checkpoints == ["pre_model_load", "post_static_allocation"]
    # note (jiannan-17): context_length = encoder_token_count + max_prev_tokens + max_new_tokens + 8
    assert build_kwargs["context_length"] == 1500 + 224 + 256 + 8
    assert build_kwargs["chunked_prefill_size"] == 0
