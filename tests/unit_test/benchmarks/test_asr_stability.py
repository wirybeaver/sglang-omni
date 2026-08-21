# SPDX-License-Identifier: Apache-2.0
"""Configuration, orchestration, and soak contracts for ASR stability."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from benchmarks.eval import asr_stability
from benchmarks.eval.asr_stability import PreparedSample
from benchmarks.tasks.asr import (
    FUN_ASR_MODEL_PATH,
    OMNI_WHISPER_MODEL_PATH,
    QWEN3_ASR_MODEL_PATH,
)

EXPECTED_REVISIONS = {
    QWEN3_ASR_MODEL_PATH: "7278e1e70fe206f11671096ffdd38061171dd6e5",
    FUN_ASR_MODEL_PATH: "854d88f94205cd17d2afdb24332130d86fbe654a",
    OMNI_WHISPER_MODEL_PATH: "06f233fe06e710322aca913c1bc4249a0d71fce1",
}


def stability_args(model_path: str = FUN_ASR_MODEL_PATH, **overrides):
    values = {
        "host": "127.0.0.1",
        "port": 8000,
        "concurrencies": "1",
        "duration_s": 1.0,
        "samples_per_language": 1,
        "request_timeout_s": 1.0,
        "max_audio_duration_s": 30.0,
        "monitor_interval_s": 0.2,
        "chaos_interval_s": 30.0,
        "cooldown_s": 0.0,
        "min_free_memory_mib": 0.0,
        "max_retained_memory_mib": 0.0,
        "gpu_index": 0,
        "gpu_process_pids": [123],
        "model_path": model_path,
        "model_revision": EXPECTED_REVISIONS[model_path],
        "include_translation": None,
        "translation_source_language": "zh",
        "check_audio_boundary": None,
        "meta": "dataset",
        "dataset_revision": "revision",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def prepared_samples() -> list[PreparedSample]:
    return [
        PreparedSample(f"{language}-{index}", language, b"RIFF", 1.0)
        for index, language in enumerate(("en", "zh", "en", "zh"))
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", [1]), ("1,4,16", [1, 4, 16])],
)
def test_parse_concurrencies(raw: str, expected: list[int]) -> None:
    assert asr_stability.parse_concurrencies(raw) == expected


@pytest.mark.parametrize("raw", ["", "0", "-1", "1,", "one"])
def test_parse_concurrencies_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        asr_stability.parse_concurrencies(raw)


@pytest.mark.parametrize(
    ("model_path", "expected_boundary"),
    [
        (QWEN3_ASR_MODEL_PATH, False),
        (FUN_ASR_MODEL_PATH, True),
        (OMNI_WHISPER_MODEL_PATH, False),
    ],
)
def test_model_defaults_are_capability_specific(
    model_path: str,
    expected_boundary: bool,
) -> None:
    args = stability_args(model_path)

    asr_stability.validate_args(args)

    assert args.include_translation is False
    assert args.check_audio_boundary is expected_boundary


@pytest.mark.parametrize("revision", [None, "", "main"])
def test_validation_requires_the_pinned_model_revision(revision: str | None) -> None:
    args = stability_args(model_revision=revision)

    with pytest.raises(ValueError, match="--model-revision"):
        asr_stability.validate_args(args)


@pytest.mark.parametrize(
    "field",
    [
        "duration_s",
        "request_timeout_s",
        "max_audio_duration_s",
        "monitor_interval_s",
        "chaos_interval_s",
        "cooldown_s",
        "min_free_memory_mib",
        "max_retained_memory_mib",
    ],
)
def test_validation_rejects_nonfinite_operational_values(field: str) -> None:
    args = stability_args()
    setattr(args, field, float("nan"))

    with pytest.raises(ValueError, match="finite"):
        asr_stability.validate_args(args)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gpu_process_pids": None}, "--gpu-process-pid"),
        ({"meta": "local-seedtts.lst"}, "distinct en and zh splits"),
    ],
)
def test_validation_rejects_unattributable_inputs(
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        asr_stability.validate_args(stability_args(**overrides))


def test_memory_retention_is_fail_closed() -> None:
    checkpoints = [
        {
            "label": "before_functional",
            "gpu_memory_used_mib": 100.0,
            "gpu_memory_free_mib": 23000.0,
            "error": None,
        },
        {
            "label": "after_cooldown",
            "gpu_memory_used_mib": 300.0,
            "gpu_memory_free_mib": 22800.0,
            "error": None,
        },
    ]
    resources = {"gpu_memory_free_mib": {"min": 2200.0}, "error": None}

    passed = asr_stability.validate_memory_retention(
        checkpoints,
        resources,
        min_free_memory_mib=2048.0,
        max_retained_memory_mib=256.0,
    )
    low_free = asr_stability.validate_memory_retention(
        checkpoints,
        resources,
        min_free_memory_mib=4096.0,
        max_retained_memory_mib=256.0,
    )
    retained = asr_stability.validate_memory_retention(
        checkpoints,
        resources,
        min_free_memory_mib=2048.0,
        max_retained_memory_mib=128.0,
    )
    unavailable = asr_stability.validate_memory_retention(
        [],
        {"available": False},
        min_free_memory_mib=0.0,
        max_retained_memory_mib=0.0,
    )

    assert passed["passed"] is True
    assert passed["retained_after_cooldown_mib"] == 200.0
    assert low_free["passed"] is False
    assert retained["passed"] is False
    assert unavailable["error"] == "required GPU memory samples are unavailable"


@pytest.mark.parametrize(
    ("resources", "cooldown_used", "checkpoint_error"),
    [
        (
            {"gpu_memory_free_mib": {"min": 23000.0}, "error": "NVML failed"},
            100.0,
            None,
        ),
        (
            {"gpu_memory_free_mib": {"min": 23000.0}, "error": None},
            float("nan"),
            None,
        ),
        (
            {"gpu_memory_free_mib": {"min": 23000.0}, "error": None},
            100.0,
            "checkpoint sampling failed",
        ),
    ],
)
def test_memory_retention_rejects_invalid_telemetry(
    resources: dict,
    cooldown_used: float,
    checkpoint_error: str | None,
) -> None:
    checkpoints = [
        {"label": "before_functional", "gpu_memory_used_mib": 100.0},
        {
            "label": "after_cooldown",
            "gpu_memory_used_mib": cooldown_used,
            "error": checkpoint_error,
        },
    ]

    result = asr_stability.validate_memory_retention(
        checkpoints,
        resources,
        min_free_memory_mib=0.0,
        max_retained_memory_mib=0.0,
    )

    assert result["passed"] is False
    assert result["error"] == "required GPU memory samples are unavailable"


def test_translation_requires_english_output() -> None:
    english = asr_stability.expect_translation_success(
        "translation",
        {"status": 200, "text": "hello world", "error": None},
    )
    untranslated = asr_stability.expect_translation_success(
        "translation",
        {"status": 200, "text": "你好世界", "error": None},
    )

    assert english["passed"] is True
    assert untranslated["passed"] is False


def test_prepared_samples_are_language_interleaved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_splits: list[str] = []

    def fake_load(_meta, *, max_samples, split, revision):
        requested_splits.append(split)
        return [
            SimpleNamespace(sample_id=f"{split}-{index}")
            for index in range(max_samples)
        ]

    def fake_prepare(sample, language):
        return PreparedSample(sample.sample_id, language, b"RIFF", 1.0)

    monkeypatch.setattr(asr_stability, "load_seedtts_samples", fake_load)
    monkeypatch.setattr(asr_stability, "prepare_sample", fake_prepare)

    samples = asr_stability.load_prepared_samples(
        stability_args(samples_per_language=2),
        "revision",
    )

    assert requested_splits == ["en", "zh"]
    assert [sample.sample_id for sample in samples] == [
        "en-0",
        "zh-0",
        "en-1",
        "zh-1",
    ]


@pytest.mark.asyncio
async def test_validation_stops_resource_monitor_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = False
    monitor_kwargs: dict[str, object] = {}

    class FakeMonitor:
        def __init__(self, **kwargs) -> None:
            monitor_kwargs.update(kwargs)
            self.interval_s = 0.2
            self.samples = [object()]
            self.error = None

        def start(self):
            return self

        def stop(self):
            nonlocal stopped
            stopped = True
            return {"available": False, "error": "fake"}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    async def fail_functional(*_args, **_kwargs):
        raise RuntimeError("functional failed")

    monkeypatch.setattr(asr_stability, "ResourceMonitor", FakeMonitor)
    monkeypatch.setattr(
        asr_stability, "load_prepared_samples", lambda *_args: prepared_samples()
    )
    monkeypatch.setattr(
        asr_stability, "evaluation_input_sha256", lambda _samples: "hash"
    )
    monkeypatch.setattr(asr_stability, "memory_checkpoint", lambda *_args: {})
    monkeypatch.setattr(
        asr_stability.aiohttp, "TCPConnector", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        asr_stability.aiohttp, "ClientSession", lambda **_kwargs: FakeSession()
    )
    monkeypatch.setattr(asr_stability, "run_functional_checks", fail_functional)
    args = stability_args(
        include_translation=False,
        check_audio_boundary=False,
        gpu_process_pids=[123, 456],
    )

    with pytest.raises(RuntimeError, match="functional failed"):
        await asr_stability.run_validation(args)

    assert stopped is True
    assert monitor_kwargs["gpu_process_pids"] == [123, 456]


class PulsingClient:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def _pulse(self, result: dict) -> dict:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.002)
            return result
        finally:
            self.active -= 1

    async def transcribe(self, _sample):
        return await self._pulse({"status": 200, "text": "ok", "latency_s": 0.002})

    translate = transcribe

    async def post_audio(self, *_args):
        return await self._pulse({"status": 400, "error": None})

    async def cancel(self, _sample):
        return await self._pulse(
            {
                "status": 200,
                "strategy": "after_first_delta",
                "cancelled": True,
                "error": None,
            }
        )

    async def stream(self, _sample):
        return await self._pulse({"status": 200, "done": True, "error": None})


@pytest.mark.asyncio
async def test_soak_bounds_total_concurrency_and_global_cadence() -> None:
    client = PulsingClient()
    args = SimpleNamespace(
        seed=7,
        duration_s=0.04,
        include_translation=True,
        chaos_interval_s=0.005,
    )

    stages, chaos = await asr_stability.run_soak(
        client,
        args,
        prepared_samples(),
        concurrencies=[4],
        checkpoint=lambda _label: {},
    )

    stage = stages[0]
    assert client.active == 0
    assert client.max_active <= 4
    assert stage["max_in_flight_observed"] == client.max_active
    assert stage["chaos_events"] == len(chaos) > 0
    assert stage["total_http_requests"] == stage["requests"] + stage["chaos_requests"]
    assert stage["translations"] == (stage["zh_requests_issued"] + 3) // 4
    assert stage["translation_observed_ratio"] == stage["translations"] / stage["zh"]


@pytest.mark.asyncio
async def test_soak_cancellation_cleans_up_child_tasks() -> None:
    started = asyncio.Event()
    active = 0

    async def block(*_args):
        nonlocal active
        active += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    client = SimpleNamespace(
        transcribe=block,
        translate=block,
        post_audio=block,
        cancel=block,
        stream=block,
    )
    args = SimpleNamespace(
        seed=7,
        duration_s=10.0,
        include_translation=False,
        chaos_interval_s=30.0,
    )
    tasks_before = set(asyncio.all_tasks())
    soak_task = asyncio.create_task(
        asr_stability.run_soak(
            client,
            args,
            prepared_samples()[:2],
            concurrencies=[2],
            checkpoint=lambda _label: {},
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)

    soak_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await soak_task
    await asyncio.sleep(0)

    assert active == 0
    assert set(asyncio.all_tasks()) <= tasks_before
