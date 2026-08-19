# SPDX-License-Identifier: Apache-2.0
"""Consumer-model contracts shared by the SeedTTS ASR benchmark."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.eval import benchmark_asr_seedtts
from benchmarks.tasks.asr import (
    FUN_ASR_MODEL_PATH,
    OMNI_WHISPER_MODEL_PATH,
    QWEN3_ASR_MODEL_PATH,
)

EXPECTED_MODEL_REVISIONS = {
    QWEN3_ASR_MODEL_PATH: "7278e1e70fe206f11671096ffdd38061171dd6e5",
    FUN_ASR_MODEL_PATH: "854d88f94205cd17d2afdb24332130d86fbe654a",
    OMNI_WHISPER_MODEL_PATH: "06f233fe06e710322aca913c1bc4249a0d71fce1",
}


def test_consumer_model_revision_table_is_exact() -> None:
    assert benchmark_asr_seedtts.PINNED_MODEL_REVISIONS == EXPECTED_MODEL_REVISIONS


@pytest.mark.parametrize("model_path", EXPECTED_MODEL_REVISIONS)
def test_seedtts_uses_the_pinned_model_revision_by_default(model_path: str) -> None:
    assert benchmark_asr_seedtts.resolve_model_revision(model_path, None) == (
        EXPECTED_MODEL_REVISIONS[model_path]
    )


def test_seedtts_rejects_a_nonpinned_model_revision() -> None:
    with pytest.raises(ValueError, match="pinned revision"):
        benchmark_asr_seedtts.resolve_model_revision(FUN_ASR_MODEL_PATH, "main")


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "0"])
def test_seedtts_monitor_interval_requires_finite_positive_value(raw: str) -> None:
    with pytest.raises(
        benchmark_asr_seedtts.argparse.ArgumentTypeError,
        match="finite and greater than zero",
    ):
        benchmark_asr_seedtts._positive_float(raw)


@pytest.mark.asyncio
async def test_seedtts_repeat_keeps_per_sample_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    per_sample = [
        {
            "id": "sample-1",
            "is_success": True,
            "audio_duration_s": 4.0,
            "text_ttft_s": 0.1,
            "inter_chunk_s": [0.02],
        }
    ]

    async def fake_run(*_args, **_kwargs):
        return {
            "summary": {
                "evaluated": 1,
                "total_samples": 1,
                "skipped": 0,
                "corpus_wer": 0.0,
                "wer_per_sample_max": 0.0,
            },
            "speed": {
                "throughput_samples_per_s": 0.5,
                "rtfx": 2.0,
                "latency_mean_s": 0.2,
                "latency_median_s": 0.2,
                "latency_p95_s": 0.2,
                "latency_p99_s": 0.2,
                "rtf_mean": 0.05,
                "rtf_p95": 0.05,
            },
            "wall_clock_s": 2.0,
            "worker": {},
            "per_sample": per_sample,
        }

    monkeypatch.setattr(benchmark_asr_seedtts, "run_asr_seedtts_once", fake_run)
    args = SimpleNamespace(
        disable_resource_monitor=True,
        gpu_index=0,
        monitor_interval_s=0.2,
        host="127.0.0.1",
        port=8000,
        model_path=FUN_ASR_MODEL_PATH,
        lang="en",
        stream=False,
        sample_util=False,
        save_raw_dir=None,
    )

    result = await benchmark_asr_seedtts._run_repeat(args, [], 1, 1)

    assert result["rtfx"] == 2.0
    assert result["audio_seconds_per_s"] == 2.0
    assert result["latency_median_s"] == 0.2
    assert result["per_sample"] == per_sample
