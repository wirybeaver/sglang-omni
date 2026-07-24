# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.eval import benchmark_whisper_translation
from benchmarks.tasks.asr import OMNI_WHISPER_MODEL_PATH
from benchmarks.tts_serving.http_contracts import ResponseBodyTooLarge


def test_translation_benchmark_revisions_are_exact() -> None:
    assert benchmark_whisper_translation.MODEL_REVISION == (
        "06f233fe06e710322aca913c1bc4249a0d71fce1"
    )
    assert benchmark_whisper_translation.DATASET_REVISION == (
        "e38a7a7fba8adcd1563b2169afc3bc7eed202a25"
    )
    assert benchmark_whisper_translation.DATASET_CONFIG == "zh_en"
    assert benchmark_whisper_translation.DATASET_SPLIT == "test"


def test_translation_server_requires_explicit_gpu_process_targets() -> None:
    args = SimpleNamespace(
        backend="server",
        concurrency=8,
        warmup_samples=1,
        max_samples=1,
        request_timeout_s=120.0,
        monitor_interval_s=0.2,
        gpu_index=0,
        gpu_process_pids=None,
    )

    with pytest.raises(ValueError, match="--gpu-process-pid"):
        benchmark_whisper_translation._validate_args(args)


def test_translation_quality_reports_exact_match() -> None:
    result = benchmark_whisper_translation._translation_quality(
        ["this is an exact translation match"],
        ["this is an exact translation match"],
    )

    assert result["corpus_wer"] == 0.0
    if result["bleu"] is not None:
        assert result["bleu"] > 99.0
        assert result["chrf"] > 99.0


def test_translation_audio_payload_uses_inline_bytes() -> None:
    audio_bytes, filename = benchmark_whisper_translation._audio_payload(
        {"bytes": b"mp3", "path": "/dataset/example.mp3"},
        index=0,
    )

    assert audio_bytes == b"mp3"
    assert filename == "example.mp3"


@pytest.mark.asyncio
async def test_translation_response_cap_becomes_request_evidence(monkeypatch) -> None:
    async def too_large(_response):
        raise ResponseBodyTooLarge(bytes_read=65, max_bytes=64)

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeSession:
        def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(benchmark_whisper_translation, "read_response_body", too_large)
    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model_path=OMNI_WHISPER_MODEL_PATH,
        source_language="zh",
    )
    sample = benchmark_whisper_translation.TranslationSample(
        sample_id="sample",
        audio_bytes=b"mp3",
        filename="sample.mp3",
        reference="hello",
        duration_s=1.0,
    )

    result = await benchmark_whisper_translation._translate_one(
        FakeSession(), args, sample
    )

    assert result["status"] == 200
    assert result["text"] == ""
    assert "read cap" in result["error"]
