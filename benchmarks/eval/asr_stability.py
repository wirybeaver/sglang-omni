# SPDX-License-Identifier: Apache-2.0
"""ASR stability validation orchestration and evidence assembly."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import soundfile as sf

from benchmarks.dataset.prepare import DATASETS, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.asr_stability_client import ASRClient
from benchmarks.eval.asr_stability_soak import looks_like_english_translation, run_soak
from benchmarks.runtime_metrics import ResourceMonitor, collect_benchmark_provenance
from benchmarks.tasks.asr import FUN_ASR_MODEL_PATH, PINNED_ASR_MODEL_REVISIONS

SAMPLE_RATE = 16000
DEFAULT_AUDIO_BOUNDARY_MODELS = {FUN_ASR_MODEL_PATH}


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    language: str
    audio_bytes: bytes
    duration_s: float


def parse_concurrencies(value: str) -> list[int]:
    try:
        values = [int(token.strip()) for token in value.split(",")]
    except ValueError as exc:
        raise ValueError("--concurrencies must contain positive integers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError("--concurrencies must contain positive integers")
    return values


def validate_args(args: argparse.Namespace) -> list[int]:
    concurrencies = (
        parse_concurrencies(args.concurrencies)
        if isinstance(args.concurrencies, str)
        else list(args.concurrencies)
    )
    if not concurrencies or any(value <= 0 for value in concurrencies):
        raise ValueError("--concurrencies must contain positive integers")

    positive_fields = (
        "duration_s",
        "request_timeout_s",
        "monitor_interval_s",
        "chaos_interval_s",
    )
    nonnegative_fields = (
        "cooldown_s",
        "min_free_memory_mib",
        "max_retained_memory_mib",
    )
    for field in positive_fields:
        _require_finite_number(args, field, minimum=0.0, inclusive=False)
    for field in nonnegative_fields:
        _require_finite_number(args, field, minimum=0.0, inclusive=True)
    _require_finite_number(
        args,
        "max_audio_duration_s",
        minimum=0.2,
        inclusive=True,
    )
    if args.samples_per_language <= 0:
        raise ValueError("--samples-per-language must be > 0")
    if args.gpu_index < 0:
        raise ValueError("--gpu-index must be nonnegative")
    if not args.gpu_process_pids or any(pid <= 0 for pid in args.gpu_process_pids):
        raise ValueError(
            "at least one positive --gpu-process-pid is required for "
            "fail-closed resource attribution"
        )

    expected_revision = PINNED_ASR_MODEL_REVISIONS.get(args.model_path)
    if expected_revision is None:
        raise ValueError(
            "--model-path must be one of the supported consumer ASR models"
        )
    if args.model_revision != expected_revision:
        raise ValueError(
            "--model-revision must match the pinned revision for --model-path"
        )
    if _is_local_source(args.meta):
        raise ValueError(
            "bilingual stability validation requires a dataset with distinct "
            "en and zh splits; one local meta.lst cannot represent both"
        )

    if args.include_translation is None:
        args.include_translation = False
    if args.include_translation and not str(args.translation_source_language).strip():
        raise ValueError(
            "--translation-source-language is required when translation is enabled"
        )
    if args.check_audio_boundary is None:
        args.check_audio_boundary = args.model_path in DEFAULT_AUDIO_BOUNDARY_MODELS
    return concurrencies


async def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Run all phases and return the schema-v2 stability artifact."""
    concurrencies = validate_args(args)
    dataset_revision = resolve_dataset_revision(args)
    samples = load_prepared_samples(args, dataset_revision)
    input_sha256 = evaluation_input_sha256(samples)
    monitor = ResourceMonitor(
        gpu_index=args.gpu_index,
        interval_s=args.monitor_interval_s,
        gpu_process_pids=args.gpu_process_pids,
    ).start()
    checkpoints: list[dict[str, Any]] = []

    timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=max(concurrencies) + 8)
    try:
        await wait_for_resource_sample(monitor)
        checkpoints.append(memory_checkpoint("before_functional", monitor))
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        ) as session:
            client = ASRClient(session, args)
            functional = await run_functional_checks(client, args, samples)
            checkpoints.append(memory_checkpoint("after_functional", monitor))
            stages, chaos = await run_soak(
                client,
                args,
                samples,
                concurrencies=concurrencies,
                checkpoint=lambda label: memory_checkpoint(label, monitor),
            )
            checkpoints.append(memory_checkpoint("after_soak", monitor))
            await asyncio.sleep(args.cooldown_s)
            checkpoints.append(memory_checkpoint("after_cooldown", monitor))
            health_status = await client.health_status()
    finally:
        resources = monitor.stop()

    memory_validation = validate_memory_retention(
        checkpoints,
        resources,
        min_free_memory_mib=args.min_free_memory_mib,
        max_retained_memory_mib=args.max_retained_memory_mib,
    )
    unexpected_errors = sum(stage["unexpected_errors"] for stage in stages)
    soak_exercised = all(
        stage["requests"] > 0
        and stage["chaos_events"] > 0
        and (not args.include_translation or stage["translations"] > 0)
        for stage in stages
    )
    passed = (
        all(check["passed"] for check in functional)
        and soak_exercised
        and unexpected_errors == 0
        and all(event["passed"] for event in chaos)
        and health_status == 200
        and memory_validation["passed"]
    )
    server_config = _server_config(args)
    return {
        "schema_version": 2,
        "passed": passed,
        "provenance": collect_benchmark_provenance(
            model_id=args.model_path,
            model_revision=args.model_revision,
            dataset_id=args.meta,
            dataset_revision=dataset_revision,
            launch_command=args.launch_command,
            server_config=server_config,
            evaluation_input_sha256=input_sha256,
        ),
        "config": _artifact_config(
            args,
            concurrencies,
            dataset_revision,
            server_config,
        ),
        "functional": functional,
        "soak_stages": stages,
        "chaos_events": chaos,
        "resources": resources,
        "memory_checkpoints": checkpoints,
        "memory_validation": memory_validation,
        "final_health_status": health_status,
        "unexpected_errors": unexpected_errors,
    }


def resolve_dataset_revision(args: argparse.Namespace) -> str | None:
    if _is_local_source(args.meta):
        return None
    if args.dataset_revision is not None:
        return args.dataset_revision
    return SEEDTTS_DATASET_REVISION if args.meta == DATASETS["seedtts"] else None


def load_prepared_samples(
    args: argparse.Namespace,
    dataset_revision: str | None,
) -> list[PreparedSample]:
    by_language: dict[str, list[PreparedSample]] = {}
    for language in ("en", "zh"):
        loaded = load_seedtts_samples(
            args.meta,
            max_samples=args.samples_per_language,
            split=language,
            revision=dataset_revision,
        )
        if not loaded:
            raise RuntimeError(
                f"No SeedTTS {language} samples loaded from {args.meta!r}"
            )
        by_language[language] = [prepare_sample(sample, language) for sample in loaded]

    prepared: list[PreparedSample] = []
    for index in range(max(map(len, by_language.values()))):
        prepared.extend(
            samples[index]
            for language in ("en", "zh")
            if index < len(samples := by_language[language])
        )
    return prepared


def prepare_sample(sample: SampleInput, language: str) -> PreparedSample:
    audio_bytes = Path(sample.ref_audio).read_bytes()
    return PreparedSample(
        sample_id=sample.sample_id,
        language=language,
        audio_bytes=audio_bytes,
        duration_s=audio_duration_s(audio_bytes),
    )


def evaluation_input_sha256(samples: list[PreparedSample]) -> str:
    digest = hashlib.sha256(b"asr-stability-input-v1\0")
    for sample in samples:
        for value in (sample.sample_id, sample.language):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(len(sample.audio_bytes).to_bytes(8, "big"))
        digest.update(sample.audio_bytes)
    return digest.hexdigest()


async def wait_for_resource_sample(monitor: ResourceMonitor) -> None:
    deadline = time.monotonic() + max(1.0, monitor.interval_s * 5)
    while not monitor.samples and monitor.error is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(0.01, remaining))


async def run_functional_checks(
    client: ASRClient,
    args: argparse.Namespace,
    samples: list[PreparedSample],
) -> list[dict[str, Any]]:
    en_sample = next(sample for sample in samples if sample.language == "en")
    zh_sample = next(sample for sample in samples if sample.language == "zh")
    en_result = await client.transcribe(en_sample)
    checks = [
        expect_success("basic_en", en_result),
        expect_success("basic_zh", await client.transcribe(zh_sample)),
    ]
    if args.include_translation:
        checks.append(
            expect_translation_success(
                "basic_zh_to_english",
                await client.translate(zh_sample),
            )
        )

    stream = await client.stream(en_sample)
    checks.append(
        {
            "name": "streaming_consistency",
            "passed": (
                stream["status"] == 200
                and stream["done"]
                and stream["text"] == en_result["text"]
            ),
            "stream_text": stream["text"],
            "non_stream_text": en_result["text"],
            **{key: value for key, value in stream.items() if key != "text"},
        }
    )

    cancellation = await client.cancel(en_sample)
    await asyncio.sleep(0.5)
    reconnect = await client.stream(en_sample)
    checks.append(
        {
            "name": "stream_cancel_and_reconnect",
            "passed": (
                cancellation["cancelled"]
                and reconnect["status"] == 200
                and reconnect["done"]
            ),
            "cancel": cancellation,
            "reconnect_status": reconnect["status"],
            "reconnect_error": reconnect["error"],
        }
    )
    checks.extend(
        [
            expect_status(
                "empty_audio",
                await client.post_audio(b"", "empty.wav", "en"),
                400,
            ),
            expect_status(
                "corrupt_audio",
                await client.post_audio(
                    b"not-an-audio-file",
                    "corrupt.wav",
                    "en",
                ),
                400,
            ),
        ]
    )
    if args.check_audio_boundary:
        checks.extend(
            [
                expect_success(
                    "near_limit_audio",
                    await client.post_audio(
                        resize_wav(
                            en_sample.audio_bytes,
                            args.max_audio_duration_s - 0.1,
                        ),
                        "near-limit.wav",
                        "en",
                    ),
                ),
                expect_status(
                    "over_limit_audio",
                    await client.post_audio(
                        resize_wav(
                            en_sample.audio_bytes,
                            args.max_audio_duration_s + 0.1,
                        ),
                        "over-limit.wav",
                        "en",
                    ),
                    400,
                ),
            ]
        )
    return checks


def expect_translation_success(name: str, result: dict[str, Any]) -> dict[str, Any]:
    text = result["text"] if isinstance(result.get("text"), str) else ""
    return {
        "name": name,
        "passed": result["status"] == 200 and looks_like_english_translation(text),
        **result,
    }


def expect_success(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": result["status"] == 200 and bool(result["text"]),
        **result,
    }


def expect_status(
    name: str,
    result: dict[str, Any],
    expected_status: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": result["status"] == expected_status,
        "expected_status": expected_status,
        **result,
    }


def resize_wav(audio_bytes: bytes, duration_s: float) -> bytes:
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        raise ValueError("cannot resize empty audio")
    target_samples = round(duration_s * SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        old_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_length = round(len(audio) * SAMPLE_RATE / sample_rate)
        audio = np.interp(
            np.linspace(0.0, 1.0, num=new_length, endpoint=False),
            old_positions,
            audio,
        ).astype(np.float32)
    repeats = max(1, (target_samples + len(audio) - 1) // len(audio))
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.tile(audio, repeats)[:target_samples],
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


def audio_duration_s(audio_bytes: bytes) -> float:
    info = sf.info(io.BytesIO(audio_bytes))
    return info.frames / float(info.samplerate)


def memory_checkpoint(label: str, monitor: ResourceMonitor) -> dict[str, Any]:
    if not monitor.samples:
        return {
            "label": label,
            "monotonic_s": time.monotonic(),
            "sample_elapsed_s": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_free_mib": None,
            "gpu_process_memory_mib": None,
            "gpu_process_rss_mib": None,
            "error": monitor.error or "resource monitor has no samples",
        }
    sample = monitor.samples[-1]
    return {
        "label": label,
        "monotonic_s": time.monotonic(),
        "sample_elapsed_s": sample.elapsed_s,
        "gpu_memory_used_mib": sample.gpu_memory_used_mib,
        "gpu_memory_free_mib": sample.gpu_memory_free_mib,
        "gpu_process_memory_mib": sample.gpu_process_memory_mib,
        "gpu_process_rss_mib": sample.gpu_process_rss_mib,
        "error": monitor.error,
    }


def validate_memory_retention(
    checkpoints: list[dict[str, Any]],
    resources: dict[str, Any],
    *,
    min_free_memory_mib: float,
    max_retained_memory_mib: float,
) -> dict[str, Any]:
    checkpoint_memory = {
        checkpoint["label"]: {
            "used_mib": checkpoint.get("gpu_memory_used_mib"),
            "free_mib": checkpoint.get("gpu_memory_free_mib"),
            "process_mib": checkpoint.get("gpu_process_memory_mib"),
            "process_rss_mib": checkpoint.get("gpu_process_rss_mib"),
            "error": checkpoint.get("error"),
        }
        for checkpoint in checkpoints
    }
    before = checkpoint_memory.get("before_functional")
    cooldown = checkpoint_memory.get("after_cooldown")
    free_summary = resources.get("gpu_memory_free_mib")
    values = (
        free_summary.get("min") if isinstance(free_summary, dict) else None,
        before.get("used_mib") if before else None,
        cooldown.get("used_mib") if cooldown else None,
    )
    telemetry_error = (
        resources.get("error") is not None
        or before is None
        or before.get("error") is not None
        or cooldown is None
        or cooldown.get("error") is not None
    )
    valid_values = all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        for value in values
    )
    if telemetry_error or not valid_values:
        return {
            "passed": False,
            "error": "required GPU memory samples are unavailable",
            "checkpoints": checkpoint_memory,
        }

    minimum_free, before_used, cooldown_used = map(float, values)
    retained_mib = cooldown_used - before_used
    failures = []
    if minimum_free < min_free_memory_mib:
        failures.append("minimum free GPU memory fell below the required floor")
    if retained_mib > max_retained_memory_mib:
        failures.append("GPU memory retained after cooldown exceeded the limit")
    return {
        "passed": not failures,
        "error": "; ".join(failures) if failures else None,
        "minimum_free_mib": minimum_free,
        "required_free_mib": min_free_memory_mib,
        "retained_after_cooldown_mib": retained_mib,
        "maximum_retained_mib": max_retained_memory_mib,
        "checkpoints": checkpoint_memory,
    }


def _require_finite_number(
    args: argparse.Namespace,
    field: str,
    *,
    minimum: float,
    inclusive: bool,
) -> None:
    value = getattr(args, field)
    valid = math.isfinite(value) and (
        value >= minimum if inclusive else value > minimum
    )
    if not valid:
        comparison = ">=" if inclusive else ">"
        raise ValueError(
            f"--{field.replace('_', '-')} must be finite and {comparison} {minimum:g}"
        )


def _is_local_source(source: str) -> bool:
    return os.path.isfile(source) or source.endswith(".lst")


def _server_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dtype": args.dtype,
        "quantization": args.quantization,
        "attention_backend": args.attention_backend,
        "mm_attention_backend": args.mm_attention_backend,
        "cuda_graph": args.cuda_graph,
        "torch_compile": args.torch_compile,
        "max_running_requests": args.max_running_requests,
        "mem_fraction_static": args.mem_fraction_static,
    }


def _artifact_config(
    args: argparse.Namespace,
    concurrencies: list[int],
    dataset_revision: str | None,
    server_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "host": args.host,
        "port": args.port,
        "duration_s": args.duration_s,
        "concurrencies": concurrencies,
        "samples_per_language": args.samples_per_language,
        "request_timeout_s": args.request_timeout_s,
        "max_audio_duration_s": args.max_audio_duration_s,
        "check_audio_boundary": args.check_audio_boundary,
        "include_translation": args.include_translation,
        "translation_source_language": (
            args.translation_source_language if args.include_translation else None
        ),
        "min_free_memory_mib": args.min_free_memory_mib,
        "max_retained_memory_mib": args.max_retained_memory_mib,
        "gpu_index": args.gpu_index,
        "monitor_interval_s": args.monitor_interval_s,
        "gpu_process_pids": args.gpu_process_pids or [],
        "chaos_interval_s": args.chaos_interval_s,
        "cooldown_s": args.cooldown_s,
        "seed": args.seed,
        "model_path": args.model_path,
        "declared_model_revision": args.model_revision,
        "dataset_revision": dataset_revision,
        "declared_server": server_config,
    }
