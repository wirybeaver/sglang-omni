# SPDX-License-Identifier: Apache-2.0
"""Whisper zh-to-English translation benchmark on pinned CoVoST2 audio."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import soundfile as sf
from jiwer import process_words

from benchmarks.dataset.prepare import (
    COVOST2_DATASET_CONFIG,
    COVOST2_DATASET_ID,
    COVOST2_DATASET_REVISION,
)
from benchmarks.runtime_metrics import ResourceMonitor, collect_benchmark_provenance
from benchmarks.tasks.asr import (
    OMNI_WHISPER_MODEL_PATH,
    PINNED_ASR_MODEL_REVISIONS,
    normalize_text,
)
from benchmarks.tts_serving.http_contracts import (
    ResponseBodyTooLarge,
    read_response_body,
)

DATASET_ID = COVOST2_DATASET_ID
DATASET_CONFIG = COVOST2_DATASET_CONFIG
DATASET_SPLIT = "test"
DATASET_REVISION = COVOST2_DATASET_REVISION
MODEL_REVISION = PINNED_ASR_MODEL_REVISIONS[OMNI_WHISPER_MODEL_PATH]
EXPECTED_SAMPLES = 4898


@dataclass(frozen=True)
class TranslationSample:
    sample_id: str
    audio_bytes: bytes
    filename: str
    reference: str
    duration_s: float


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--backend",
        choices=("server", "transformers"),
        default="server",
        help="Run the SGLang server endpoint or a same-revision HF reference.",
    )
    parser.add_argument("--model-path", default=OMNI_WHISPER_MODEL_PATH)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--dataset-split", default=DATASET_SPLIT)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--source-language",
        default="zh",
        help="Optional Whisper source-language hint; use an empty value to omit it.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--concurrency", type=_positive_int, default=8)
    parser.add_argument("--warmup-samples", type=int, default=8)
    parser.add_argument("--request-timeout-s", type=_positive_float, default=120.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--monitor-interval-s", type=_positive_float, default=0.2)
    parser.add_argument(
        "--gpu-process-pid",
        dest="gpu_process_pids",
        type=_positive_int,
        action="append",
        help=(
            "NVML/host PID to attribute to the server; repeat for multiple GPU "
            "processes. Required for --backend server."
        ),
    )
    parser.add_argument(
        "--launch-command",
        default=os.environ.get("SGLANG_OMNI_BENCHMARK_LAUNCH_COMMAND"),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--torch-compile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument("--mem-fraction-static", type=float, default=0.65)
    parser.add_argument("--output", default="whisper_translation_results.json")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    constraints = (
        (
            args.model_path == OMNI_WHISPER_MODEL_PATH,
            "--model-path must match the pinned Whisper model",
        ),
        (
            args.model_revision == MODEL_REVISION,
            "--model-revision must match the pinned Whisper revision",
        ),
        (
            args.dataset_id == DATASET_ID,
            "--dataset-id must match the pinned CoVoST2 dataset",
        ),
        (
            args.dataset_config == DATASET_CONFIG,
            "--dataset-config must match the pinned CoVoST2 config",
        ),
        (
            args.dataset_split == DATASET_SPLIT,
            "--dataset-split must match the pinned CoVoST2 split",
        ),
        (
            args.dataset_revision == DATASET_REVISION,
            "--dataset-revision must match the pinned CoVoST2 revision",
        ),
        (args.concurrency > 0, "--concurrency must be greater than zero"),
        (args.warmup_samples >= 0, "--warmup-samples must be nonnegative"),
        (args.max_samples >= 0, "--max-samples must be nonnegative"),
        (
            math.isfinite(args.request_timeout_s) and args.request_timeout_s > 0,
            "--request-timeout-s must be finite and greater than zero",
        ),
        (
            math.isfinite(args.monitor_interval_s) and args.monitor_interval_s > 0,
            "--monitor-interval-s must be finite and greater than zero",
        ),
        (args.gpu_index >= 0, "--gpu-index must be nonnegative"),
        (
            not args.gpu_process_pids or all(pid > 0 for pid in args.gpu_process_pids),
            "--gpu-process-pid values must be positive",
        ),
        (
            args.backend != "server" or bool(args.gpu_process_pids),
            "at least one --gpu-process-pid is required for server resource "
            "attribution",
        ),
    )
    for valid, message in constraints:
        if not valid:
            raise ValueError(message)


def load_samples(args: argparse.Namespace) -> list[TranslationSample]:
    from datasets import Audio, load_dataset

    dataset = load_dataset(
        args.dataset_id,
        args.dataset_config,
        split=args.dataset_split,
        revision=args.dataset_revision,
    ).cast_column("audio", Audio(decode=False))
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    elif len(dataset) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} CoVoST2 samples, got {len(dataset)}"
        )

    samples: list[TranslationSample] = []
    for index, row in enumerate(dataset):
        audio_bytes, filename = _audio_payload(row["audio"], index=index)
        samples.append(
            TranslationSample(
                sample_id=Path(filename).stem or f"sample-{index}",
                audio_bytes=audio_bytes,
                filename=filename,
                reference=str(row["gt"]),
                duration_s=_duration_s(audio_bytes),
            )
        )
    return samples


def _audio_payload(audio: dict[str, Any], *, index: int) -> tuple[bytes, str]:
    filename = str(audio.get("path") or f"sample-{index}.mp3")
    audio_bytes = audio.get("bytes")
    if audio_bytes is None:
        audio_bytes = Path(filename).read_bytes()
    if not isinstance(audio_bytes, bytes):
        raise TypeError(f"Unexpected audio payload type: {type(audio_bytes).__name__}")
    return audio_bytes, Path(filename).name


async def run_benchmark(
    args: argparse.Namespace,
    samples: list[TranslationSample],
) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("No CoVoST2 samples were loaded")

    timeout = aiohttp.ClientTimeout(total=args.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for sample in samples[: args.warmup_samples]:
            await _translate_one(session, args, sample)

        monitor = ResourceMonitor(
            gpu_index=args.gpu_index,
            interval_s=args.monitor_interval_s,
            gpu_process_pids=args.gpu_process_pids,
        ).start()
        started = time.perf_counter()
        semaphore = asyncio.Semaphore(args.concurrency)

        async def _bounded(sample: TranslationSample) -> dict[str, Any]:
            async with semaphore:
                return await _translate_one(session, args, sample)

        try:
            results = await asyncio.gather(*(_bounded(sample) for sample in samples))
        finally:
            wall_clock_s = time.perf_counter() - started
            resources = monitor.stop()

    return _build_result(args, samples, results, wall_clock_s, resources)


def run_transformers_reference(
    args: argparse.Namespace,
    samples: list[TranslationSample],
) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("No CoVoST2 samples were loaded")
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    from sglang_omni.utils.audio import load_audio

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        revision=args.model_revision,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        revision=args.model_revision,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()

    def _translate(sample: TranslationSample) -> dict[str, Any]:
        audio = load_audio(
            sample.audio_bytes,
            source_name="Whisper HF reference",
            target_sample_rate=16000,
        )
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            device="cuda",
            dtype=torch.bfloat16,
        )
        generate_kwargs = {"task": "translate"}
        if args.source_language:
            generate_kwargs["language"] = args.source_language
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(input_features, **generate_kwargs)
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return {
            "status": 200,
            "text": text,
            "latency_s": time.perf_counter() - started,
            "error": None,
        }

    for sample in samples[: args.warmup_samples]:
        _translate(sample)
    monitor = ResourceMonitor(
        gpu_index=args.gpu_index,
        interval_s=args.monitor_interval_s,
        gpu_process_pids=args.gpu_process_pids or [os.getpid()],
    ).start()
    started = time.perf_counter()
    try:
        results = [_translate(sample) for sample in samples]
    finally:
        wall_clock_s = time.perf_counter() - started
        resources = monitor.stop()
    return _build_result(args, samples, results, wall_clock_s, resources)


def _build_result(
    args: argparse.Namespace,
    samples: list[TranslationSample],
    results: list[dict[str, Any]],
    wall_clock_s: float,
    resources: dict[str, Any],
) -> dict[str, Any]:
    successful = [result for result in results if result["status"] == 200]
    references = [
        normalize_text(sample.reference, "en")
        for sample, result in zip(samples, results, strict=True)
        if result["status"] == 200
    ]
    hypotheses = [
        normalize_text(result["text"], "en")
        for result in results
        if result["status"] == 200
    ]
    latencies = [result["latency_s"] for result in successful]
    audio_seconds = sum(
        sample.duration_s
        for sample, result in zip(samples, results, strict=True)
        if result["status"] == 200
    )
    quality = _translation_quality(references, hypotheses)
    resource_attribution_passed = args.backend != "server" or (
        resources.get("available") is True and resources.get("error") is None
    )
    return {
        "schema_version": 1,
        "passed": (
            len(successful) == len(samples)
            and all(hypotheses)
            and resource_attribution_passed
        ),
        "provenance": collect_benchmark_provenance(
            model_id=args.model_path,
            model_revision=args.model_revision,
            dataset_id=f"{args.dataset_id}/{args.dataset_config}/{args.dataset_split}",
            dataset_revision=args.dataset_revision,
            launch_command=args.launch_command,
            server_config={
                "backend": args.backend,
                "dtype": args.dtype,
                "attention_backend": args.attention_backend,
                "cuda_graph": args.cuda_graph,
                "torch_compile": args.torch_compile,
                "max_running_requests": args.max_running_requests,
                "mem_fraction_static": args.mem_fraction_static,
            },
            evaluation_input_sha256=_evaluation_input_sha256(samples),
        ),
        "config": {
            "backend": args.backend,
            "samples": len(samples),
            "concurrency": args.concurrency,
            "warmup_samples": min(args.warmup_samples, len(samples)),
            "source_language": args.source_language or None,
            "gpu_process_pids": args.gpu_process_pids
            or ([os.getpid()] if args.backend == "transformers" else []),
        },
        "summary": {
            "total": len(samples),
            "evaluated": len(successful),
            "skipped": len(samples) - len(successful),
            "wall_clock_s": wall_clock_s,
            "requests_per_s": len(successful) / max(wall_clock_s, 1e-9),
            "audio_seconds_per_s": audio_seconds / max(wall_clock_s, 1e-9),
            "latency_mean_s": statistics.mean(latencies) if latencies else None,
            "latency_p95_s": _percentile(latencies, 95),
            **quality,
        },
        "resources": resources,
        "per_sample": [
            {
                "id": sample.sample_id,
                "reference": sample.reference,
                "hypothesis": result["text"],
                "duration_s": sample.duration_s,
                **result,
            }
            for sample, result in zip(samples, results, strict=True)
        ],
    }


async def _translate_one(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    sample: TranslationSample,
) -> dict[str, Any]:
    form = aiohttp.FormData()
    form.add_field("model", args.model_path)
    form.add_field("response_format", "json")
    if args.source_language:
        form.add_field("language", args.source_language)
    form.add_field(
        "file",
        sample.audio_bytes,
        filename=sample.filename,
        content_type=_content_type(sample.filename),
    )
    started = time.perf_counter()
    status: int | None = None
    try:
        async with session.post(
            f"http://{args.host}:{args.port}/v1/audio/translations",
            data=form,
        ) as response:
            status = response.status
            body = (await read_response_body(response)).decode(
                "utf-8", errors="replace"
            )
            text = ""
            if response.status == 200:
                try:
                    text = str(json.loads(body).get("text", ""))
                except json.JSONDecodeError:
                    pass
            return {
                "status": response.status,
                "text": text,
                "latency_s": time.perf_counter() - started,
                "error": None if response.status == 200 else body[:500],
            }
    except (aiohttp.ClientError, asyncio.TimeoutError, ResponseBodyTooLarge) as exc:
        return {
            "status": status,
            "text": "",
            "latency_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _evaluation_input_sha256(samples: list[TranslationSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        fields = (
            sample.sample_id.encode(),
            sample.reference.encode(),
            hashlib.sha256(sample.audio_bytes).digest(),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def _translation_quality(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, Any]:
    if not references:
        return {"corpus_wer": None, "bleu": None, "chrf": None}
    metrics: dict[str, Any] = {
        "corpus_wer": process_words(references, hypotheses).wer,
        "bleu": None,
        "chrf": None,
    }
    try:
        from sacrebleu.metrics import BLEU, CHRF

        metrics["bleu"] = BLEU().corpus_score(hypotheses, [references]).score
        metrics["chrf"] = CHRF().corpus_score(hypotheses, [references]).score
    except ImportError:
        metrics["quality_warning"] = (
            "sacrebleu is not installed; run `uv pip install sacrebleu` "
            "to report BLEU/chrF"
        )
    return metrics


def _duration_s(audio_bytes: bytes) -> float:
    try:
        info = sf.info(io.BytesIO(audio_bytes))
        return info.frames / float(info.samplerate) if info.samplerate else 0.0
    except RuntimeError:
        return 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round((percentile / 100.0) * (len(ordered) - 1))),
    )
    return ordered[index]


def _content_type(filename: str) -> str:
    return "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"


def main() -> None:
    args = parse_args()
    _validate_args(args)
    samples = load_samples(args)
    if args.backend == "transformers":
        result = run_transformers_reference(args, samples)
    else:
        result = asyncio.run(run_benchmark(args, samples))
    output = Path(os.path.abspath(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"passed": result["passed"], "output": str(output)}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
