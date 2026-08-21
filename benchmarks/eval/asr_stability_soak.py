# SPDX-License-Identifier: Apache-2.0
"""Bounded concurrency, chaos cadence, and stage metrics for ASR soak tests."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

LATENCY_RESERVOIR_SIZE = 8192


async def run_soak(
    client: Any,
    args: Any,
    samples: list[Any],
    *,
    concurrencies: list[int],
    checkpoint: Callable[[str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every concurrency stage behind one global per-stage request cap."""
    randomizer = random.Random(args.seed)
    stage_duration_s = args.duration_s / len(concurrencies)
    stages: list[dict[str, Any]] = []
    chaos_events: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        stage, events = await _run_stage(
            client,
            args,
            samples,
            concurrency,
            stage_duration_s,
            checkpoint,
            randomizer,
        )
        stages.append(stage)
        chaos_events.extend(events)
    return stages, chaos_events


async def _run_stage(
    client: Any,
    args: Any,
    samples: list[Any],
    concurrency: int,
    duration_s: float,
    checkpoint: Callable[[str], dict[str, Any]],
    randomizer: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + duration_s
    counters: dict[str, int | float] = {
        key: 0
        for key in (
            "requests_issued",
            "requests",
            "successes",
            "unexpected_errors",
            "en",
            "zh",
            "zh_requests_issued",
            "translations",
            "chaos_requests",
            "chaos_events",
            "successful_audio_s",
        )
    }
    latency = _LatencyReservoir(args.seed ^ concurrency)
    slots = asyncio.Semaphore(concurrency)
    active_requests = 0
    max_in_flight = 0
    chaos_events: list[dict[str, Any]] = []

    async def limited(
        call: Callable[..., Awaitable[dict[str, Any]]],
        *call_args: Any,
    ) -> dict[str, Any]:
        nonlocal active_requests, max_in_flight
        async with slots:
            active_requests += 1
            max_in_flight = max(max_in_flight, active_requests)
            try:
                return await call(*call_args)
            finally:
                active_requests -= 1

    async def worker(worker_id: int) -> None:
        index = worker_id
        while time.monotonic() < deadline:
            sample = samples[index % len(samples)]
            index += concurrency
            counters["requests_issued"] += 1
            translate = False
            if sample.language == "zh":
                zh_sequence = int(counters["zh_requests_issued"])
                counters["zh_requests_issued"] += 1
                translate = args.include_translation and zh_sequence % 4 == 0
            result = await limited(
                client.translate if translate else client.transcribe,
                sample,
            )
            counters["translations"] += int(translate)
            counters["requests"] += 1
            counters[sample.language] += 1
            latency.add(result["latency_s"])
            valid_text = bool(result["text"])
            if translate:
                valid_text = looks_like_english_translation(result["text"])
            if result["status"] == 200 and valid_text:
                counters["successes"] += 1
                counters["successful_audio_s"] += sample.duration_s
            else:
                counters["unexpected_errors"] += 1

    async def chaos_worker() -> None:
        event_number = 0
        while True:
            if event_number:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                await asyncio.sleep(min(args.chaos_interval_s, remaining))
                if time.monotonic() >= deadline:
                    return
            else:
                await asyncio.sleep(0)
            sample = randomizer.choice(samples)
            if event_number % 2 == 0:
                counters["chaos_requests"] += 1
                result = await limited(
                    client.post_audio,
                    b"invalid",
                    "intentional-corrupt.wav",
                    sample.language,
                )
                event = {
                    "stage_concurrency": concurrency,
                    "kind": "malformed",
                    "status": result["status"],
                    "passed": result["status"] == 400,
                    "error": result["error"],
                }
            else:
                counters["chaos_requests"] += 2
                cancel = await limited(client.cancel, sample)
                reconnect = await limited(client.stream, sample)
                event = {
                    "stage_concurrency": concurrency,
                    "kind": "cancel_reconnect",
                    "status": cancel["status"],
                    "cancel_strategy": cancel["strategy"],
                    "passed": (
                        cancel["cancelled"]
                        and reconnect["status"] == 200
                        and reconnect["done"]
                    ),
                    "cancel_error": cancel["error"],
                    "reconnect_error": reconnect["error"],
                }
            chaos_events.append(event)
            counters["chaos_events"] += 1
            event_number += 1

    started = time.monotonic()
    tasks = [asyncio.create_task(worker(index)) for index in range(concurrency)]
    tasks.append(asyncio.create_task(chaos_worker()))
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    elapsed_s = time.monotonic() - started
    zh_requests = int(counters["zh"])
    return (
        {
            "concurrency": concurrency,
            "max_in_flight_observed": max_in_flight,
            "duration_s": elapsed_s,
            **counters,
            "total_http_requests": (
                int(counters["requests"]) + int(counters["chaos_requests"])
            ),
            "translation_target_ratio": 0.25 if args.include_translation else 0.0,
            "translation_observed_ratio": (
                float(counters["translations"]) / zh_requests if zh_requests else None
            ),
            "throughput_requests_per_s": (
                float(counters["successes"]) / max(elapsed_s, 1e-9)
            ),
            "audio_seconds_per_s": (
                float(counters["successful_audio_s"]) / max(elapsed_s, 1e-9)
            ),
            **latency.summary(),
            "memory": checkpoint(f"after_concurrency_{concurrency}"),
        },
        chaos_events,
    )


class _LatencyReservoir:
    def __init__(self, seed: int) -> None:
        self._randomizer = random.Random(seed)
        self._values: list[float] = []
        self._total = 0.0
        self._count = 0

    def add(self, value: float) -> None:
        self._count += 1
        self._total += value
        if len(self._values) < LATENCY_RESERVOIR_SIZE:
            self._values.append(value)
        else:
            replacement = self._randomizer.randrange(self._count)
            if replacement < LATENCY_RESERVOIR_SIZE:
                self._values[replacement] = value

    def summary(self) -> dict[str, Any]:
        return {
            "latency_mean_s": self._total / self._count if self._count else None,
            "latency_p95_s": _percentile(self._values, 95),
            "latency_samples_seen": self._count,
            "latency_reservoir_size": len(self._values),
            "latency_percentile_method": "bounded_reservoir",
        }


def looks_like_english_translation(text: str) -> bool:
    latin = sum("a" <= character.lower() <= "z" for character in text)
    cjk = sum(
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in text
    )
    return latin > cjk


def _percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((value / 100.0) * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]
