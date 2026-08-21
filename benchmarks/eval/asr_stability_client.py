# SPDX-License-Identifier: Apache-2.0
"""Bounded HTTP/SSE client used by the ASR stability benchmark."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from benchmarks.tasks.asr import OMNI_WHISPER_MODEL_PATH
from benchmarks.tts_serving.http_contracts import (
    MAX_HTTP_RESPONSE_BYTES,
    ResponseBodyTooLarge,
    read_response_body,
)

STREAM_ROUTE_HEADERS = {"x-sglang-omni-route-stream": "true"}
MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_SSE_EVENTS = 100_000
PREHEADER_CANCEL_MODELS = {OMNI_WHISPER_MODEL_PATH}
PREHEADER_CANCEL_DELAY_S = 0.05


class ASRClient:
    """Exercise one ASR server through its public audio interface."""

    def __init__(self, session: aiohttp.ClientSession, args: Any) -> None:
        self._session = session
        self._host = args.host
        self._port = args.port
        self._model_path = args.model_path
        self._translation_source_language = str(
            getattr(args, "translation_source_language", "")
        ).strip()
        self._request_timeout_s = getattr(args, "request_timeout_s", 60.0)

    async def transcribe(self, sample: Any) -> dict[str, Any]:
        return await self.post_audio(
            sample.audio_bytes,
            f"{sample.sample_id}.wav",
            sample.language,
        )

    async def translate(self, sample: Any) -> dict[str, Any]:
        return await self.post_audio(
            sample.audio_bytes,
            f"{sample.sample_id}.wav",
            self._translation_source_language,
            endpoint="translations",
        )

    async def post_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        language: str | None,
        *,
        endpoint: str = "transcriptions",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            async with self._session.post(
                self._url(endpoint),
                data=self._form(audio_bytes, filename, language),
            ) as response:
                try:
                    body_bytes = await read_response_body(response)
                except ResponseBodyTooLarge as exc:
                    return _request_result(response.status, started, error=str(exc))
                body = body_bytes.decode("utf-8", errors="replace")
                text, error = _response_text(response.status, body)
                return _request_result(
                    response.status,
                    started,
                    text=text,
                    body=body[:500],
                    error=error,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return _request_result(
                0,
                started,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def stream(self, sample: Any) -> dict[str, Any]:
        status = 0
        reader = _SSEReader()
        first_event_latency_s: float | None = None
        delta_events = 0
        seen_done_event = False
        seen_done_sentinel = False
        final_text = ""
        started = time.perf_counter()
        error: str | None = None
        try:
            async with self._session.post(
                self._url(),
                data=self._sample_form(sample, stream=True),
                headers=STREAM_ROUTE_HEADERS,
            ) as response:
                status = response.status
                async for event in reader.events(response):
                    if event is None:
                        seen_done_sentinel = True
                        break
                    if first_event_latency_s is None:
                        first_event_latency_s = time.perf_counter() - started
                    event_type = event.get("type")
                    if event_type == "transcript.text.delta":
                        delta_events += 1
                    elif event_type == "transcript.text.done":
                        seen_done_event = True
                        if isinstance(event.get("text"), str):
                            final_text = event["text"]
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ResponseBodyTooLarge,
            ValueError,
        ) as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "status": status,
            "events": reader.event_count,
            "delta_events": delta_events,
            "response_bytes": reader.bytes_read,
            "first_event_latency_s": first_event_latency_s,
            "done": seen_done_event and seen_done_sentinel,
            "text": final_text,
            "error": error,
        }

    async def cancel(self, sample: Any) -> dict[str, Any]:
        if self._model_path in PREHEADER_CANCEL_MODELS:
            return await self._cancel_before_headers(sample)

        response: aiohttp.ClientResponse | None = None
        status = 0
        reader = _SSEReader()
        received_event = False
        error: str | None = None
        try:
            response = await self._session.post(
                self._url(),
                data=self._sample_form(sample, stream=True),
                headers=STREAM_ROUTE_HEADERS,
            )
            status = response.status
            received_event = await asyncio.wait_for(
                reader.read_first_delta(response),
                timeout=min(10.0, self._request_timeout_s),
            )
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ResponseBodyTooLarge,
            ValueError,
        ) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if response is not None:
                response.close()
        return {
            "status": status,
            "strategy": "after_first_delta",
            "cancelled": status == 200 and received_event,
            "received_event": received_event,
            "response_bytes": reader.bytes_read,
            "error": error,
        }

    async def health_status(self) -> int:
        try:
            async with self._session.get(
                f"http://{self._host}:{self._port}/health"
            ) as response:
                await read_response_body(response)
                return response.status
        except (aiohttp.ClientError, asyncio.TimeoutError, ResponseBodyTooLarge):
            return 0

    async def _cancel_before_headers(self, sample: Any) -> dict[str, Any]:
        """Disconnect a terminal-only stream while its first result is pending."""

        async def post_stream() -> aiohttp.ClientResponse:
            return await self._session.post(
                self._url(),
                data=self._sample_form(sample, stream=True),
                headers=STREAM_ROUTE_HEADERS,
            )

        response: aiohttp.ClientResponse | None = None
        request_task = asyncio.create_task(post_stream())
        status = 0
        cancelled = False
        error: str | None = None
        try:
            await asyncio.sleep(PREHEADER_CANCEL_DELAY_S)
            if request_task.done():
                response = request_task.result()
                status = response.status
                error = "response completed before the pre-header cancellation"
            else:
                request_task.cancel()
                outcome = (await asyncio.gather(request_task, return_exceptions=True))[
                    0
                ]
                cancelled = isinstance(outcome, asyncio.CancelledError)
                if not cancelled and isinstance(outcome, BaseException):
                    error = f"{type(outcome).__name__}: {outcome}"
                elif not cancelled:
                    response = outcome
                    status = response.status
                    error = "response completed before the pre-header cancellation"
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if not request_task.done():
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
            if response is not None:
                response.close()
        return {
            "status": status,
            "strategy": "before_first_event",
            "cancelled": cancelled,
            "received_event": False,
            "response_bytes": 0,
            "error": error,
        }

    def _sample_form(self, sample: Any, *, stream: bool) -> aiohttp.FormData:
        return self._form(
            sample.audio_bytes,
            f"{sample.sample_id}.wav",
            sample.language,
            stream=stream,
        )

    def _form(
        self,
        audio_bytes: bytes,
        filename: str,
        language: str | None,
        *,
        stream: bool = False,
    ) -> aiohttp.FormData:
        form = aiohttp.FormData()
        form.add_field("model", self._model_path)
        if language is not None:
            form.add_field("language", language)
        form.add_field("response_format", "json")
        if stream:
            form.add_field("stream", "true")
        form.add_field(
            "file",
            audio_bytes,
            filename=filename,
            content_type="audio/wav",
        )
        return form

    def _url(self, endpoint: str = "transcriptions") -> str:
        return f"http://{self._host}:{self._port}/v1/audio/{endpoint}"


class _SSEReader:
    def __init__(self) -> None:
        self.bytes_read = 0
        self.event_count = 0

    async def events(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncIterator[dict[str, Any] | None]:
        async for raw_line in response.content:
            self.bytes_read = _checked_stream_bytes(self.bytes_read, raw_line)
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                yield None
                return
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            self.event_count += 1
            if self.event_count > MAX_SSE_EVENTS:
                raise ValueError(
                    f"SSE event count exceeded benchmark cap ({MAX_SSE_EVENTS})"
                )
            yield event

    async def read_first_delta(self, response: aiohttp.ClientResponse) -> bool:
        async for event in self.events(response):
            if event is None or event.get("type") == "transcript.text.done":
                return False
            if (
                event.get("type") == "transcript.text.delta"
                and isinstance(event.get("delta"), str)
                and event["delta"]
            ):
                return True
        return False


def _response_text(status: int, body: str) -> tuple[str, str | None]:
    if status != 200:
        return "", None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return "", "response body is not valid JSON"
    if not isinstance(payload, dict):
        return "", "response JSON must be an object"
    if not isinstance(payload.get("text"), str):
        return "", "response JSON has no string text field"
    return payload["text"], None


def _request_result(
    status: int,
    started: float,
    *,
    text: str = "",
    body: str = "",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "text": text,
        "body": body,
        "latency_s": time.perf_counter() - started,
        "error": error,
    }


def _checked_stream_bytes(total: int, line: bytes) -> int:
    if len(line) > MAX_SSE_LINE_BYTES:
        raise ResponseBodyTooLarge(
            bytes_read=len(line),
            max_bytes=MAX_SSE_LINE_BYTES,
        )
    next_total = total + len(line)
    if next_total > MAX_HTTP_RESPONSE_BYTES:
        raise ResponseBodyTooLarge(
            bytes_read=next_total,
            max_bytes=MAX_HTTP_RESPONSE_BYTES,
        )
    return next_total
