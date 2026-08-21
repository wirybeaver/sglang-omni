# SPDX-License-Identifier: Apache-2.0
"""HTTP/SSE contracts for the ASR stability client."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import aiohttp
import pytest

from benchmarks.eval import asr_stability_client
from benchmarks.eval.asr_stability import PreparedSample
from benchmarks.eval.asr_stability_client import ASRClient
from benchmarks.tasks.asr import FUN_ASR_MODEL_PATH, OMNI_WHISPER_MODEL_PATH
from benchmarks.tts_serving.http_contracts import ResponseBodyTooLarge


class FakeContent:
    def __init__(self, lines: list[bytes] | None = None, body: bytes = b"") -> None:
        self._lines = lines or []
        self._body = body

    def __aiter__(self):
        async def generate():
            for line in self._lines:
                yield line

        return generate()

    def iter_chunked(self, _size: int):
        async def generate():
            yield self._body

        return generate()


class FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        lines: list[bytes] | None = None,
        body: bytes = b'{"text":"hi"}',
    ) -> None:
        self.status = status
        self.content = FakeContent(lines, body)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def close(self) -> None:
        self.closed = True


class FakeRequest:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __await__(self):
        async def resolve():
            return self._response

        return resolve().__await__()

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *_args):
        return None


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs):
        self.requests.append((url, kwargs))
        return FakeRequest(self.response)


def client_args(model_path: str = FUN_ASR_MODEL_PATH) -> SimpleNamespace:
    return SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model_path=model_path,
        translation_source_language="zh",
        request_timeout_s=1.0,
    )


def sample() -> PreparedSample:
    return PreparedSample("sample", "en", b"RIFF", 1.0)


@pytest.mark.asyncio
async def test_transport_error_becomes_request_evidence() -> None:
    class FailingSession:
        def post(self, *_args, **_kwargs):
            raise aiohttp.ClientConnectionError("connection refused")

    result = await ASRClient(FailingSession(), client_args()).post_audio(
        b"RIFF",
        "sample.wav",
        "en",
    )

    assert result["status"] == 0
    assert result["text"] == ""
    assert "connection refused" in result["error"]


@pytest.mark.asyncio
async def test_response_cap_becomes_request_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def too_large(_response):
        raise ResponseBodyTooLarge(bytes_read=65, max_bytes=64)

    session = FakeSession(FakeResponse())
    monkeypatch.setattr(asr_stability_client, "read_response_body", too_large)

    result = await ASRClient(session, client_args()).post_audio(
        b"RIFF",
        "sample.wav",
        "en",
    )

    assert result["status"] == 200
    assert result["text"] == ""
    assert "read cap" in result["error"]


@pytest.mark.asyncio
async def test_stream_uses_route_header_and_terminal_contract() -> None:
    session = FakeSession(
        FakeResponse(
            lines=[
                b'data: {"type":"transcript.text.delta","delta":"hi"}\n',
                b'data: {"type":"transcript.text.done","text":"hi"}\n',
                b"data: [DONE]\n",
            ]
        )
    )

    result = await ASRClient(session, client_args()).stream(sample())

    assert session.requests[0][1]["headers"] == {"x-sglang-omni-route-stream": "true"}
    assert result["status"] == 200
    assert result["done"] is True
    assert result["delta_events"] == 1
    assert result["first_event_latency_s"] is not None
    assert result["text"] == "hi"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lines", "expected_cancelled"),
    [
        ([b'data: {"type":"transcript.text.delta","delta":"hi"}\n'], True),
        (
            [
                b'data: {"type":"transcript.text.done","text":"complete"}\n',
                b"data: [DONE]\n",
            ],
            False,
        ),
    ],
)
async def test_cancel_requires_a_nonterminal_delta(
    lines: list[bytes],
    expected_cancelled: bool,
) -> None:
    response = FakeResponse(lines=lines)

    result = await ASRClient(FakeSession(response), client_args()).cancel(sample())

    assert result["received_event"] is expected_cancelled
    assert result["cancelled"] is expected_cancelled
    assert response.closed is True


@pytest.mark.asyncio
async def test_whisper_cancels_before_terminal_only_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    class BlockingSession:
        async def post(self, *_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    monkeypatch.setattr(asr_stability_client, "PREHEADER_CANCEL_DELAY_S", 0.0)

    result = await ASRClient(
        BlockingSession(),
        client_args(OMNI_WHISPER_MODEL_PATH),
    ).cancel(sample())

    assert started.is_set() and stopped.is_set()
    assert result["strategy"] == "before_first_event"
    assert result["cancelled"] is True
    assert result["received_event"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lines", "max_total"),
    [
        ([b"x" * (asr_stability_client.MAX_SSE_LINE_BYTES + 1)], None),
        ([b"data: {}\n", b"data: {}\n"], 10),
    ],
)
async def test_stream_response_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    lines: list[bytes],
    max_total: int | None,
) -> None:
    if max_total is not None:
        monkeypatch.setattr(asr_stability_client, "MAX_HTTP_RESPONSE_BYTES", max_total)

    result = await ASRClient(
        FakeSession(FakeResponse(lines=lines)),
        client_args(),
    ).stream(sample())

    assert "ResponseBodyTooLarge" in result["error"]
