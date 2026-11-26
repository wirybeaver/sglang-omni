# SPDX-License-Identifier: Apache-2.0
"""Keep the recorded utterance in a cascade dialog for an audio-language model."""

import base64
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import Field

from sglang_omni.audio.segmentation import (
    EMPTY_UTTERANCE,
    FINISH_UTTERANCE,
    UtteranceSegmenter,
)
from sglang_omni.audio.wav import pcm16_wav
from sglang_omni.models.streaming_cascade.base import (
    StreamComponent,
    audio_bytes,
    control_chunk,
)
from sglang_omni.models.streaming_cascade.dialog import Dialog, DialogConfig
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import SESSION_INPUT_SAMPLE_RATE_HZ, TURN_QUEUED
from sglang_omni.scheduling.session import ChunkEmitter


class AudioDialogConfig(DialogConfig):
    kind: Literal["audio_dialog"] = "audio_dialog"
    max_utterance_seconds: int = Field(default=60, gt=0)


class AudioDialog(Dialog):
    def __init__(self, config: AudioDialogConfig) -> None:
        super().__init__(config)
        self.segmenter = UtteranceSegmenter()
        self.pcm = bytearray()
        self.max_utterance_bytes = (
            config.max_utterance_seconds * SESSION_INPUT_SAMPLE_RATE_HZ * 2
        )

    def adopt(self, previous: StreamComponent) -> None:
        super().adopt(previous)
        if isinstance(previous, AudioDialog):
            self.segmenter = previous.segmenter
            self.pcm = previous.pcm
        else:
            pass

    def state_bytes(self) -> int:
        return super().state_bytes() + len(self.pcm)

    def collect(self, actions: list[TimedChunk | str]) -> None:
        assert self.emit is not None
        for action in actions:
            if isinstance(action, TimedChunk):
                samples = audio_bytes(action)
                if len(self.pcm) + len(samples) > self.max_utterance_bytes:
                    raise ValueError("utterance exceeds max_utterance_seconds")
                else:
                    self.pcm.extend(samples)
            elif action in (FINISH_UTTERANCE, EMPTY_UTTERANCE):
                wav = (
                    base64.b64encode(
                        pcm16_wav(bytes(self.pcm), SESSION_INPUT_SAMPLE_RATE_HZ)
                    ).decode("ascii")
                    if self.pcm
                    else None
                )
                self.pcm.clear()
                self.turns += 1
                self.waiting.append((self.turns, "", None, wav))
                self.emit(control_chunk(TURN_QUEUED, turn=self.turns))
            else:
                pass

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        self.emit = emit
        async for chunk in inputs:
            self.handle(chunk)
            self.collect(self.segmenter.step(chunk))
            self.respond_to_waiting()
        self.collect(self.segmenter.finish())
        self.respond_to_waiting()
