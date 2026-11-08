# SPDX-License-Identifier: Apache-2.0
"""Ming audio generation and replay adapter for teacher capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import queue
import threading
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from sglang_omni.capture.backend import CaptureInput, CaptureReply
from sglang_omni.models.ming3_asr.preprocessing import Ming3ASRPreprocessor
from sglang_omni.models.ming3_asr.stages import (
    create_preprocessing_executor,
    create_thinker_executor,
    load_checkpoint_config,
)
from sglang_omni.models.ming3_asr.text_streaming import Ming3ASRRequestData
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class MingCaptureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: str
    model_revision: str
    runtime_revision: str
    capture_layer_ids: tuple[int, ...] = Field(min_length=1)
    max_sequence_tokens: int = Field(default=4096, gt=1)
    max_new_tokens: int = Field(default=512, gt=0)
    device: str = "cuda:0"


class MingCaptureBackend:
    def __init__(self, configuration: MingCaptureConfig) -> None:
        self.configuration: MingCaptureConfig = configuration
        checkpoint = load_checkpoint_config(configuration.model_path)
        text_configuration = checkpoint["text_config"]
        self.end_token_id: int = text_configuration["eos_token_id"]
        layer_ids = configuration.capture_layer_ids
        if tuple(sorted(set(layer_ids))) != layer_ids or not all(
            0 <= layer_id < text_configuration["num_hidden_layers"]
            for layer_id in layer_ids
        ):
            raise ValueError(
                "Capture layers must be unique, ascending decoder indices within the checkpoint"
            )
        else:
            pass
        self.provenance: dict[str, JsonValue] = {
            "configuration": configuration.model_dump(mode="json"),
            "checkpoint_config": checkpoint,
            "tokenizer_revision": configuration.model_revision,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "sglang")
            },
            "layer_semantics": "zero-based decoder output with residual merged",
            "final_state": "post-norm before LM-head scaling",
            "capture": "full unchunked prefill; eager; TP1; radix cache disabled",
        }
        self.preprocessor: Ming3ASRPreprocessor = create_preprocessing_executor(
            configuration.model_path
        ).fn
        self.scheduler: OmniScheduler[Ming3ASRRequestData] = create_thinker_executor(
            configuration.model_path,
            capture_layer_ids=layer_ids,
            max_seq_len=configuration.max_sequence_tokens,
            max_new_tokens=configuration.max_new_tokens,
            stream_emit_interval_seconds=0,
            pre_lm_max_batch_size=1,
            pre_lm_max_batch_wait_ms=0,
            request_build_max_workers=1,
            request_build_max_pending=1,
            device=configuration.device,
        )
        self.thread: threading.Thread = threading.Thread(
            target=self.scheduler.start, name="teacher-capture"
        )
        self.thread.start()

    def run_request(self, payload: StagePayload) -> StagePayload:
        self.scheduler.inbox.put(
            IncomingMessage(
                request_id=payload.request_id, type="new_request", data=payload
            )
        )
        while True:
            try:
                message = self.scheduler.outbox.get(timeout=1)
            except queue.Empty:
                if not self.thread.is_alive():
                    raise RuntimeError(
                        "Capture scheduler exited before returning a result"
                    )
                else:
                    continue
            if message.type == "error":
                raise RuntimeError(f"Capture request failed: {message.data}")
            elif message.type == "result":
                assert message.request_id == payload.request_id
                assert isinstance(message.data, StagePayload)
                return message.data
            else:
                pass

    def capture(self, request: CaptureInput) -> CaptureReply:
        audio_path = request.inputs.get("audio_path")
        if not isinstance(audio_path, str):
            raise ValueError("Ming capture requires inputs.audio_path")
        elif set(request.params) - {"language", "prompt"}:
            raise ValueError(
                "Ming capture accepts only language and prompt parameters; generation is configured once"
            )
        else:
            pass
        audio_bytes = Path(audio_path).read_bytes()
        metadata: dict[str, JsonValue] = {
            "audio_sha256": hashlib.sha256(audio_bytes).hexdigest()
        }
        payload = StagePayload(
            request_id=request.request_id + "-generate",
            request=OmniRequest(
                inputs={"audio_bytes": audio_bytes}, params=request.params
            ),
            data=None,
        )
        state = self.preprocessor(payload).data
        if (
            len(state["input_ids"]) + self.configuration.max_new_tokens
            >= self.configuration.max_sequence_tokens
        ):
            return CaptureReply(
                sequence=None,
                metadata=metadata,
                rejection="insufficient_context_budget",
            )
        else:
            pass
        generated = self.run_request(payload).data
        token_ids = generated["output_ids"]
        metadata.update(
            prompt_token_count=len(state["input_ids"]),
            generated_token_ids=token_ids,
            finish_reason=generated["finish_reason"],
        )
        if (
            generated["finish_reason"] != "stop"
            or not token_ids
            or token_ids[-1] != self.end_token_id
        ):
            return CaptureReply(
                sequence=None,
                metadata=metadata,
                rejection="generation_did_not_end_with_eos",
            )
        else:
            pass
        replay = self.run_request(
            StagePayload(
                request_id=request.request_id + "-replay",
                request=OmniRequest(inputs={}, params={"teacher_token_ids": token_ids}),
                data=state,
            )
        ).data
        assert replay["capture"] is not None
        return CaptureReply(sequence=replay["capture"], metadata=metadata)

    def close(self) -> None:
        self.scheduler.stop()
        self.thread.join()


def create_capture_backend(configuration: dict[str, JsonValue]) -> MingCaptureBackend:
    return MingCaptureBackend(MingCaptureConfig.model_validate(configuration))
