# SPDX-License-Identifier: Apache-2.0
"""Inference-only request interface for external teacher capture consumers."""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from sglang_omni.capture.features import CapturedSequence


class CaptureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    inputs: dict[str, JsonValue]
    params: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(kw_only=True)
class CaptureReply:
    sequence: CapturedSequence | None
    metadata: dict[str, JsonValue]
    rejection: str | None = None


class CaptureBackend(Protocol):
    @property
    def provenance(self) -> dict[str, JsonValue]: ...

    def capture(self, request: CaptureInput) -> CaptureReply: ...

    def close(self) -> None: ...
