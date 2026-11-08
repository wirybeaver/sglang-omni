# SPDX-License-Identifier: Apache-2.0
"""Model-independent, token-aligned teacher capture results."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class TokenSpan:
    start_token_index: int
    end_token_index: int


@dataclass(kw_only=True)
class CaptureRequest:
    input_ids: torch.Tensor
    generated_spans: tuple[TokenSpan, ...]
    layer_ids: tuple[int, ...]
    result: CapturedSequence | None = None


@dataclass(frozen=True, kw_only=True)
class CapturedSequence:
    input_ids: torch.Tensor
    generated_spans: tuple[TokenSpan, ...]
    layer_ids: tuple[int, ...]
    auxiliary_hidden_states: torch.Tensor
    final_hidden_states: torch.Tensor


def capture_sequences(
    requests: list[CaptureRequest],
    sequence_token_counts: list[int],
    prefix_token_counts: list[int],
    auxiliary_hidden_states: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> None:
    """Copy complete prefill rows into independently owned CPU results."""
    assert len(requests) == len(sequence_token_counts) == len(prefix_token_counts)
    assert all(count == 0 for count in prefix_token_counts)
    assert auxiliary_hidden_states.ndim == final_hidden_states.ndim == 2
    assert (
        auxiliary_hidden_states.shape[0]
        == final_hidden_states.shape[0]
        == sum(sequence_token_counts)
    )
    token_offset = 0
    for request, token_count in zip(requests, sequence_token_counts, strict=True):
        assert request.input_ids.ndim == 1 and len(request.input_ids) == token_count
        assert (
            auxiliary_hidden_states.shape[1]
            == len(request.layer_ids) * final_hidden_states.shape[1]
        )
        assert all(
            0 <= span.start_token_index < span.end_token_index <= token_count
            for span in request.generated_spans
        )
        request.result = CapturedSequence(
            input_ids=request.input_ids.detach().to(device="cpu").clone(),
            generated_spans=request.generated_spans,
            layer_ids=request.layer_ids,
            auxiliary_hidden_states=auxiliary_hidden_states[
                token_offset : token_offset + token_count
            ]
            .detach()
            .to(device="cpu")
            .clone(),
            final_hidden_states=final_hidden_states[
                token_offset : token_offset + token_count
            ]
            .detach()
            .to(device="cpu")
            .clone(),
        )
        token_offset += token_count
