# SPDX-License-Identifier: Apache-2.0
"""Token alignment and storage ownership of model-independent capture."""

import pytest
import torch

from sglang_omni.capture.features import CaptureRequest, TokenSpan, capture_sequences


def test_variable_length_capture_preserves_spans_and_independent_storage() -> None:
    requests = [
        CaptureRequest(
            input_ids=torch.tensor([10, 11, 12, 13]),
            generated_spans=(
                TokenSpan(start_token_index=1, end_token_index=2),
                TokenSpan(start_token_index=3, end_token_index=4),
            ),
            layer_ids=(0, 2),
        ),
        CaptureRequest(
            input_ids=torch.tensor([20, 21, 22]),
            generated_spans=(TokenSpan(start_token_index=1, end_token_index=3),),
            layer_ids=(0, 2),
        ),
    ]
    final_states = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    auxiliary_states = final_states.repeat(1, 2)
    capture_sequences(requests, [4, 3], [0, 0], auxiliary_states, final_states)
    final_states.zero_()
    auxiliary_states.zero_()
    assert requests[0].result.input_ids.tolist() == [10, 11, 12, 13]
    assert requests[0].result.generated_spans == (
        TokenSpan(start_token_index=1, end_token_index=2),
        TokenSpan(start_token_index=3, end_token_index=4),
    )
    assert requests[1].result.input_ids.tolist() == [20, 21, 22]
    assert requests[1].result.final_hidden_states[0].tolist() == [12, 13, 14]
    assert requests[1].result.auxiliary_hidden_states[-1].tolist() == [18, 19, 20] * 2
    assert requests[1].result.final_hidden_states.dtype == torch.float32
    with pytest.raises(AssertionError):
        capture_sequences(requests, [4, 3], [0, 1], auxiliary_states, final_states)
