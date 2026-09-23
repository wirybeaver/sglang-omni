# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniCPM-o variable-length DiT execution."""

import torch

from sglang_omni.models.minicpm_o.components.token2wav.dit import (
    CausalConvBlock,
    PackedSequenceMetadata,
)


def test_packed_metadata_round_trips_valid_frames() -> None:
    lengths = torch.tensor([3, 5, 2])
    mask = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
    metadata = PackedSequenceMetadata.from_mask(mask.unsqueeze(1), guard_width=2)
    dense = torch.arange(3 * 5 * 4).reshape(3, 5, 4)

    unpacked = metadata.unpack(metadata.pack(dense))

    torch.testing.assert_close(unpacked[metadata.valid], dense[metadata.valid])
    assert torch.count_nonzero(unpacked[~metadata.valid]) == 0


def test_packed_causal_conv_preserves_sequence_boundaries() -> None:
    torch.manual_seed(0)
    block = CausalConvBlock(4, 4).eval()
    rows = [torch.randn(length, 4) for length in (3, 5, 2)]

    expected = torch.cat([block(row.unsqueeze(0)).squeeze(0) for row in rows])
    lengths = torch.tensor([len(row) for row in rows])
    mask = torch.arange(int(lengths.max())).unsqueeze(0) < lengths.unsqueeze(1)
    metadata = PackedSequenceMetadata.from_mask(mask.unsqueeze(1), guard_width=2)
    actual = block.forward_packed(torch.cat(rows), metadata)

    torch.testing.assert_close(actual, expected)
