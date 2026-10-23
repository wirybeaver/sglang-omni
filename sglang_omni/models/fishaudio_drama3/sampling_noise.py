# SPDX-License-Identifier: Apache-2.0
"""Request-local Philox inputs and the PyTorch sampling fallback."""

from __future__ import annotations

from dataclasses import dataclass

import torch

UNSEEDED_REQUEST = -1
MAXIMUM_REQUEST_SEED = (1 << 63) - 1
PHILOX_WORD_MASK = (1 << 32) - 1
PHILOX_ROUNDS = 10
PHILOX_MULTIPLIER_A = 0xD2511F53
PHILOX_MULTIPLIER_B = 0xCD9E8D57
PHILOX_KEY_INCREMENT_A = 0x9E3779B9
PHILOX_KEY_INCREMENT_B = 0xBB67AE85
UNIFORM_MANTISSA_BITS = 23


def checked_sampling_seed(seed: int | None) -> int:
    if seed is None:
        return UNSEEDED_REQUEST
    elif (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not (0 <= seed <= MAXIMUM_REQUEST_SEED)
    ):
        raise ValueError(
            f"Drama 3 request: seed must be an integer in "
            f"[0, {MAXIMUM_REQUEST_SEED}]; got {seed!r}"
        )
    else:
        return seed


@dataclass(frozen=True, kw_only=True)
class RequestSampling:
    seeds: torch.Tensor
    positions: torch.Tensor

    def exponential_noise(
        self, candidate_indices: torch.Tensor, stream_id: int
    ) -> torch.Tensor:
        """Philox4x32-10 with the same counters as the seeded Triton draws."""
        seed_low = self.seeds[:, None] & PHILOX_WORD_MASK
        seed_high = (self.seeds[:, None] >> 32) ^ (self.positions[:, None] >> 32)
        counter_zero = candidate_indices.to(torch.int64)
        counter_one = torch.zeros_like(counter_zero)
        counter_two = torch.full_like(counter_zero, stream_id)
        counter_three = self.positions[:, None] & PHILOX_WORD_MASK
        for _ in range(PHILOX_ROUNDS):
            product_a = counter_zero * PHILOX_MULTIPLIER_A
            product_b = counter_two * PHILOX_MULTIPLIER_B
            counter_zero = (
                ((product_b >> 32) & PHILOX_WORD_MASK) ^ counter_one ^ seed_low
            )
            counter_one = product_b & PHILOX_WORD_MASK
            counter_two = (
                ((product_a >> 32) & PHILOX_WORD_MASK) ^ counter_three ^ seed_high
            )
            counter_three = product_a & PHILOX_WORD_MASK
            seed_low = (seed_low + PHILOX_KEY_INCREMENT_A) & PHILOX_WORD_MASK
            seed_high = (seed_high + PHILOX_KEY_INCREMENT_B) & PHILOX_WORD_MASK
        uniform = (
            (counter_zero >> (32 - UNIFORM_MANTISSA_BITS)).to(torch.float32) + 0.5
        ) * (1.0 / (1 << UNIFORM_MANTISSA_BITS))
        return -torch.log(uniform)
