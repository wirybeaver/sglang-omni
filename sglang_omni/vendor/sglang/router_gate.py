# SPDX-License-Identifier: Apache-2.0
"""Bailing router opt-out for the pinned engine dependency."""

import torch
import torch.nn.functional as functional
from sglang.srt.models import bailing_moe_v3
from sglang.srt.models.bailing_moe_v3 import BailingMoEGate


class LinearBailingMoEGate(BailingMoEGate):
    """Use the configured linear backend for every token count."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return functional.linear(hidden_states.to(self.weight.dtype), self.weight, None)


def patch_bailing_router_gate(disable_router_gate_matvec: bool) -> None:
    # note (wirybeaver): Remove this shim and factory interception once the pinned dependency includes https://github.com/sgl-project/sglang/pull/49142.
    if disable_router_gate_matvec:
        bailing_moe_v3.BailingMoEGate = LinearBailingMoEGate
    else:
        bailing_moe_v3.BailingMoEGate = BailingMoEGate
