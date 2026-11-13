# SPDX-License-Identifier: Apache-2.0
"""Router selection stays local to gates created under each factory setting."""

from unittest.mock import Mock, patch, sentinel

import pytest
import torch
from sglang.srt.models import bailing_moe_v3
from sglang.srt.models.bailing_moe_v3 import BailingMoEGate
from transformers import PretrainedConfig

from sglang_omni.vendor.sglang.router_gate import patch_bailing_router_gate


def test_router_opt_out_preserves_existing_gate_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bailing_moe_v3, "BailingMoEGate", BailingMoEGate)
    config = PretrainedConfig(hidden_size=4, num_experts=2)
    hidden_states = Mock(is_cuda=True, shape=(2, 4))
    hidden_states.to.return_value = sentinel.converted_input
    patch_bailing_router_gate(False)
    enabled_gate = bailing_moe_v3.BailingMoEGate(config, params_dtype=torch.float32)
    patch_bailing_router_gate(True)
    disabled_gate = bailing_moe_v3.BailingMoEGate(config, params_dtype=torch.float32)
    patch_bailing_router_gate(False)
    restored_gate = bailing_moe_v3.BailingMoEGate(config, params_dtype=torch.float32)
    with (
        patch.object(
            bailing_moe_v3, "router_gate_matvec", return_value=sentinel.matvec
        ),
        patch("torch.nn.functional.linear", return_value=sentinel.linear) as linear,
    ):
        assert enabled_gate(hidden_states) is sentinel.matvec
        assert restored_gate(hidden_states) is sentinel.matvec
        assert disabled_gate(hidden_states) is sentinel.linear
        linear.assert_called_once_with(
            sentinel.converted_input, disabled_gate.weight, None
        )
