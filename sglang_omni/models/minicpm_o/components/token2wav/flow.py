# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Flow for MiniCPM-o."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.minicpm_o.components.token2wav.conformer import (
    UpsampleConformerEncoderV2,
    make_pad_mask,
)
from sglang_omni.models.minicpm_o.components.token2wav.dit import DiT
from sglang_omni.models.minicpm_o.components.token2wav.fixed_packed import (
    FixedPackedLayout,
    build_fixed_packed_layout,
)
from sglang_omni.models.minicpm_o.components.token2wav.flow_cuda_graph import (
    CapturedPackedFlowGraph,
    FlowCudaGraphRunner,
)
from sglang_omni.models.minicpm_o.components.token2wav.packed_dit_cuda_graph import (
    CapturedPackedDiTGraph,
)


class CausalConditionalCFM(torch.nn.Module):

    def __init__(self, estimator: DiT, inference_cfg_rate: float = 0.7) -> None:
        super().__init__()
        self.estimator = estimator
        self.inference_cfg_rate = inference_cfg_rate
        self.out_channels = estimator.out_channels
        self.graph_runner: FlowCudaGraphRunner | None = None
        self.register_buffer(
            "rand_noise",
            torch.randn([1, self.out_channels, 50 * 600]),
            persistent=False,
        )

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        *,
        packed_valid_frames: int | None = None,
    ) -> torch.Tensor:
        t = t_span[0].expand(x.size(0))
        dt = t_span[1] - t_span[0]
        assert self.inference_cfg_rate > 0, "inference_cfg_rate better > 0"
        paired_mask = torch.cat([mask, mask], dim=0)
        paired_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        paired_speaker_embeddings = torch.cat(
            [speaker_embeddings, torch.zeros_like(speaker_embeddings)], dim=0
        )
        paired_mel_conditioning = torch.cat(
            [mel_conditioning, torch.zeros_like(mel_conditioning)], dim=0
        )
        graph_runner = self.graph_runner
        packed_runner = self.estimator.packed_graph_runner
        is_packed = self.estimator.enable_variable_length and x.shape[0] > 1
        packed_capacity = (
            packed_runner.fit(x.shape[0], x.shape[2], packed_valid_frames)
            if packed_runner is not None and is_packed
            else None
        )
        graph_key = (
            graph_runner.fit(x)
            if graph_runner is not None
            and (not is_packed or packed_capacity is not None)
            else None
        )
        packed_layout = (
            build_fixed_packed_layout(
                paired_mask.bool().squeeze(1).sum(dim=1, dtype=torch.int32),
                graph_key[1] if graph_key is not None else x.shape[2],
                packed_capacity,
                self.estimator.blocks[0].conv.kernel_size - 1,
            )
            if packed_capacity is not None
            else None
        )
        with (
            graph_runner.lock if graph_key is not None else nullcontext(),
            packed_runner.lock if packed_layout is not None else nullcontext(),
        ):
            captured_flow = (
                graph_runner.prepare(
                    graph_key,
                    x,
                    paired_mu,
                    paired_mask,
                    paired_speaker_embeddings,
                    paired_mel_conditioning,
                    packed_layout,
                )
                if graph_key is not None
                else None
            )
            captured_packed = (
                packed_runner.prepare(packed_layout)
                if packed_layout is not None
                else None
            )
            for step in range(1, len(t_span)):
                if isinstance(captured_flow, CapturedPackedFlowGraph):
                    graph_runner.run_packed_step(
                        captured_flow, packed_runner, captured_packed, t, dt
                    )
                elif captured_flow is not None:
                    graph_runner.run_step(captured_flow, t, dt)
                else:
                    x = self.euler_step(
                        x,
                        t,
                        dt,
                        paired_mu,
                        paired_mask,
                        paired_speaker_embeddings,
                        paired_mel_conditioning,
                        packed_layout=packed_layout,
                        packed_graph=captured_packed,
                    )
                t = t + dt
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t_span[step]
                else:
                    pass
            return (
                captured_flow.output[:, :, : x.shape[2]].clone()
                if captured_flow is not None
                else x
            )

    def euler_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        paired_mu: torch.Tensor,
        paired_mask: torch.Tensor,
        paired_speaker_embeddings: torch.Tensor,
        paired_mel_conditioning: torch.Tensor,
        *,
        packed_layout: FixedPackedLayout | None = None,
        packed_graph: CapturedPackedDiTGraph | None = None,
    ) -> torch.Tensor:
        derivative = self.estimator.forward(
            torch.cat([x, x], dim=0),
            paired_mask,
            paired_mu,
            torch.cat([t, t], dim=0),
            paired_speaker_embeddings,
            paired_mel_conditioning,
            packed_layout=packed_layout,
            packed_graph=packed_graph,
        )
        conditional_derivative, unconditional_derivative = derivative.chunk(2, dim=0)
        guided_derivative = (
            (1.0 + self.inference_cfg_rate) * conditional_derivative
            - self.inference_cfg_rate * unconditional_derivative
        )
        return x + dt * guided_derivative

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        mel_conditioning: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        packed_valid_frames: int | None = None,
    ) -> torch.Tensor:
        if n_timesteps <= 0:
            raise ValueError("n_timesteps must be positive")
        else:
            pass
        if mu.size(2) > self.rand_noise.size(2):
            raise ValueError(
                "Combined reference and generated audio exceed 600 seconds"
            )
        else:
            pass
        z = (
            self.rand_noise[:, :, : mu.size(2)].expand(mu.size(0), -1, -1).clone()
            * temperature
        )
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(
            z,
            t_span,
            mu,
            mask,
            speaker_embeddings,
            mel_conditioning,
            packed_valid_frames=packed_valid_frames,
        )


class CausalMaskedDiffWithXvec(torch.nn.Module):

    def __init__(
        self,
        encoder: UpsampleConformerEncoderV2,
        decoder: CausalConditionalCFM,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: Literal["mel"] = "mel",
        vocab_size: int = 6561,
    ) -> None:
        super().__init__()
        if output_type != "mel":
            raise ValueError("MiniCPM-o flow output must be mel")
        else:
            pass
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.pre_lookahead_len = int(encoder.pre_lookahead_layer.pre_lookahead_len)
        self.up_rate = int(encoder.up_layer.stride)
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.speaker_embedding_projection = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_dim, output_size)
        self.decoder = decoder

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens: torch.Tensor,
        token_lengths: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_token_lengths: torch.Tensor,
        prompt_mel: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        assert speech_tokens.shape[0] == prompt_tokens.shape[0], (
            f"flow batch size mismatch: speech_tokens={speech_tokens.shape[0]} "
            f"prompt_tokens={prompt_tokens.shape[0]}"
        )
        speaker_embeddings = F.normalize(speaker_embeddings, dim=1)
        speaker_embeddings = self.speaker_embedding_projection(speaker_embeddings)
        prompt_row_lengths = prompt_token_lengths.tolist()
        generated_row_lengths = token_lengths.tolist()
        combined_tokens = pad_sequence(
            [
                torch.cat(
                    [prompt_tokens[i, :prompt_length], speech_tokens[i, :token_length]]
                )
                for i, (prompt_length, token_length) in enumerate(
                    zip(prompt_row_lengths, generated_row_lengths, strict=True)
                )
            ],
            batch_first=True,
        )
        combined_token_lengths = prompt_token_lengths + token_lengths
        token_mask = (
            (~make_pad_mask(combined_token_lengths))
            .unsqueeze(-1)
            .to(speaker_embeddings)
        )
        embedded_tokens = (
            self.input_embedding(torch.clamp(combined_tokens, min=0)) * token_mask
        )
        hidden_states, _ = self.encoder.forward(embedded_tokens, combined_token_lengths)
        frame_mask = (
            ~make_pad_mask(
                combined_token_lengths * self.up_rate, hidden_states.shape[1]
            )
        ).to(hidden_states)
        hidden_states = self.encoder_proj(hidden_states) * frame_mask.unsqueeze(-1)
        mel_conditioning = torch.zeros_like(hidden_states)
        for i, prompt_length in enumerate(prompt_row_lengths):
            prompt_frames = prompt_length * self.up_rate
            mel_conditioning[i, :prompt_frames] = prompt_mel[i, :prompt_frames]
        mel_conditioning = mel_conditioning.transpose(1, 2).contiguous()
        packed_valid_frames = (
            2
            * sum(
                min(
                    (prompt_length + token_length) * self.up_rate,
                    hidden_states.shape[1],
                )
                for prompt_length, token_length in zip(
                    prompt_row_lengths, generated_row_lengths, strict=True
                )
            )
            if len(generated_row_lengths) > 1
            else None
        )
        predicted_mel = self.decoder.forward(
            mu=hidden_states.transpose(1, 2).contiguous(),
            mask=frame_mask.unsqueeze(1),
            speaker_embeddings=speaker_embeddings,
            mel_conditioning=mel_conditioning,
            n_timesteps=n_timesteps,
            packed_valid_frames=packed_valid_frames,
        )
        generated = [
            predicted_mel[
                i,
                :,
                prompt_length
                * self.up_rate : (prompt_length + token_length)
                * self.up_rate,
            ]
            for i, (prompt_length, token_length) in enumerate(
                zip(prompt_row_lengths, generated_row_lengths, strict=True)
            )
        ]
        return pad_sequence(
            [row.transpose(0, 1) for row in generated], batch_first=True
        ).transpose(1, 2)
