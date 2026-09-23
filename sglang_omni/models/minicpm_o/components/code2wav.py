# SPDX-License-Identifier: Apache-2.0
"""Vocode MiniCPM-o codec tokens with a cached speaker reference."""

from __future__ import annotations

import io
import logging
import os
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.weight_loader import resolve_dtype, resolve_model_path
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key

FLOW_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

OUTPUT_SAMPLE_RATE = 24000
CODEC_TOKEN_RATE = 25
SAMPLES_PER_CODEC_TOKEN = OUTPUT_SAMPLE_RATE // CODEC_TOKEN_RATE

logger = logging.getLogger(__name__)


def plan_flow_groups(
    frame_lengths: Sequence[int],
    *,
    max_gap_frames: int,
    pad_budget_percent: float,
) -> list[list[int]]:
    """Partition length-sorted rows into the fewest groups within a pad budget."""
    if max_gap_frames < 0:
        raise ValueError("Flow merge max gap must be non-negative")
    if pad_budget_percent < 0:
        raise ValueError("Flow merge pad budget must be non-negative")
    if any(length <= 0 for length in frame_lengths):
        raise ValueError("Flow frame lengths must be positive")

    ordered = tuple(
        sorted(enumerate(frame_lengths), key=lambda item: (item[1], item[0]))
    )
    if not ordered:
        return []
    baseline_work = sum(frame_lengths)
    row_count = len(ordered)

    @lru_cache(maxsize=None)
    def optimal_partition(
        start: int, groups_left: int, max_group_gap: int
    ) -> tuple[int, int, tuple[int, ...]] | None:
        if groups_left == 0:
            return (0, max_group_gap, ()) if start == row_count else None
        if row_count - start < groups_left:
            return None

        best: tuple[int, int, tuple[int, ...]] | None = None
        shortest = ordered[start][1]
        end_limit = row_count - groups_left + 1
        for end in range(start + 1, end_limit + 1):
            longest = ordered[end - 1][1]
            gap = longest - shortest
            if gap > max_gap_frames:
                break
            suffix = optimal_partition(end, groups_left - 1, max(max_group_gap, gap))
            if suffix is None:
                continue
            candidate = (
                (end - start) * longest + suffix[0],
                suffix[1],
                (end,) + suffix[2],
            )
            if best is None or candidate < best:
                best = candidate
        return best

    for group_count in range(1, row_count + 1):
        plan = optimal_partition(0, group_count, 0)
        if (
            plan is not None
            and (plan[0] / baseline_work - 1) * 100 <= pad_budget_percent + 1e-9
        ):
            start = 0
            groups = []
            for end in plan[2]:
                groups.append([index for index, _ in ordered[start:end]])
                start = end
            return groups
    raise AssertionError("positive Flow frame lengths must have a feasible partition")


class MiniCPMOCode2Wav(nn.Module):
    """Convert codec tokens into a float32 waveform with Token2wav."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
        n_timesteps: int = 10,
        prompt_wav: str | None = None,
        flow_merge_max_gap_frames: int = 384,
        flow_merge_pad_budget_percent: float = 25.0,
    ) -> None:
        super().__init__()
        from sglang_omni.models.minicpm_o.components.token2wav.vocoder import Token2Wav

        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"Token2wav requires a CUDA device, got {device}")
        self.device_context = torch.cuda.device(dev.index or 0)

        model_dir = str(resolve_model_path(model_path))
        asset_dir = os.path.join(model_dir, "assets", "token2wav")
        if not os.path.isdir(asset_dir):
            raise FileNotFoundError(
                f"token2wav assets not found at {asset_dir}; copy the "
                "checkpoint's assets/token2wav directory next to the weights"
            )
        if dtype is None:
            torch_dtype = torch.float32
        elif isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        else:
            torch_dtype = resolve_dtype(dtype)
        if torch_dtype not in FLOW_DTYPES:
            raise ValueError(
                f"Code2Wav dtype must be float32, float16, or bfloat16, got {dtype}"
            )
        with self.device_context:
            self.token2wav = Token2Wav(
                Path(asset_dir), device=dev, dtype=torch_dtype, n_timesteps=n_timesteps
            )

        if prompt_wav is None:
            default_wav = os.path.join(model_dir, "assets", "HT_ref_audio.wav")
            prompt_wav = default_wav if os.path.isfile(default_wav) else None
        self.default_prompt_wav = prompt_wav
        self.flow_merge_max_gap_frames = flow_merge_max_gap_frames
        self.flow_merge_pad_budget_percent = flow_merge_pad_budget_percent
        # Keyed by reference so a mixed-reference batch never thrashes one slot.
        self.prompt_cache: OrderedDict[str, tuple] = OrderedDict()
        self.prompt_cache_capacity = 32
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.eval()

    @torch.inference_mode()
    def forward(
        self,
        *,
        codec_tokens: torch.Tensor,
        prompt_wav: str | bytes | None = None,
        **_: object,
    ) -> dict[str, object]:
        """Vocode EOS-stripped codec tokens using the supplied or default reference."""
        tokens = codec_tokens.reshape(-1).tolist()
        if not tokens:
            waveform = np.zeros(0, dtype=np.float32)
        else:
            with self.device_context:
                reference = self.resolve_prompt_wav(prompt_wav)
                waveform = self.vocode([tokens], reference)[0]
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def resolve_prompt_wav(self, prompt_wav: str | bytes | None) -> str | bytes:
        if prompt_wav is not None:
            resolved = prompt_wav
        elif self.default_prompt_wav is None:
            raise ValueError("No speaker-reference audio supplied or default available")
        else:
            resolved = self.default_prompt_wav
        return resolved

    @staticmethod
    def prompt_key(prompt_wav: str | bytes) -> str:
        if isinstance(prompt_wav, bytes):
            return f"bytes:{hash_bytes(prompt_wav)}"
        return reference_path_cache_key(prompt_wav) or f"path:{prompt_wav}"

    def speaker_prompt(
        self, prompt_wav: str | bytes | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_wav = self.resolve_prompt_wav(prompt_wav)
        prompt_key = self.prompt_key(prompt_wav)
        cached = self.prompt_cache.get(prompt_key)
        if cached is not None:
            self.prompt_cache.move_to_end(prompt_key)
            return cached
        # Bytes references decode in memory; they are not spilled to a temp file.
        source = io.BytesIO(prompt_wav) if isinstance(prompt_wav, bytes) else prompt_wav
        prompt = self.token2wav.prepare_prompt(source)
        self.prompt_cache[prompt_key] = prompt
        if len(self.prompt_cache) > self.prompt_cache_capacity:
            self.prompt_cache.popitem(last=False)
        return prompt

    def vocode(
        self,
        token_sequences: Sequence[Sequence[int]],
        prompt_wav: str | bytes | Sequence[str | bytes] | None = None,
    ) -> list[np.ndarray]:
        """Batch flow across references and preserve each HiFT sequence boundary."""
        if not token_sequences:
            return []
        if any(len(tokens) == 0 for tokens in token_sequences):
            raise ValueError("codec token sequences must be non-empty")

        batch_size = len(token_sequences)
        if isinstance(prompt_wav, (list, tuple)):
            if len(prompt_wav) != batch_size:
                raise ValueError(
                    f"prompt_wav count {len(prompt_wav)} does not match "
                    f"token sequence count {batch_size}"
                )
            references = list(prompt_wav)
        else:
            references = [prompt_wav] * batch_size

        token_lens = [len(tokens) for tokens in token_sequences]
        device = self.token2wav.device
        prompts = [self.speaker_prompt(reference) for reference in references]
        up_rate = self.token2wav.flow.up_rate
        frame_lengths = [
            (int(prompt[1][0]) + token_len) * up_rate
            for prompt, token_len in zip(prompts, token_lens, strict=True)
        ]
        flow_groups = plan_flow_groups(
            frame_lengths,
            max_gap_frames=self.flow_merge_max_gap_frames,
            pad_budget_percent=self.flow_merge_pad_budget_percent,
        )
        grouped_work = sum(
            len(indices) * max(frame_lengths[index] for index in indices)
            for indices in flow_groups
        )
        logger.info(
            f"MiniCPM-o Flow grouped rows={batch_size} groups={len(flow_groups)} "
            f"frames={frame_lengths} padding={(grouped_work / sum(frame_lengths) - 1) * 100:.1f}%"
        )

        mel_rows: dict[int, torch.Tensor] = {}
        for indices in flow_groups:
            speech_tokens = pad_sequence(
                [
                    torch.tensor(
                        token_sequences[index], dtype=torch.int32, device=device
                    )
                    for index in indices
                ],
                batch_first=True,
            )
            speech_tokens_lens = torch.tensor(
                [token_lens[index] for index in indices],
                dtype=torch.int32,
                device=device,
            )
            group_prompts = [prompts[index] for index in indices]
            if len({id(prompt) for prompt in group_prompts}) == 1:
                (
                    prompt_speech_tokens,
                    prompt_speech_tokens_lens,
                    speaker_embedding,
                    prompt_mels,
                ) = group_prompts[0]
                group_size = len(indices)
                prompt_speech_tokens = prompt_speech_tokens.expand(
                    group_size, -1
                ).contiguous()
                prompt_speech_tokens_lens = prompt_speech_tokens_lens.expand(
                    group_size
                ).contiguous()
                speaker_embedding = speaker_embedding.expand(
                    group_size, -1
                ).contiguous()
                prompt_mels = prompt_mels.expand(group_size, -1, -1).contiguous()
            else:
                token_width = max(prompt[0].numel() for prompt in group_prompts)
                prompt_speech_tokens = torch.cat(
                    [
                        torch.nn.functional.pad(
                            prompt[0].reshape(1, -1),
                            (0, token_width - prompt[0].numel()),
                        )
                        for prompt in group_prompts
                    ]
                )
                prompt_speech_tokens_lens = torch.cat(
                    [prompt[1] for prompt in group_prompts]
                )
                speaker_embedding = torch.cat([prompt[2] for prompt in group_prompts])
                mel_frames = max(prompt[3].shape[1] for prompt in group_prompts)
                prompt_mels = torch.cat(
                    [
                        torch.nn.functional.pad(
                            prompt[3], (0, 0, 0, mel_frames - prompt[3].shape[1])
                        )
                        for prompt in group_prompts
                    ]
                )

            with torch.amp.autocast(
                "cuda",
                dtype=self.token2wav.dtype,
                enabled=self.token2wav.dtype != torch.float32,
            ):
                group_mel = self.token2wav.flow.inference(
                    speech_tokens,
                    speech_tokens_lens,
                    prompt_speech_tokens,
                    prompt_speech_tokens_lens,
                    prompt_mels,
                    speaker_embedding,
                    self.token2wav.n_timesteps,
                )
            for row, index in enumerate(indices):
                mel_rows[index] = group_mel[row, :, : token_lens[index] * up_rate]

        length_groups: dict[int, list[int]] = {}
        for idx, token_len in enumerate(token_lens):
            length_groups.setdefault(token_len, []).append(idx)

        waveform_rows: dict[int, torch.Tensor] = {}
        # note (MayDomine): padding changes HiFT's noncausal convolution boundaries.
        for token_len, indices in length_groups.items():
            speech_feat = torch.stack([mel_rows[index] for index in indices]).float()
            wav, _ = self.token2wav.hift(speech_feat=speech_feat)
            for row, idx in enumerate(indices):
                waveform_rows[idx] = wav[row].reshape(-1)[
                    : token_len * SAMPLES_PER_CODEC_TOKEN
                ]
        wav = (
            pad_sequence(
                [waveform_rows[idx] for idx in range(batch_size)], batch_first=True
            )
            .float()
            .cpu()
        )
        return [
            wav[idx, : token_len * SAMPLES_PER_CODEC_TOKEN].numpy()
            for idx, token_len in enumerate(token_lens)
        ]
