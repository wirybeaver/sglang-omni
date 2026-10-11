# SPDX-License-Identifier: Apache-2.0
"""Encode streaming windows at any request progress in one batch."""

from collections import defaultdict
from collections.abc import Sequence

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache

from sglang_omni.models.nemotron3_5_asr.cache import (
    NemotronBatchAttentionCache,
    NemotronBatchPaddingCache,
)
from sglang_omni.models.nemotron3_5_asr.encoder_state_pool import (
    EncoderStateSlot,
    NemotronEncoderStatePool,
    PooledAttentionCache,
    PooledPaddingCache,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderCausalConvPaddingCache,
)


def encode_streaming_batch(
    model: Nemotron3_5AsrForRNNT,
    input_features: Sequence[torch.Tensor],
    prompt_ids: torch.Tensor,
    *,
    attention_caches: Sequence[DynamicCache],
    padding_caches: Sequence[NemotronAsrStreamingEncoderCausalConvPaddingCache],
    num_lookahead_tokens: int,
) -> torch.Tensor:
    assert not model.training
    encoder = model.encoder
    groups: dict[int, list[int]] = defaultdict(list)
    for index, features in enumerate(input_features):
        groups[features.shape[1]].append(index)

    # note (Li Gang): Both mel lengths produce equally long encoder windows.
    subsampled: dict[int, torch.Tensor] = {}
    for indices in groups.values():
        features = torch.cat([input_features[index] for index in indices], dim=0)
        padding_cache = NemotronBatchPaddingCache(
            [padding_caches[index] for index in indices]
        )
        hidden_states = encoder.subsampling(
            features, attention_mask=None, padding_cache=padding_cache
        )
        subsampled.update(zip(indices, hidden_states.split(1), strict=True))
    hidden_states = torch.cat(
        [subsampled[index] for index in range(len(input_features))], dim=0
    )
    hidden_states *= encoder.input_scale

    attention_cache = NemotronBatchAttentionCache(attention_caches)
    padding_cache = NemotronBatchPaddingCache(padding_caches)
    sequence_length = hidden_states.shape[1]
    attention_mask = attention_cache.create_mask(
        sequence_length,
        hidden_states.device,
        model.config.encoder_config.sliding_window - 1,
        num_lookahead_tokens,
    )
    return encode_projected_frames(
        model, hidden_states, prompt_ids, attention_mask, attention_cache, padding_cache
    )


def encode_pooled_streaming_batch(
    model: Nemotron3_5AsrForRNNT,
    input_features: Sequence[torch.Tensor],
    prompt_ids: torch.Tensor,
    *,
    encoder_slots: Sequence[EncoderStateSlot],
    num_lookahead_tokens: int,
) -> torch.Tensor:
    """Encode mixed first/subsequent windows while updating persistent slots."""
    assert not model.training
    pool = encoder_slots[0].pool
    groups: dict[tuple[bool, int], list[int]] = defaultdict(list)
    for index, (features, slot) in enumerate(
        zip(input_features, encoder_slots, strict=True)
    ):
        groups[(slot.seen_frames == 0, features.shape[1])].append(index)
    encoded_by_row: dict[int, torch.Tensor] = {}
    for (is_first_chunk, _), indices in groups.items():
        features = torch.cat([input_features[index] for index in indices])
        slot_ids = torch.tensor(
            [encoder_slots[index].slot_id for index in indices],
            device=pool.layout.device,
        )
        encoded_frames = encode_pooled_windows(
            model,
            features,
            prompt_ids[indices],
            slot_ids,
            pool,
            is_first_chunk=is_first_chunk,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        encoded_by_row.update(zip(indices, encoded_frames.split(1), strict=True))
        for index in indices:
            encoder_slots[index].seen_frames += encoded_frames.shape[1]
    return torch.cat([encoded_by_row[index] for index in range(len(encoder_slots))])


def encode_pooled_windows(
    model: Nemotron3_5AsrForRNNT,
    input_features: torch.Tensor,
    prompt_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    pool: NemotronEncoderStatePool,
    *,
    is_first_chunk: bool,
    num_lookahead_tokens: int,
) -> torch.Tensor:
    """Encode equal-shaped windows and update pool state through tensor slot IDs."""
    padding_cache = PooledPaddingCache(
        pool=pool, slot_ids=slot_ids, is_first_chunk=is_first_chunk
    )
    hidden_states = model.encoder.subsampling(
        input_features, attention_mask=None, padding_cache=padding_cache
    )
    hidden_states *= model.encoder.input_scale
    attention_cache = PooledAttentionCache(pool=pool, slot_ids=slot_ids)
    chunk_frames = hidden_states.shape[1]
    encoded_frames = encode_projected_frames(
        model,
        hidden_states,
        prompt_ids,
        attention_cache.create_mask(chunk_frames, num_lookahead_tokens),
        attention_cache,
        padding_cache,
    )
    pool.seen_frames.index_add_(0, slot_ids, torch.full_like(slot_ids, chunk_frames))
    return encoded_frames


def encode_projected_frames(
    model: Nemotron3_5AsrForRNNT,
    hidden_states: torch.Tensor,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_cache: NemotronBatchAttentionCache | PooledAttentionCache,
    padding_cache: NemotronBatchPaddingCache | PooledPaddingCache,
) -> torch.Tensor:
    """Share Conformer and locale projection between dynamic and pooled state."""
    position_embeddings = model.encoder.encode_positions(
        hidden_states, cached_frames=attention_mask.shape[-1] - hidden_states.shape[1]
    )
    all_masked_rows = torch.all(~attention_mask, dim=-1)
    for encoder_layer in model.encoder.layers:
        hidden_states = encoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            all_masked_rows=all_masked_rows,
            position_embeddings=position_embeddings,
            past_key_values=attention_cache,
            padding_cache=padding_cache,
            use_cache=True,
        )

    one_hot = F.one_hot(
        prompt_ids.to(hidden_states.device), num_classes=model.config.num_prompts
    ).to(hidden_states.dtype)
    one_hot = one_hot[:, None, :].expand(-1, hidden_states.shape[1], -1)
    fused = model.prompt_projector(torch.cat([hidden_states, one_hot], dim=-1))
    return model.encoder_projector(fused)
