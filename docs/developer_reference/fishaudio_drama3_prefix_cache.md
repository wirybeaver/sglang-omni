# Fish Audio Drama 3 prefix-cache investigation

Investigated on 2026-10-14 against SGLang-Omni baseline
`65211c90530e02feff68e75135b729f7a5538a86` and its pinned SGLang 0.5.21.
This note describes the opt-in implementation and design requirements for issue
[#2797](https://github.com/sgl-project/sglang-omni/issues/2797). End-to-end performance
and full real-checkpoint quality qualification remain incomplete.

## Local implementation

The local change enables an opt-in path with
`--tts_engine.engine.disable_radix_cache false`. The builder selects SGLang's
`extra_buffer` Mamba strategy and registers Drama's GDN backend with the
linear-attention model registry. The shared cache factory constructs
`UnifiedRadixCache` with both `FULL` and `MAMBA` components for hybrid pools.
The default remains uncached.

The single-checkpoint tracking path selects the current mapped slot directly,
avoiding a repeated host upload of zero column indices. Other buffer layouts and
explicit tracking positions retain the original selection path. A one-token
prefill graph bucket covers cache hits that leave only the final prompt row.

Each reusable prompt identity hashes the complete pair of frames at `p` and
`p + 1`. The final prompt identity is unique within the engine, keeping the
generated suffix private: scheduler output IDs only contain semantic tokens,
so they cannot identify generated residual codebooks. Real frame rows remain
the input to the model and the RAS history. A hit stages only the uncached
suffix while retaining the scheduler's absolute positions.

Reuse is limited to the recurrent checkpoints SGLang actually stores, rather
than every matching token boundary. Short prompts can have no checkpoint;
a newly encountered branch can require another prefill before its checkpoint
is available. Checkpoint buffers also consume GPU memory, so the reduction in
prefill work must be evaluated alongside memory capacity and concurrent load.
This change does not cache reference-encoder outputs.

The regression scenario in `test_graph_engine.py` compares cached reference and
optimized engines across a decode checkpoint boundary, checks a one-token graph
hit, changes a residual codebook at a cache boundary, and checks reuse after
abort. It also retains a short cached-versus-uncached output check. Run it with:

```bash
uv run python -m pytest tests/unit_test/fishaudio_drama3/test_graph_engine.py \
  -k prefix_cache_reuses_prompt
```

The scenario uses the existing synthetic GPU checkpoint to exercise the runtime.
It is not a substitute for output-quality and latency measurements with the real
Drama 3 checkpoint. Long cached and uncached trajectories can diverge at BF16
logit ties; matching the optimized and reference cache paths does not establish
cached-versus-uncached bitwise equivalence.

## Which prefix?

The common principle is reusing a computation whose entire input dependency
is unchanged. The existing CosyVoice3 streaming cache and Drama 3 prompt
caching apply that principle to different parts of the pipeline.

| Cache | Reused work | Lifetime and identity |
| --- | --- | --- |
| CosyVoice3 AR KV | Causal language-model attention over text and speech embeddings | Within decoding; CUDA serving can also use scheduler radix matching of actual embedding identities |
| CosyVoice3 streaming Flow prefix | DiT attention and causal convolution work for completed mel chunks | Within one streaming request, separately for each Euler step and CFG half |
| Drama 3 prompt prefix | Hybrid backbone and codebook-tower prefill | Across requests sharing identical multimodal frame history, with a matching recurrent-state checkpoint |
| Encoded reference audio | Audio encoder outputs | Across requests with the same reference; independent of AR prefix caching |

In official CosyVoice code, `Qwen2Encoder.forward_one_step` passes
`past_key_values`; `inference_bistream` starts a new `cache = None` and carries it
through text/speech increments. That is request continuation, not evidence of
automatic cross-request reuse. See the
[official LLM implementation](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/llm/llm.py).

SGLang-Omni's separate Flow optimization is explicit in
[`prefix_cache.py`](../../sglang_omni/models/fun_cosyvoice3/prefix_cache.py):
`PrefixKVPool` stores K/V per Euler step and layer; `PrefixCacheRow` also stores
positional-convolution context. Complete causal chunks are immutable because
later chunks cannot influence them, causal convolutions look left, and each
hop uses the same noise. Incomplete chunks are recomputed. Each streaming
request owns its CFG-twin rows, with full-history fallback when the pool is
full and release on termination; see `hop_batch_with_prefix` in
[`streaming_vocoder.py`](../../sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py).
This pool cannot simply be attached to Drama's AR engine.

For CosyVoice3's AR path,
[`build_embedding_cache_key_ids`](../../sglang_omni/models/fun_cosyvoice3/request_builders.py)
hashes every actual embedding row. Its
[`engine_builder.py`](../../sglang_omni/models/fun_cosyvoice3/engine_builder.py)
disables radix caching on MLX/MPS but does not disable it in the CUDA defaults.
That identity lesson is relevant to Drama even though its hybrid state and
tower require additional work.

## Model and dependencies

The [official model card](https://huggingface.co/fishaudio/drama-3) describes a
35B MoE model with 3B active parameters and a causal 10-codebook MTP decoder.
The [checkpoint configuration](https://huggingface.co/fishaudio/drama-3/blob/023b58c/config.json)
has 40 backbone layers: 30 linear-attention layers and 10 full-attention
layers, in repeating groups of three linear layers followed by one full
layer. Hidden width is 2048. Linear attention uses a convolution width of
4 and float32 recurrent state. The tower has four shared enrichment layers
and nine residual-codebook layers, all full attention. The checkpoint places
one residual codebook in the backbone input.

The local implementation maps the backbone onto SGLang's Qwen3.5 MoE model.
`Drama3TextConfig.full_attention_layer_ids` appends all 13 tower layer IDs to
the backbone's 10 full-attention layer IDs, so their K/V lives in the same
paged pool. The 30 linear layers retain separate recurrent state. Sources:
[`sglang_model.py`](../../sglang_omni/models/fishaudio_drama3/sglang_model.py),
[`text_config.py`](../../sglang_omni/models/fishaudio_drama3/text_config.py), and
the pool checks in
[`model_runner.py`](../../sglang_omni/models/fishaudio_drama3/model_runner.py).

A prompt frame is `[main_token, code_0, ..., code_9]`. For audio rows the main
token encodes semantic code 0. Nevertheless, identical main tokens do not
mean identical model inputs: `embed_backbone` also consumes code 1 for this
checkpoint, and `shared_input` consumes codes 2 through 9. A shared cache must
identify the entire relevant frame history, not just semantic IDs or the
reference transcript. See
[`audio_decoder.py`](../../sglang_omni/models/fishaudio_drama3/audio_decoder.py).

The prompt template puts reference transcripts and audio frames before the
target text. Requests with the same ordered references can therefore share
a substantial leading segment even when the target text differs. Changing
the speaker tags, reference ordering, transcript, or audio codes changes the
prefix identity. See
[`build_prompt_rows`](../../sglang_omni/models/fishaudio_drama3/prompt.py).

## The tower's one-frame dependency

The tower's cache at position `p` is conditioned on backbone hidden state
`p`, input frame `p`, **and frame `p + 1`**. `AudioDecoder.prefill` receives
`next_frames`, feeding the next frame's semantic code to the shared layers
and its residual codes to successive residual layers.
[`next_frame_labels`](../../sglang_omni/models/fishaudio_drama3/model_inputs.py)
constructs these shifted labels. The last row receives a temporary EOS label,
then the first-frame sampling pass overwrites that tower slot.

Consequently, if two requests share exactly `N` input rows and diverge at row
`N`, the tower cache at row `N - 1` is not necessarily reusable. A valid design
must either match the next-frame dependency as part of each cache position's
identity, or recompute that boundary row with the new next frame. Reusing
backbone K/V alone while silently retaining the old tower boundary is wrong.
This is an inference from the local data flow above, independent of kernel
choice.

The boundary rule must also apply when caching generated history, handling
request retraction, and inserting a cache entry after completion. The cache
must not publish the last tower slot under an identity that omits its sampled
next frame.

## Safe integration requirements

The baseline disables and rejects radix caching in
[`Drama3EngineBuilder`](../../sglang_omni/models/fishaudio_drama3/engine_builder.py).
It also requires one complete prefill, while
[`extend_frame_rows`](../../sglang_omni/models/fishaudio_drama3/decode_state.py)
rejects any suffix-only extend. Removing the configuration guard alone cannot
enable correct reuse.

1. Use complete multimodal identity, including the tower's next-frame
   dependency. Keep vocabulary IDs used for inference separate from any
   scheduler-only cache identity. A model-local cache is naturally scoped to
   its loaded checkpoint; persistent or shared caches also need a model and
   execution-configuration namespace.
2. Restore full-attention K/V for both backbone and tower, plus the Gated
   DeltaNet temporal state and convolution window at the **same prefix
   boundary**. SGLang's
   [GDN backend](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/layers/attention/linear/gdn_backend.py)
   consumes and mutates both `conv` and `temporal` state. A final recurrent
   state cannot be sliced backward like token K/V.
3. Reuse SGLang's hybrid cache ownership where possible. Its
   [Mamba component](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/unified_cache/components/mamba.py)
   matches checkpoints and establishes a per-request copy on write; two
   continuations must not mutate the same recurrent-state slot. The available
   checkpoint boundary may be shorter than the full-KV match.
4. Stage only uncached suffix frames while preserving absolute positions and
   full-history repetition-aware sampling state. A cache hit must not turn
   the first new frame into an attention reset or erase the request's RAS
   history. Preserve uncached retraction and normal finish/abort cleanup.
5. Audit eager prefill, breakable prefill graphs, first-frame graphs, fused
   tower decode, eviction, and request-slot reuse. Their tensor shapes and
   lifetime assumptions were established with radix caching disabled.

For numerical validation, use the real checkpoint and hardware: compare
uncached versus cache-hit suffix logits, generated codebooks, and decoded
audio under controlled sampling. Include identical text with different
residual reference codes, a shared prefix with a different next frame, and
two continuations branching from the same recurrent checkpoint. CPU tests
can establish identity, suffix staging, and lifecycle decisions, but cannot
establish numerical parity of GDN or attention kernels.

### Runtime precision contract

The opt-in prefix-cache implementation needs two SGLang runtime changes:

- `TritonAttnBackend.use_unified_prefill` selects the unified full-attention
  prefill kernel without changing decode splitting or lean attention. The
  Drama 3 builder enables it before graph capture when prefix caching is on.
- The Triton GDN prefill path must forward `track_state` and `track_chunk_idx`
  to the existing FP32 snapshot kernel and restore that snapshot. Copying the
  BF16 intermediate chunk state into an FP32 pool does not recover its lost
  precision.

Stock SGLang 0.5.21 does not provide this complete contract. Installing only
the Omni changes is insufficient; the matching SGLang changes are required
before enabling prefix caching. Cache-disabled execution keeps its existing
attention policy. Numerical consistency does not replace speech-quality or
performance qualification.

For performance validation, separately report reference encoding time,
prefill time, hit length, TTFA, throughput, and cache memory. Compare cold
requests, repeated references with different text, and repeated complete
prompts. Do not attribute savings from reference encoding or the vocoder to
AR prefix reuse. Keep run-specific measurements and their raw evidence separate
from this design note.

## Prompt-only prototype

The opt-in factory option
`--tts_engine.factory.prefix_cache_prompt_only true` requires
`--tts_engine.engine.disable_radix_cache false` and the paired runtime changes.
Its default is false, retaining the existing cache path for comparison.
The [paired SGLang 0.5.21 patch](fishaudio_drama3_sglang_0_5_21.patch)
contains both the precision fixes and prompt-only runtime changes. Apply it to
the parent directory of the installed sglang package in an isolated environment;
it does not change the dependency pin or apply itself during server startup.

The builder fixes the request pool's mamba_cache_prompt_only policy before
CUDA graph capture. Prefill still records the valid FP32 checkpoint and publishes
it through the existing cache path. The existing prompt-only request policy then
blocks generated-tail inserts. Decode continues updating live KV and recurrent
state, while skipping checkpoint-index selection, mask construction/upload,
checkpoint saving and checkpoint-boundary bookkeeping. Decode graph capture omits
the tracking slots and save path; prefill graphs retain their tracking inputs.
This policy is fixed for the engine lifetime and cannot be toggled during serving.

The prototype keeps tracking-buffer allocation and the existing finish/abort
release path. It does not claim a memory-capacity improvement. Retraction still
rebuilds the complete uncached history and must be qualified on hardware.

The paired SGLang changes cover the request pool, decode preparation, result
processing, hybrid attention tracking and decode graph capture. The CPU boundary
regression exercises repeated decode preparation with stale prefill tracking
metadata and verifies that the published checkpoint remains untouched. The
existing GPU cache regression now compares prompt-only against ordinary caching
and uncached generation, including a decode boundary, one-token suffix, changed
acoustic input and abort/reuse. The prototype has passed a small end-to-end
quality screen and completed one performance A/B pair. The experiment was stopped without an observed TTFA gain.
A complete performance schedule, exact output parity, and retraction/eviction
qualification remain unverified.
