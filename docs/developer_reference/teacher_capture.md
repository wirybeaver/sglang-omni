# Teacher capture interface

Omni exposes inference-only teacher capture. Dataset iteration, training masks, persistent feature caches, rejection journals and resume cursors belong to the training consumer. There is no standalone dataset collector in Omni.

## Model adapter

A factory implements the `CaptureBackend` interface in `sglang_omni.capture.backend`. It accepts a model configuration and returns a backend with `capture(request)`, `close()` and immutable provenance. `CaptureInput.inputs` and `params` carry model-specific JSON values; the generic worker does not interpret audio paths, images or prompt formats. The backend owns preprocessing, frozen teacher generation, replay and inference-resource cleanup. A result contains either a complete `CapturedSequence` or an explicit generation rejection. Unexpected execution failures terminate the worker and are not successful samples.

`CapturedSequence` contains expanded input IDs, generated-token spans, ordered decoder layer IDs, concatenated auxiliary states and post-norm final states before LM-head scaling. Generated spans may be disjoint. The contract contains no training loss mask, EOS constant, model width, fixed layer count or draft algorithm. Consumers decide which generated positions to supervise. Tensors have no batch dimension, retain their native dtype and own independent CPU storage.

SGLang-backed adapters set `ARRequestData.teacher_capture` on replay requests. The shared ModelRunner requests FULL capture and splits the prefill result using actual request lengths. It rejects cached prefixes, incomplete rows and mixed capture/non-capture batches. An adapter whose runner overrides these hooks must explicitly compose the capture behavior. Each model still needs a validated capture setter, layer semantics and final-state implementation; adding this interface does not automatically enable every model.

## Isolated inference worker

The `sglang_omni.capture.worker` module loads the configured factory and accepts sequential JSON requests on stdin. Stdout is reserved for a versioned readiness message and one response per request; inference logs go to stderr. A response acknowledges completion of a CPU tensor artifact at the path supplied by the caller. The caller must consume that artifact before issuing its next request. This is a local, shared-filesystem transport, not a remote HTTP endpoint.

The worker owns only the current inference result. The training consumer owns the temporary artifact directory, worker process and durable sample publication. Closing stdin closes the backend. Fatal execution errors produce a nonzero process exit. Separate Python executables let training and inference retain independent uv dependencies.

## Ming ASR adapter

The first adapter is `sglang_omni.models.ming3_asr.capture:create_capture_backend`. Its input requires `audio_path`; optional parameters are `language` and `prompt`. Capture layers come from configuration and are validated against the checkpoint. Ming preprocessing, audio embedding injection and the native decoder are reused. A supplied language prefix stays in the prompt; a teacher-generated language prefix stays in the generated span.

The adapter requires greedy generation ending with checkpoint EOS, then replays the exact IDs with the same encoded audio. Replay uses one disposable completion token to exercise normal scheduler cleanup; that token is excluded from captured rows. Eager execution, TP1, disabled overlap/radix cache and unchunked prefill are mandatory for this first implementation.

CPU tests establish request slicing, span retention, dtype and storage ownership. GPU numerical parity, native transcription regression, KDA isolation, training and export remain unvalidated. SGLang source `391f4e6f175177571d8e5471daa684d1fcaf9c99` is the inspected candidate API; stock 0.5.23 is not a validated capture runtime. The adapter fails if the final-state output field is unavailable.
