# Drama3 tested source snapshot, 2026-10-26

This branch preserves the model/server/test source used by the successful H200 c16 experiment. It is based on the original request-seed foundation ca6fb6b6f, before the later publication rebase. The server and test files were checked against every file in the frozen source inventory. The client-only deferred-WAV modification is preserved as the original patch in [source-and-protocol.json](source-and-protocol.json), with hashes, launch script, environment and per-visit mean results. Applying that patch reconstructs the frozen application file contents; the additional snapshot metadata directory is outside the original inventory.

The client overlay queues retained PCM during generation and writes WAVs only after the timed runner and client wall clocks stop. Its caller must then invoke flush_deferred_streaming_audio. It was used by the experiment harness; it is not automatically invoked by the stock benchmark CLI. These branches publish the tested source and protocol, rather than a standalone deployment of the archived Modal controller.

## Source pairs

| Experiment/arm | Omni branch | SGLang branch and commit | Prefix cache |
| --- | --- | --- | --- |
| GDN A | bench/drama3-gdn-abba-20261026 | bench/drama3-base-20261026, 9f6678aa | Off |
| GDN B | bench/drama3-gdn-abba-20261026 | bench/drama3-gdn-20261026, 31913fac | Off |
| Prefix A | bench/drama3-gdn-abba-20261026 | bench/drama3-gdn-20261026, 31913fac | Off |
| Prefix B | bench/drama3-gdn-prefix-abba-20261026 | bench/drama3-gdn-prefix-20261026, dae7917ffd | Prompt-only |

The SGLang branches are in wirybeaver/sglang; the Omni branches are in wirybeaver/sglang-omni. Load each TTS process from the matching source pair with an explicit PYTHONPATH. The old sglang 0.5.22 distribution metadata in the benchmark image does not identify the executed TTS source. The image used Torch 2.14.1+cu130, FlashInfer 0.7.2 and sglang-kernel 0.4.10; the complete installed-version record is in the JSON. The repository's original dependency pins alone do not reconstruct this environment.

## What the three PRs contain

[SGLang #46619](https://github.com/sgl-project/sglang/pull/46619) captures the GDN prefill kernel chain and supplies the FP32 snapshot plumbing. [Omni #3234](https://github.com/sgl-project/sglang-omni/pull/3234) adapts Omni to APIs already present in the SGLang base 9f6678aa: KV-configurator arguments, prefill read-end returns, held-row admission and scheduler interval/pause state. Those API changes were not introduced by #46619. Its attention-recorder adaptation is test instrumentation.

[Omni #3235](https://github.com/sgl-project/sglang-omni/pull/3235) stacks prompt-only prefix caching on #3234 and references the paired SGLang cache-policy changes. #3234 and #3235 were later rebased for review; they are not the exact historical benchmark commits. The tested foundation includes request-local seeds from [#3134](https://github.com/sgl-project/sglang-omni/pull/3134), shared by both arms.

## How #46619 was measured

Drama3 already selected breakable prefill graphs, registered its tower attention layers, and used RequestIdPrefillCudaGraphRunner. The generic Omni adapters made that existing path work with the newer SGLang base. Selecting Triton prefill allowed #46619 to capture the GDN layers inside that path.

Both A and B used the same Omni application, checkpoint, reference cohort, request/server seed 1234, sampling settings, compile-default 0 and cache-off policy. Only the SGLang source changed from 9f6678aa to 31913fac. This measures the incremental GDN change, not the combined effect of upgrading Omni, adding seeds or enabling prefix caching.

The workload was streaming c16 on one H200, all pipeline stages on GPU 0. Each A/B visit started a fresh server, including the adjacent B visits, then ran one canary, 128 substantial warmup requests, 256 prompt-prime requests, 256 shape warmup requests and immediately 256 timed requests. Early 32-pair WER/SIM scoring passed before performance. The measured order was A, B, B, A; the predeclared stop rule skipped the conditional tail. Middle B/A timed WAVs were retained and scored as 256 matched pairs after performance. Warmups and WAV persistence were outside timed wall time.

| Visit | Arm | Requests/s | Mean TTFA ms |
| --- | --- | ---: | ---: |
| 1 | A | 9.157 | 265.3 |
| 2 | B | 10.321 | 224.1 |
| 3 | B | 10.301 | 222.3 |
| 4 | A | 9.262 | 258.4 |

Mean request throughput improved 11.96%, and mean TTFA fell from 261.9 to 223.2 ms (14.77%). Timed requests had no errors or max-token hits. Main WER was A 1.6752% / B 1.3838%; speaker similarity was A 63.738 / B 63.882. Prefix caching was disabled and cached-token totals were zero in every timed leg.

The separate prefix experiment fixed GDN graph support and executed A, B, B, A, A, B. Mean request throughput improved 4.28%, with sampled WER/SIM gates passing. These are scoped c16 results without an independent A/A noise calibration; they are not fresh GPU qualifications of the rebased PR heads.
