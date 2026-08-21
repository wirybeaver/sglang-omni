# SPDX-License-Identifier: Apache-2.0
"""Functional and sustained-load validation for consumer ASR servers.

The harness exercises Qwen3-ASR, Fun-ASR-Nano, or Whisper through the public
OpenAI-compatible audio endpoints and writes one schema-v2 evidence artifact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.asr_stability import run_validation
from benchmarks.tasks.asr import (
    FUN_ASR_MODEL_PATH,
    OMNI_WHISPER_MODEL_PATH,
    QWEN3_ASR_MODEL_PATH,
)

# Retain the documented async entry point while keeping this executable thin.
main_async = run_validation


def _number(value: str, *, integer: bool, minimum: float, inclusive: bool):
    try:
        parsed = int(value) if integer else float(value)
    except ValueError as exc:
        kind = "integer" if integer else "number"
        raise argparse.ArgumentTypeError(f"expected a {kind}, got {value!r}") from exc
    valid = math.isfinite(parsed) and (
        parsed >= minimum if inclusive else parsed > minimum
    )
    if not valid:
        comparison = "at least" if inclusive else "greater than"
        raise argparse.ArgumentTypeError(f"value must be {comparison} {minimum:g}")
    return parsed


def _positive_int(value: str) -> int:
    return _number(value, integer=True, minimum=0, inclusive=False)


def _positive_float(value: str) -> float:
    return _number(value, integer=False, minimum=0, inclusive=False)


def _nonnegative_float(value: str) -> float:
    return _number(value, integer=False, minimum=0, inclusive=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--model-path",
        default=FUN_ASR_MODEL_PATH,
        help=(
            "Served ASR model id. Supported consumer profiles: "
            f"{QWEN3_ASR_MODEL_PATH}, {FUN_ASR_MODEL_PATH}, and "
            f"{OMNI_WHISPER_MODEL_PATH}."
        ),
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Exact revision loaded by the server; never inferred.",
    )
    parser.add_argument("--meta", default=DATASETS["seedtts"])
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Exact dataset revision; canonical SeedTTS uses a pinned default.",
    )
    parser.add_argument("--duration-s", type=_positive_float, default=1800.0)
    parser.add_argument("--concurrencies", default="1,4,8,16")
    parser.add_argument("--samples-per-language", type=_positive_int, default=20)
    parser.add_argument("--request-timeout-s", type=_positive_float, default=60.0)
    parser.add_argument("--max-audio-duration-s", type=_positive_float, default=30.0)
    parser.add_argument(
        "--check-audio-boundary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Check just-under/just-over max audio duration. Defaults on for "
            "Fun-ASR; disable for endpoints that chunk long audio."
        ),
    )
    parser.add_argument(
        "--include-translation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Exercise /v1/audio/translations; defaults off.",
    )
    parser.add_argument(
        "--translation-source-language",
        default="zh",
        help="Required source-language hint used for translation checks.",
    )
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--mm-attention-backend", default=None)
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--torch-compile", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument(
        "--min-free-memory-mib",
        type=_nonnegative_float,
        default=2048.0,
    )
    parser.add_argument(
        "--max-retained-memory-mib",
        type=_nonnegative_float,
        default=256.0,
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--monitor-interval-s", type=_positive_float, default=0.2)
    parser.add_argument(
        "--gpu-process-pid",
        dest="gpu_process_pids",
        type=_positive_int,
        action="append",
        help=(
            "NVML PID to include in GPU-process metrics; repeat for multiple "
            "GPU processes."
        ),
    )
    parser.add_argument("--chaos-interval-s", type=_positive_float, default=30.0)
    parser.add_argument("--cooldown-s", type=_nonnegative_float, default=5.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--launch-command",
        default=os.environ.get("SGLANG_OMNI_BENCHMARK_LAUNCH_COMMAND"),
    )
    parser.add_argument("--output", default="asr_stability_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_validation(args))
    output = Path(os.path.abspath(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"passed": result["passed"], "output": str(output)}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
