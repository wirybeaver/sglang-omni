# SPDX-License-Identifier: Apache-2.0
"""Serve sequential capture requests in an isolated inference environment."""

import argparse
import contextlib
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch

from sglang_omni.capture.backend import CaptureBackend, CaptureInput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factory", required=True, help="module:function creating a CaptureBackend"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    arguments = parser.parse_args()
    module_name, factory_name = arguments.factory.split(":", 1)
    protocol_output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        factory = getattr(importlib.import_module(module_name), factory_name)
        backend: CaptureBackend = factory(json.loads(arguments.config.read_text()))
        try:
            package_directory = Path(__file__).resolve().parents[1]
            source_digest = hashlib.sha256()
            for source_path in sorted(package_directory.rglob("*.py")):
                source_digest.update(
                    str(source_path.relative_to(package_directory)).encode()
                )
                source_digest.update(source_path.read_bytes())
            provenance = dict(
                backend.provenance, omni_source_sha256=source_digest.hexdigest()
            )
            print(
                json.dumps({"schema_version": 1, "provenance": provenance}),
                file=protocol_output,
                flush=True,
            )
            for line in sys.stdin:
                request = CaptureInput.model_validate_json(line)
                reply = backend.capture(request)
                assert (reply.sequence is None) == (reply.rejection is not None)
                metadata = dict(reply.metadata)
                if reply.sequence is not None:
                    sequence = reply.sequence
                    torch.save(
                        {
                            "input_ids": sequence.input_ids,
                            "auxiliary_hidden_states": sequence.auxiliary_hidden_states,
                            "final_hidden_states": sequence.final_hidden_states,
                        },
                        arguments.artifact_path,
                    )
                    metadata.update(
                        generated_spans=[
                            [span.start_token_index, span.end_token_index]
                            for span in sequence.generated_spans
                        ],
                        layer_ids=list(sequence.layer_ids),
                    )
                else:
                    pass
                print(
                    json.dumps(
                        {
                            "request_id": request.request_id,
                            "rejection": reply.rejection,
                            "metadata": metadata,
                        }
                    ),
                    file=protocol_output,
                    flush=True,
                )
        finally:
            backend.close()


if __name__ == "__main__":
    main()
else:
    pass
