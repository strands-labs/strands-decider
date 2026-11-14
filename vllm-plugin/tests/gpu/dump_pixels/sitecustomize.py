"""Test only, never for serving: on PYTHONPATH, every vLLM worker appends a digest of each image's pixel values, as the
Gemma 4 encoder receives them, to ``$STRANDS_DECIDER_PIXEL_DUMP``.<pid>.jsonl. ``tests/gpu/client.py pixels`` compares
them with the transformers reference's (``reference.py --pixels``), so that the pixels vLLM encodes are checked to be
the specification's, bit for bit."""

import hashlib
import json
import os

_PATH = os.environ.get("STRANDS_DECIDER_PIXEL_DUMP")


def _install():
    try:
        import torch
        from vllm.model_executor.models import gemma4_mm, gemma4_unified
    except Exception:
        return

    def digests(pv):
        f32 = pv.detach().to("cpu", torch.float32).contiguous()
        bf16 = f32.to(torch.bfloat16).view(torch.int16).contiguous()
        return {
            "dtype": str(pv.dtype),
            "float32": hashlib.sha256(f32.numpy().tobytes()).hexdigest()[:16],
            "bfloat16": hashlib.sha256(bf16.numpy().tobytes()).hexdigest()[:16],
            "shape": list(pv.shape),
        }

    for cls in (
        gemma4_mm.Gemma4ForConditionalGeneration,
        gemma4_unified.Gemma4UnifiedForConditionalGeneration,
    ):
        original = cls.__dict__.get("_process_image_input")
        if original is None or getattr(original, "_dump", False):
            continue

        def wrapped(self, image_input, _original=original):
            pv = image_input["pixel_values"]
            items = (
                list(pv) if isinstance(pv, (list, tuple)) else [pv[i] for i in range(pv.shape[0])]
            )
            with open(f"{_PATH}.{os.getpid()}.jsonl", "a") as f:
                for t in items:
                    f.write(json.dumps(digests(t)) + "\n")
            return _original(self, image_input)

        wrapped._dump = True
        cls._process_image_input = wrapped


if _PATH:
    _install()
