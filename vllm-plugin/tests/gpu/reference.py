"""The transformers reference for a Strands Decider checkpoint, text and images, written apart from
the plugin.

    python reference.py CKPT --requests REQUESTS.jsonl --out OUT.jsonl
    python reference.py CKPT --rows ROWS.jsonl --out OUT.jsonl [--annotate]

``--requests``: System One requests, rendered with the core. Each image is decoded as v1 decodes it,
resized to the core's W x H with the core's ``resize_image`` and prepared by the checkpoint's image
processor with its resize off: ``Gemma4ImageProcessorPil``, or the 12B's
``Gemma4UnifiedImageProcessor``. ``--rows``: pre-rendered text prompts (``token_ids``,
``prefix_end``, ``label_token_ids``); ``--annotate`` writes each back with the reference's
``probabilities``.

The model is transformers' own class for the layout, in ``--dtype`` with SDPA, one prompt per
forward pass. The prefix_bidir mask replaces Gemma 4's masks (causal, windowed, and the "vision"
image mask) on every layer: token i sees j when j < S, or when i >= S. The readout is the core's: a
bf16 GEMM of the final-norm hidden state at the last token, then in float32 the soft-cap and the
temperature, then a softmax over the question's own codes.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch


def decode(data: bytes):
    """v1's decode: the first frame, EXIF orientation, transparency on white, RGB."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data), formats=("PNG", "JPEG", "WEBP", "GIF"))
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


class Reference:
    def __init__(self, ckpt: Path, device: str, dtype: str) -> None:
        import transformers
        from safetensors.torch import load_file

        self.device, self.dtype = device, getattr(torch, dtype)
        cfg = json.loads((ckpt / "config.json").read_text())
        dc = json.loads((ckpt / "decision_config.json").read_text())
        arch = cfg["architectures"][0]
        if arch == "Gemma4TextModel":
            self.model = transformers.Gemma4TextModel.from_pretrained(
                ckpt, dtype=self.dtype, attn_implementation="sdpa"
            )
            self.decoder, text = self.model, cfg
        else:
            config = transformers.AutoConfig.from_pretrained(ckpt)
            config.audio_config = None  # E2B's and E4B's audio encoder has no SDPA path
            self.model = getattr(transformers, arch).from_pretrained(
                ckpt, config=config, dtype=self.dtype, attn_implementation="sdpa"
            )
            self.decoder, text = self.model.model, cfg["text_config"]
            unified = arch == "Gemma4UnifiedForConditionalGeneration"
            processor = (
                transformers.Gemma4UnifiedImageProcessor
                if unified
                else transformers.Gemma4ImageProcessorPil
            )
            self.processor = processor.from_pretrained(ckpt)
        self.model.to(device).eval()
        self.readout = load_file(str(ckpt / "readout.safetensors"))["weight"].to(
            device, torch.bfloat16
        )
        self.softcap = float(dc.get("final_logit_softcapping") or text["final_logit_softcapping"])
        self.temperature = float(dc["temperature"])

    def pixels(self, images, counts):
        """(pixel_values, image_position_ids) for the decoded images, which the processor must count
        as the core does (``counts``)."""
        from strands_decider.core import resize_image, resized_size

        prepared = [resize_image(im, *resized_size(*im.size)) for im in images]
        out = self.processor(images=prepared, do_resize=False, return_tensors="pt")
        got = torch.as_tensor(out["num_soft_tokens_per_image"]).view(-1).tolist()
        if got != counts:
            raise SystemExit(f"the processor counts {got} soft tokens, the core {counts}")
        return out["pixel_values"].to(self.device, self.dtype), out["image_position_ids"].to(
            self.device
        )

    @torch.inference_mode()
    def probabilities(self, ids, s: int, k: int, mm=None) -> list[float]:
        j = torch.arange(len(ids), device=self.device)
        mask = ((j[None, :] < s) | (j[:, None] >= s))[None, None]
        kwargs = dict(
            input_ids=torch.tensor([ids], device=self.device),
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            use_cache=False,
        )
        if mm is not None:
            kwargs.update(pixel_values=mm[0], image_position_ids=mm[1])
        h = self.decoder(**kwargs).last_hidden_state[0, -1]
        z = torch.nn.functional.linear(
            h.to(torch.bfloat16), self.readout
        ).float()  # rounded to bf16
        z = torch.tanh(z / self.softcap) * self.softcap / self.temperature
        return torch.softmax(z[:k], dim=-1).tolist()


def run_requests(ref: Reference, a) -> None:
    from strands_decider.core import Request, load_tokenizer, render

    tok = load_tokenizer(a.ckpt)
    with open(a.out, "w") as out:
        for i, line in enumerate(open(a.requests)):
            rendered = render(
                Request.model_validate(json.loads(line)), tok, max_length=a.max_length
            )
            images = [decode(im.data) for im in rendered.images]
            mm = ref.pixels(images, [im.tokens for im in rendered.images]) if images else None
            for name, q in rendered.questions.items():
                p = ref.probabilities(list(q.ids), q.state_len, len(q.code_token_ids), mm)
                out.write(
                    json.dumps(
                        {"request": i, "question": name, "images": len(images), "probabilities": p}
                    )
                    + "\n"
                )


def run_rows(ref: Reference, a) -> None:
    rows = [json.loads(x) for x in open(a.rows) if x.strip()][: a.n]
    with open(a.out, "w") as out:
        for i, r in enumerate(rows):
            p = ref.probabilities(r["token_ids"], r["prefix_end"], len(r["label_token_ids"]))
            out.write(
                json.dumps(
                    {**r, "probabilities": p} if a.annotate else {"row": i, "probabilities": p}
                )
                + "\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--requests")
    ap.add_argument("--rows")
    ap.add_argument("--n", type=int, default=10**9)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--annotate", action="store_true", help="--rows: write each row with its probabilities"
    )
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dtype", default="bfloat16", help="the serving precision; float32 for the CPU tier"
    )
    a = ap.parse_args()
    ref = Reference(a.ckpt, a.device, a.dtype)
    if a.requests:
        run_requests(ref, a)
    if a.rows:
        run_rows(ref, a)


if __name__ == "__main__":
    main()
