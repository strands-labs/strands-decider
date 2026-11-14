"""The transformers reference for a Strands Decider checkpoint, text and images, written from the specification alone.

    python reference.py CKPT --requests REQUESTS.jsonl --out OUT.jsonl [--resize pil|torchvision] [--pixels PIX.jsonl]
    python reference.py CKPT --rows ROWS.jsonl --out OUT.jsonl [--n 60]

``--requests``: JSON Lines of System One requests (``state``, ``questions``, optional ``images`` as base64). Each is
rendered with the core library (``strands_decider.core.render``) and the checkpoint's prompt template. Each image is
decoded as v1 decodes it, resized to the specification's W x H with Pillow's bicubic resampling and prepared by the
checkpoint's image processor with its resize off (section 3.3): ``Gemma4ImageProcessorPil``, or the 12B's
``Gemma4UnifiedImageProcessor``. ``--resize torchvision`` instead lets transformers' torchvision backend resize the
decoded image itself, as stock vLLM would; it measures that difference and is not the reference.

``--rows``: JSON Lines of pre-rendered text prompts (``token_ids``, ``prefix_end``, ``label_token_ids``, optionally a
stored reference's ``probabilities``), as the GPU checks' rows; with stored probabilities, top-1 agreement and |dp|
against them are printed. With ``--annotate`` each row is written back with the reference's probabilities.

The model is transformers' own class for the checkpoint's layout (``Gemma4ForConditionalGeneration``,
``Gemma4UnifiedForConditionalGeneration``, or ``Gemma4TextModel`` for a bare decoder), in bf16 with SDPA, one prompt
per forward pass. The prefix_bidir mask goes to every layer as a dict keyed by layer type, which replaces Gemma 4's own
masks (causal, windowed, and the "vision" image mask) on every layer: token i sees j when j < S, or when i >= S. The
readout is the specification's: a bf16 GEMM of the final-norm hidden state at the last token, then in float32 the
soft-cap and the temperature, then a softmax over the question's own codes. Every output line carries the prompt's ids
digest, so a runtime's answers can be matched to it question by question.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from pathlib import Path

import torch

IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")


def spec_size(w: int, h: int) -> tuple[int, int]:
    """Section 3.3's W and H."""
    f = math.sqrt(645120 / (h * w))
    H = math.floor((f * h) / 48) * 48
    W = math.floor((f * w) / 48) * 48
    if H == 0:
        H, W = 48, min(math.floor(w / h) * 48, 13440)
    elif W == 0:
        W, H = 48, min(math.floor(h / w) * 48, 13440)
    return W, H


def bicubic(im, width, height):
    """Section 3.3's resize, for a core library without ``resize_image`` (which is the same): Pillow's bicubic
    resampling, the width's pass and then the height's, each left out when its size does not change. Two calls,
    because Pillow 13 does the height's pass first in some downscales."""
    if im.size[0] != width:
        im = im.resize((width, im.size[1]), resample=3)  # 3 = Image.BICUBIC
    if im.size[1] != height:
        im = im.resize((width, height), resample=3)
    return im


def decode(data: bytes):
    """v1's decode: first frame, EXIF orientation, transparency onto white, RGB (specification, section 3.3)."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def pixel_digests(pv: torch.Tensor) -> dict[str, str]:
    """sha256 prefixes of one image's pixel values as float32 (the processor's output) and as bf16 (what the vision
    encoder computes on), so that a runtime's pixels can be compared exactly."""
    f32 = pv.to(torch.float32).contiguous()
    bf16 = f32.to(torch.bfloat16).view(torch.int16).contiguous()
    return {
        "float32": hashlib.sha256(f32.numpy().tobytes()).hexdigest()[:16],
        "bfloat16": hashlib.sha256(bf16.numpy().tobytes()).hexdigest()[:16],
        "shape": list(pv.shape),
    }


def ids_digest(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:16]


class Reference:
    def __init__(
        self, ckpt: Path, device: str, dtype: str = "bfloat16", device_map: str | None = None
    ) -> None:
        import transformers
        from safetensors.torch import load_file

        self.ckpt, self.device = ckpt, device
        self.causal = False
        cfg = json.loads((ckpt / "config.json").read_text())
        self.dc = json.loads((ckpt / "decision_config.json").read_text())
        arch = cfg["architectures"][0]
        self.dtype = dtype = getattr(torch, dtype)
        place = {"device_map": device_map} if device_map else {}
        if arch == "Gemma4TextModel":
            self.model = transformers.Gemma4TextModel.from_pretrained(
                ckpt, dtype=dtype, attn_implementation="sdpa", **place
            )
            self.decoder, self.processor = self.model, None
            text = cfg
        else:
            cls = getattr(transformers, arch)
            config = transformers.AutoConfig.from_pretrained(ckpt)
            config.audio_config = (
                None  # no audio: E2B's and E4B's audio encoder has no SDPA path in transformers
            )
            self.model = cls.from_pretrained(
                ckpt, config=config, dtype=dtype, attn_implementation="sdpa", **place
            )
            self.decoder = self.model.model
            text = cfg["text_config"]
            if arch == "Gemma4UnifiedForConditionalGeneration":
                self.processor = transformers.Gemma4UnifiedImageProcessor.from_pretrained(ckpt)
                self.torchvision = self.processor
            else:
                self.processor = transformers.Gemma4ImageProcessorPil.from_pretrained(ckpt)
                self.torchvision = transformers.Gemma4ImageProcessor.from_pretrained(ckpt)
        if not device_map:
            self.model.to(device)
        self.model.eval()
        w = load_file(str(ckpt / "readout.safetensors"))["weight"]
        self.readout = w.to(device=device, dtype=torch.bfloat16)
        self.softcap = float(
            self.dc.get("final_logit_softcapping") or text["final_logit_softcapping"]
        )
        self.temperature = float(self.dc["temperature"])

    def pixels(self, images, resize: str):
        """(pixel_values, image_position_ids, soft-token counts, per-image sha256 of the pixel values)."""
        if not images:
            return None
        if resize == "pil":
            try:  # the specification's resize, from the core library
                from strands_decider.core import resize_image
            except ImportError:
                resize_image = bicubic
            prepared = [resize_image(im, *spec_size(*im.size)) for im in images]
            out = self.processor(images=prepared, do_resize=False, return_tensors="pt")
        else:
            out = self.torchvision(images=images, return_tensors="pt")
        pv, pos = out["pixel_values"], out["image_position_ids"]
        counts = [
            int(n) for n in torch.as_tensor(out["num_soft_tokens_per_image"]).view(-1).tolist()
        ]
        shas = [pixel_digests(pv[i]) for i in range(len(images))]
        return pv.to(self.device, self.dtype), pos.to(self.device), counts, shas

    @torch.inference_mode()
    def probabilities(self, ids, s: int, k: int, mm=None) -> list[float]:
        n = len(ids)
        j = torch.arange(n, device=self.device)
        allowed = (j[None, :] < s) | (j[:, None] >= s)
        if (
            self.causal
        ):  # diagnostic: Gemma 4's causal attention (no window) instead of prefix_bidir
            allowed = j[None, :] <= j[:, None]
        mask = allowed[None, None]
        kwargs = dict(
            input_ids=torch.tensor([ids], device=self.device),
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            use_cache=False,
        )
        if mm is not None:
            kwargs.update(pixel_values=mm[0], image_position_ids=mm[1])
        h = self.decoder(**kwargs).last_hidden_state[0, -1].to(self.readout.device)
        z = torch.nn.functional.linear(
            h.to(torch.bfloat16), self.readout
        ).float()  # the bf16 GEMM, rounded to bf16
        z = torch.tanh(z / self.softcap) * self.softcap / self.temperature
        return torch.softmax(z[:k], dim=-1).tolist()


def run_requests(ref: Reference, a) -> None:
    from strands_decider.core import render
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(a.ckpt / "tokenizer.json"))
    prompt = ref.dc["prompt"]
    out = open(a.out, "w")
    pix = open(a.pixels, "w") if a.pixels else None
    t0 = time.time()
    for i, line in enumerate(open(a.requests)):
        body = json.loads(line)
        rendered = render(body, tok, prompt, max_length=a.max_length)
        images = [decode(im.data) for im in rendered.images]
        mm = ref.pixels(images, a.resize)
        if mm is not None:
            if mm[2] != [im.tokens for im in rendered.images]:
                raise SystemExit(
                    f"request {i}: the processor counts {mm[2]}, the core {[im.tokens for im in rendered.images]}"
                )
            if pix:
                for im, sha in zip(rendered.images, mm[3], strict=True):
                    pix.write(
                        json.dumps({"request": i, "image_sha256": im.sha256, "pixels": sha}) + "\n"
                    )
        for name, q in rendered.questions.items():
            p = ref.probabilities(list(q.ids), q.state_len, len(q.code_token_ids), mm)
            out.write(
                json.dumps(
                    {
                        "request": i,
                        "question": name,
                        "n_tokens": len(q.ids),
                        "state_len": q.state_len,
                        "images": len(images),
                        "ids": ids_digest(q.ids),
                        "probabilities": p,
                    }
                )
                + "\n"
            )
        if i % 50 == 0:
            print(f"{i} requests, {time.time() - t0:.0f} s", flush=True)
    out.close()


def run_rows(ref: Reference, a) -> None:
    rows = [json.loads(x) for x in open(a.rows) if x.strip()][: a.n]
    agree, dps = 0, []
    with open(a.out, "w") as out:
        for i, r in enumerate(rows):
            p = ref.probabilities(r["token_ids"], r["prefix_end"], len(r["label_token_ids"]))
            if (
                a.annotate
            ):  # the row itself, with the reference's probabilities (for client.py --rows)
                out.write(json.dumps({**r, "probabilities": p}) + "\n")
                continue
            out.write(json.dumps({"row": i, "id": r.get("id"), "probabilities": p}) + "\n")
            if "probabilities" in r:
                q = r["probabilities"]
                agree += max(range(len(p)), key=p.__getitem__) == max(
                    range(len(q)), key=q.__getitem__
                )
                dps.append(max(abs(x - y) for x, y in zip(p, q, strict=True)))
    if dps:
        print(
            json.dumps(
                {
                    "rows": len(dps),
                    "top1": agree / len(dps),
                    "mean_dp": sum(dps) / len(dps),
                    "max_dp": max(dps),
                }
            )
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
    ap.add_argument("--resize", choices=("pil", "torchvision"), default="pil")
    ap.add_argument("--pixels", help="write each image's pixel-values digest here")
    ap.add_argument(
        "--annotate",
        action="store_true",
        help="--rows: write each row with the reference's probabilities",
    )
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        help="the model's dtype: bfloat16 is the reference; float32 is a "
        "diagnostic of the reference's own rounding (the readout stays a bf16 GEMM)",
    )
    ap.add_argument(
        "--mask",
        choices=("prefix_bidir", "causal"),
        default="prefix_bidir",
        help="causal is a diagnostic: how much the mask moves a checkpoint's answers",
    )
    ap.add_argument(
        "--device-map",
        default=None,
        help="e.g. auto, to spread a float32 model over the visible GPUs",
    )
    a = ap.parse_args()
    ap_causal = a.mask == "causal"
    ref = Reference(a.ckpt, a.device, a.dtype, a.device_map)
    ref.causal = ap_causal
    if a.requests:
        run_requests(ref, a)
    if a.rows:
        run_rows(ref, a)


if __name__ == "__main__":
    main()
