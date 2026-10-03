"""Image input: the multimodal Qwen3.5 torso, with images inside `<state>`.

Qwen3.5 checkpoints are natively multimodal; `StrandsDeciderModel._load_torso` keeps
only the text decoder. `VisionDeciderModel` keeps the vision tower (ViT and patch
merger, frozen) and loads a text checkpoint's adapter and head onto the same decoder
inside it, so a published checkpoint such as v19 answers questions about images with
no retraining. Images go inside `<state>` as Qwen vision placeholders, before the state
text. Everything after the state is text, so the shared-prefix cache, the pointer
readout, the temperatures and every confidence formula are unchanged.

Positions are passed explicitly on the image path. Qwen3.5 uses M-RoPE: an image
advances the three rotary axes by its grid size, not its token count, so text after an
image sits at `token index + rope_delta` (negative). Left to itself, a suffix-only
forward after a cached prefix builds positions from the full prefix+suffix mask, and
reads a `rope_deltas` left on the module by the previous request.

Needs transformers >= 5.18 and Pillow (`pip install "strands-decider[vision]"`).
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from .infer import EngineConfig, SystemOneEngine, UnforkableCache, _expand_cache, _to_answer
from .modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    apply_temperature,
    checkpoint_dir,
    config_path,
    gather_options,
    load_head_state,
    masked_log_softmax,
    pool_last_token,
)
from .prompting import RenderedQuestion, render_question, render_state
from .schema import Answer, Content, Question, SystemOneRequest, SystemOneResponse, Usage

if TYPE_CHECKING:
    from PIL import Image

VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"
MIN_TRANSFORMERS = (5, 18)


# ---- prompt ------------------------------------------------------------------------


def render_image_state(state: Content, n_images: int) -> str:
    """`render_state(state)` with one unexpanded placeholder per image, before the text.

    Images come first so the expensive part of the prefix (the vision tower and the
    image tokens through the decoder) is identical across the questions of a request.
    """
    text = render_state(state)
    if not n_images:
        return text
    placeholders = "\n".join(f"{VISION_START}{IMAGE_PAD}{VISION_END}" for _ in range(n_images))
    body = text[len("<state>\n") : -len("\n</state>\n")]
    inner = f"{placeholders}\n{body}" if body else placeholders
    return f"<state>\n{inner}\n</state>\n"


def expand_image_tokens(prompt: str, tokens_per_image: Sequence[int]) -> str:
    """Replace the i-th `<|image_pad|>` with `tokens_per_image[i]` copies.

    The processor does the same expansion; doing it on the string lets the tokeniser's
    offsets be read against exactly the text the model sees.
    """
    parts = prompt.split(IMAGE_PAD)
    if len(parts) - 1 != len(tokens_per_image):
        raise ValueError(
            f"prompt holds {len(parts) - 1} image placeholders but "
            f"{len(tokens_per_image)} token counts were given"
        )
    out = [parts[0]]
    for n, tail in zip(tokens_per_image, parts[1:], strict=True):
        out.append(IMAGE_PAD * int(n))
        out.append(tail)
    return "".join(out)


def image_tokens_for_grid(grid_thw: Sequence[Sequence[int]], merge_size: int) -> list[int]:
    """Tokens each image contributes after the patch merger: t*h*w // merge_size**2."""
    m = merge_size * merge_size
    return [int(t) * int(h) * int(w) // m for t, h, w in grid_thw]


# ---- images ------------------------------------------------------------------------


def _pil() -> Any:
    try:
        from PIL import Image as PILImage
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('image input needs Pillow: pip install "strands-decider[vision]"') from e
    return PILImage


# What docs/vision.md promises. Anything else Pillow can open (EPS through Ghostscript, for
# one) is refused before it is parsed further.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")


def decode_image(data: str, max_pixels: int = 4096 * 4096) -> Image.Image:
    """A base64 image (optionally a `data:image/...;base64,` URI) as an upright RGB image.

    Only IMAGE_FORMATS are opened, and the size is checked from the header before any
    pixel is decoded, so a small file that expands to a huge bitmap is refused cheaply.
    """
    payload = re.sub(r"^data:image/[\w.+-]+;base64,", "", data.strip())
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image is not valid base64: {e}") from e
    pil = _pil()
    try:
        img = pil.open(io.BytesIO(raw), formats=IMAGE_FORMATS)
    except Exception as e:
        raise ValueError(f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {e}") from e
    w, h = img.size
    if w * h > max_pixels:
        raise ValueError(f"image is {w}x{h}; at most {max_pixels:,} pixels are accepted")
    try:
        img.load()
    except Exception as e:
        raise ValueError(f"image could not be decoded: {e}") from e
    from PIL import ImageOps

    img = ImageOps.exif_transpose(img)  # phone photos store their rotation in EXIF
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = pil.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = pil.alpha_composite(white, rgba)  # transparent areas read as white, not black
    return fit_image(img, 0)


def fit_image(img: Image.Image, long_side: int) -> Image.Image:
    """RGB, scaled down (never up) so the longer side is at most `long_side` (0: as is)."""
    img = img.convert("RGB")
    w, h = img.size
    if long_side and max(w, h) > long_side:
        s = long_side / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), _pil().BICUBIC)
    return img


# ---- model -------------------------------------------------------------------------


def _check_transformers() -> None:
    import transformers

    found = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
    if found < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"image input needs transformers >= {'.'.join(map(str, MIN_TRANSFORMERS))}, "
            f"found {transformers.__version__}"
        )


def mm_token_type_ids(torso: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """0 for text, 1 for image tokens: what the Qwen processor returns beside input_ids."""
    cfg: Any = torso.config
    is_image: torch.Tensor = input_ids == cfg.image_token_id
    return is_image.to(torch.int32)


def qwen_base(torso: nn.Module) -> nn.Module:
    """The `Qwen3_5Model` under any PEFT wrapping (PeftModel -> LoraModel -> model)."""
    base = getattr(torso, "base_model", torso)
    found: nn.Module = getattr(base, "model", base)
    return found


class VisionDeciderModel(StrandsDeciderModel):
    """A Strands Decider whose torso keeps Qwen3.5's vision tower. Readout inherited."""

    # The image tensors of the batch being forwarded. The parent's forward and KL
    # reference call `self.encode(input_ids, attention_mask, past_key_values=...)`,
    # which knows nothing of images; `forward` sets this for the duration of the call.
    _mm: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)

    @staticmethod
    def hidden_size(torso: nn.Module) -> int:
        # The multimodal config is composite; the decoder width is on text_config.
        cfg: Any = torso.config
        return int(cfg.get_text_config().hidden_size)

    @staticmethod
    def is_hybrid(torso: nn.Module) -> bool:
        cfg = getattr(torso, "config", None)
        text = cfg.get_text_config() if cfg is not None and hasattr(cfg, "get_text_config") else cfg
        types = getattr(text, "layer_types", None) or []
        return any(t != "full_attention" for t in types)

    @staticmethod
    def _load_torso(
        config: StrandsDeciderConfig,
        device_map: str | None,
        attn_implementation: str | None,
    ) -> nn.Module:
        _check_transformers()
        import transformers

        base_cfg = transformers.AutoConfig.from_pretrained(config.base_model)
        if base_cfg.model_type != "qwen3_5":
            raise ValueError(
                f"image input needs a multimodal qwen3_5 checkpoint, got {base_cfg.model_type!r}"
            )
        kwargs: dict[str, Any] = {"dtype": getattr(torch, config.torch_dtype)}
        if device_map:
            kwargs["device_map"] = device_map
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        full = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            config.base_model, **kwargs
        )
        torso: Any = full.model  # .visual (ViT + merger) and .language_model
        torso.config.use_cache = True
        for p in torso.visual.parameters():
            p.requires_grad_(False)
        loaded: nn.Module = torso
        return loaded

    def attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        cfg = self.config
        # PEFT full-matches a string `target_modules` against each module name. Scoped
        # to the language model, so the ViT's own projections are never adapted.
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=rf"^language_model\..*\.({'|'.join(cfg.lora_targets)})$",
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.torso = get_peft_model(self.torso, lora_cfg)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pixel_values is None and image_grid_thw is None:
            pixel_values, image_grid_thw = self._mm
        extra: dict[str, Any] = {}
        if image_grid_thw is not None:
            extra["mm_token_type_ids"] = mm_token_type_ids(self.torso, input_ids)
        if position_ids is not None:
            extra["position_ids"] = position_ids
        out = self.torso(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
            **extra,
        )
        hidden: torch.Tensor = out.last_hidden_state
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_slots: torch.Tensor,
        labels: torch.Tensor | None = None,
        label_dist: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        past_key_values: Any = None,
        temperature: Any | None = None,
        opt_idx: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._mm = (pixel_values, image_grid_thw)
        try:
            return super().forward(
                input_ids, attention_mask, n_slots,
                labels=labels, label_dist=label_dist, weights=weights,
                past_key_values=past_key_values, temperature=temperature, opt_idx=opt_idx,
            )
        finally:
            self._mm = (None, None)

    def frozen_slot_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_slots: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The untouched torso's option-number reading, with the image in the prompt."""
        self._mm = (pixel_values, image_grid_thw)
        try:
            return super().frozen_slot_log_probs(input_ids, attention_mask, n_slots)
        finally:
            self._mm = (None, None)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device_map: str | None = None,
        attn_implementation: str | None = None,
    ) -> VisionDeciderModel:
        """Load a Strands Decider checkpoint (e.g. v19) onto the multimodal torso.

        Text checkpoints trained their adapter on the bare decoder (`layers.N...`); in
        the multimodal torso the same decoder sits under `language_model.layers.N...`,
        so the adapter's keys are remapped onto it. Text behaviour is unchanged: the
        weights the text path runs through are identical.
        """
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        path = checkpoint_dir(path)
        config = StrandsDeciderConfig.from_json(config_path(path))
        if config.head_type != "pointer":
            raise ValueError("image input needs a pointer-head checkpoint")
        lora_file = os.path.join(path, "lora", "adapter_model.safetensors")
        if config.use_lora and not os.path.exists(lora_file):
            raise FileNotFoundError(f"{lora_file}: missing, but the checkpoint config sets use_lora")
        head_state = load_head_state(path)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = cls(config, cls._load_torso(config, device_map, attn_implementation), tok)
        if config.use_lora:
            model.attach_lora()
            state = {
                k if ".language_model." in k
                else k.replace("base_model.model.", "base_model.model.language_model.", 1): v
                for k, v in load_file(lora_file).items()
            }
            res: Any = set_peft_model_state_dict(model.torso, state)
            missing = [k for k in res.missing_keys if "lora_" in k]
            if res.unexpected_keys or missing:
                raise RuntimeError(
                    f"adapter does not fit the multimodal decoder: unexpected "
                    f"{res.unexpected_keys[:3]}, missing {missing[:3]}"
                )
        model.head.load_state_dict(head_state)
        model.head.to(torch.float32)
        model.eval()
        return model


# ---- engine ------------------------------------------------------------------------


@dataclass
class VisionEngineConfig(EngineConfig):
    # Longest image side after resizing. Fixed per deployment so the token budget is
    # predictable: 448 px square is 196 tokens (patch 16, merge 2).
    image_long_side: int = 448
    max_images: int = 4
    # Refused before decoding: 4096 x 4096 covers a 12 MP phone photo and bounds memory.
    max_image_pixels: int = 4096 * 4096


class VisionEngine(SystemOneEngine):
    """SystemOneEngine that also answers over images. Text-only requests take the
    unchanged text path."""

    def __init__(
        self,
        model: VisionDeciderModel,
        config: VisionEngineConfig | None = None,
        image_processor: Any = None,
    ):
        super().__init__(model, config or VisionEngineConfig())
        self.vcfg: VisionEngineConfig = (
            self.cfg if isinstance(self.cfg, VisionEngineConfig) else VisionEngineConfig()
        )
        if image_processor is None:
            # Pinned to the PIL backend. `Qwen2VLImageProcessor` picks torchvision when it is
            # installed, and that changes pixel values enough to flip some answers; the PIL
            # backend gives the same answers everywhere, and is what every result was measured on.
            from transformers import Qwen2VLImageProcessorPil

            image_processor = Qwen2VLImageProcessorPil.from_pretrained(model.config.base_model)
        self.image_processor = image_processor

    def _reset_positions(self) -> None:
        # The text path leaves positions to transformers, which would read a
        # `rope_deltas` stored on the torso by any earlier plain image forward (a
        # training or evaluation step on the same model). The image path below passes
        # positions explicitly and stores none.
        base: Any = qwen_base(self.model.torso)
        if hasattr(base, "rope_deltas"):
            base.rope_deltas = None

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        self._reset_positions()
        if not request.images:
            return super().evaluate(request)
        if len(request.images) > self.vcfg.max_images:
            raise ValueError(f"{len(request.images)} images; this server takes at most {self.vcfg.max_images}")
        images = [fit_image(decode_image(b, self.vcfg.max_image_pixels), self.vcfg.image_long_side)
                  for b in request.images]
        names = list(request.questions)
        rendered = [render_question(request.questions[n]) for n in names]
        answers: dict[str, Answer] = {}
        total = 0
        for start in range(0, len(names), self.cfg.max_batch):
            chunk = rendered[start : start + self.cfg.max_batch]
            try:
                probs, ntok = self._image_probs(request.state, images, chunk)
            except UnforkableCache as e:
                raise RuntimeError(f"hybrid cache could not be forked: {e}") from e
            total += ntok
            for i, name in enumerate(names[start : start + self.cfg.max_batch]):
                rq = chunk[i]
                answers[name] = _to_answer(
                    rq, probs[i, : rq.n_slots].tolist(),
                    ordinal_smoothing=self.model.config.ordinal_smoothing,
                )
        return SystemOneResponse(
            model=self.cfg.model_name, answers=answers,
            usage=Usage(input_tokens=total, output_tokens=len(names)),
        )

    def ask_images(
        self, state: Content, questions: dict[str, Question], images: Sequence[str]
    ) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions, images=list(images)))

    def _fit_images(
        self, state: Content, images: list[Image.Image], question_texts: list[str]
    ) -> tuple[list[int], list[list[int]], dict[str, torch.Tensor]]:
        """Question reserve first, then the image block whole, then state text.

        As in `_fit`: with `strict_window` an over-long prompt is refused (HTTP 422);
        otherwise the questions keep their reserve and the state text loses its end, the
        same side `_fit` and training cut. The images are never cut.
        """
        max_len = self.model.config.max_length
        enc = self.tok(question_texts, add_special_tokens=False, return_offsets_mapping=True)
        q, offs = enc["input_ids"], enc["offset_mapping"]
        longest = max(len(x) for x in q)

        img = self.image_processor(images=images, return_tensors="pt")
        counts = image_tokens_for_grid(img["image_grid_thw"].tolist(), self.image_processor.merge_size)
        mm = {"pixel_values": img["pixel_values"].to(self.device),
              "image_grid_thw": img["image_grid_thw"].to(self.device)}
        s = self.tok(expand_image_tokens(render_image_state(state, len(images)), counts),
                     add_special_tokens=True)["input_ids"]
        vend = self.tok.convert_tokens_to_ids(VISION_END)
        keep = max(i for i, t in enumerate(s) if t == vend) + 1  # through the last image

        if self.cfg.strict_window:
            if len(s) + longest > max_len:
                raise ValueError(
                    f"prompt of {len(s) + longest} tokens exceeds the context window "
                    f"of {max_len} tokens"
                )
            self._last_offsets = list(offs)
            return s, q, mm
        reserve = min(longest, max(1, int(max_len * self.cfg.max_question_fraction)))
        cut = [max(0, len(x) - reserve) for x in q]
        self._last_offsets = [o[c:] for o, c in zip(offs, cut, strict=True)]
        q = [x[c:] for x, c in zip(q, cut, strict=True)]
        budget = max_len - reserve
        if keep > budget:
            raise ValueError(
                f"the images take {keep} tokens of a {budget}-token state budget; "
                "send fewer or smaller images"
            )
        s = s[:budget]
        return s, q, mm

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _image_probs(
        self, state: Content, images: list[Image.Image], rendered: list[RenderedQuestion]
    ) -> tuple[torch.Tensor, int]:
        m = len(rendered)
        s, q, mm = self._fit_images(state, images, [rq.text for rq in rendered])
        prefix_ids = torch.tensor([s], device=self.device)
        n = prefix_ids.size(1)
        tt = mm_token_type_ids(self.model.torso, prefix_ids)
        base: Any = qwen_base(self.model.torso)
        mpos, delta_t = base.get_rope_index(
            prefix_ids, tt, image_grid_thw=mm["image_grid_thw"]
        )
        delta = int(delta_t.view(-1)[0])
        text_pos = torch.arange(n, device=self.device).view(1, 1, -1)
        prefix_out = self.model.torso(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            position_ids=torch.cat([text_pos, mpos], dim=0),  # [4, 1, n]: text + 3 M-RoPE rows
            mm_token_type_ids=tt,
            use_cache=True,
            return_dict=True,
            **mm,
        )
        cache = prefix_out.past_key_values if m == 1 else _expand_cache(prefix_out.past_key_values, m)

        suffix_ids, suffix_mask = self._pad(q)
        full_mask = torch.cat(
            [torch.ones(m, n, dtype=suffix_mask.dtype, device=self.device), suffix_mask], dim=1
        )
        st = (torch.arange(suffix_ids.size(1), device=self.device) + n).view(1, 1, -1).expand(1, m, -1)
        model = cast(VisionDeciderModel, self.model)
        hidden = model.encode(
            suffix_ids, full_mask, past_key_values=cache,
            position_ids=torch.cat([st, (st + delta).expand(3, m, -1)], dim=0),
        )
        pooled = pool_last_token(hidden, full_mask).to(torch.float32)
        options = gather_options(hidden, self._option_idx(rendered, 0)).to(torch.float32)
        head: Any = self.model.head
        logits = apply_temperature(head(pooled, options),
                                   self._temperatures([rq.kind for rq in rendered]))
        probs = masked_log_softmax(logits, torch.tensor([rq.n_slots for rq in rendered], device=self.device))
        return probs.exp(), n + int(suffix_mask.sum().item())


def load_vision_engine(
    checkpoint: str,
    config: EngineConfig | None = None,
    *,
    attn_implementation: str | None = None,
    image_long_side: int = 448,
) -> VisionEngine:
    """A vision engine configured as `create_app` configures a text one (device, prefix
    cache, `strict_window`, `max_batch`, model name), plus the image settings."""
    base = config or EngineConfig()
    names = {f.name for f in fields(EngineConfig)}
    cfg = VisionEngineConfig(**{k: v for k, v in asdict(base).items() if k in names},
                             image_long_side=image_long_side)
    model = VisionDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
    return VisionEngine(model, cfg)
