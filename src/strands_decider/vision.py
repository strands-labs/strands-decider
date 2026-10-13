"""Image and audio input: a multimodal torso (Qwen3.5 or Gemma 4), media inside `<state>`.

Qwen3.5 and Gemma 4 checkpoints are natively multimodal; `StrandsDeciderModel._load_torso`
keeps only the text decoder. `VisionDeciderModel` keeps the encoders (frozen: Qwen's ViT
and patch merger; Gemma's vision encoder and its projection into the decoder, and with
`audio=True` its audio encoder and projection) and loads a text checkpoint's adapter and
head onto the same decoder inside it, so a published checkpoint such as v19 or
strands-decider-E4B-gemma4-v1-2610 answers questions about images (and, on Gemma 4 E2B /
E4B, audio) with no retraining. Media go inside `<state>` as the family's placeholders,
images first, then audio clips, then the state text. Everything after the state is text, so
the shared-prefix cache, the pointer readout, the temperatures and every confidence formula
are unchanged.

The two families differ in how media enter the decoder:

* Qwen3.5 (images only): `<|vision_start|>` + one `<|image_pad|>` per merged patch +
  `<|vision_end|>`, the count read from the processor's `image_grid_thw`. Positions are
  passed explicitly on the image path. Qwen3.5 uses M-RoPE: an image advances the three
  rotary axes by its grid size, not its token count, so text after an image sits at
  `token index + rope_delta` (negative). Left to itself, a suffix-only forward after a
  cached prefix builds positions from the full prefix+suffix mask, and reads a
  `rope_deltas` left on the module by the previous request.
* Gemma 4: an image is `<|image>` + one `<|image|>` per soft token + `<image|>`, the count
  the processor reports (`num_soft_tokens_per_image`, at most `max_soft_tokens`: 280 for
  E2B and E4B); an audio clip is `<|audio>` + one `<|audio|>` per 40 ms + `<audio|>` (see
  audio.py). Gemma uses plain 1-D RoPE, one position per token, so the default positions
  (`arange + cached length`) are already right for a suffix after a cached prefix, and no
  position state is kept on the module.

Needs transformers >= 5.18 and Pillow (`pip install "strands-decider[vision]"`); audio also
needs soundfile and scipy (`[audio]`).
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
    HostEmbedding,
    StrandsDeciderConfig,
    StrandsDeciderModel,
    apply_temperature,
    base_revision,
    checkpoint_dir,
    config_path,
    ensure_bos,
    gather_options,
    load_head_state,
    masked_log_softmax,
    pool_last_token,
    serving_attn_implementation,
)
from .prompting import RenderedQuestion, render_question, render_state
from .schema import Answer, Content, Question, SystemOneRequest, SystemOneResponse, Usage

if TYPE_CHECKING:
    from PIL import Image

VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"
# Gemma 4: begin-of-image, the soft-token placeholder, end-of-image (its processor's
# `boi_token`, `image_token`, `eoi_token`), and the same three for audio.
GEMMA_BOI = "<|image>"
GEMMA_IMAGE = "<|image|>"
GEMMA_EOI = "<image|>"
GEMMA_BOA = "<|audio>"
GEMMA_AUDIO = "<|audio|>"
GEMMA_EOA = "<audio|>"
MIN_TRANSFORMERS = (5, 18)

QWEN = "qwen3_5"
GEMMA = "gemma4"
# model_type of the multimodal checkpoint -> (start, placeholder, end)
IMAGE_TOKENS = {QWEN: (VISION_START, IMAGE_PAD, VISION_END), GEMMA: (GEMMA_BOI, GEMMA_IMAGE, GEMMA_EOI)}
AUDIO_TOKENS = {GEMMA: (GEMMA_BOA, GEMMA_AUDIO, GEMMA_EOA)}
# Longest image side the engine scales to before the processor, per family. Qwen's token
# count follows the pixels (448 px square: 196 tokens); Gemma 4's processor resizes every
# image to its own fixed budget (280 tokens, about 645,000 pixels), so a cap there only
# discards detail, and none is applied.
DEFAULT_LONG_SIDE = {QWEN: 448, GEMMA: 0}


def image_tokens(family: str) -> tuple[str, str, str]:
    """(start, placeholder, end) of an image in the prompt of `family` (a model_type)."""
    if family not in IMAGE_TOKENS:
        raise ValueError(f"image input needs a qwen3_5 or gemma4 checkpoint, got {family!r}")
    return IMAGE_TOKENS[family]


def audio_tokens(family: str) -> tuple[str, str, str]:
    """(start, placeholder, end) of an audio clip in the prompt of `family`."""
    if family not in AUDIO_TOKENS:
        raise ValueError(f"audio input needs a gemma4 checkpoint with an audio encoder, got {family!r}")
    return AUDIO_TOKENS[family]


# ---- prompt ------------------------------------------------------------------------


def render_image_state(state: Content, n_images: int, family: str = QWEN, n_audio: int = 0) -> str:
    """`render_state(state)` with one unexpanded placeholder per image, then per audio clip,
    before the text.

    Media come first so the expensive part of the prefix (the encoders and the media
    tokens through the decoder) is identical across the questions of a request.
    """
    text = render_state(state)
    if not n_images and not n_audio:
        return text
    blocks = ["".join(image_tokens(family))] * n_images
    if n_audio:
        blocks += ["".join(audio_tokens(family))] * n_audio
    placeholders = "\n".join(blocks)
    body = text[len("<state>\n") : -len("\n</state>\n")]
    inner = f"{placeholders}\n{body}" if body else placeholders
    return f"<state>\n{inner}\n</state>\n"


def expand_placeholders(prompt: str, pad: str, counts: Sequence[int]) -> str:
    """Replace the i-th `pad` with `counts[i]` copies.

    The processor does the same expansion; doing it on the string lets the tokeniser's
    offsets be read against exactly the text the model sees.
    """
    parts = prompt.split(pad)
    if len(parts) - 1 != len(counts):
        raise ValueError(
            f"prompt holds {len(parts) - 1} {pad} placeholders but {len(counts)} token counts were given"
        )
    out = [parts[0]]
    for n, tail in zip(counts, parts[1:], strict=True):
        out.append(pad * int(n))
        out.append(tail)
    return "".join(out)


def expand_image_tokens(prompt: str, tokens_per_image: Sequence[int], family: str = QWEN) -> str:
    """Replace the i-th image placeholder with `tokens_per_image[i]` copies."""
    return expand_placeholders(prompt, image_tokens(family)[1], tokens_per_image)


def image_tokens_for_grid(grid_thw: Sequence[Sequence[int]], merge_size: int) -> list[int]:
    """Tokens each image contributes after the patch merger: t*h*w // merge_size**2."""
    m = merge_size * merge_size
    return [int(t) * int(h) * int(w) // m for t, h, w in grid_thw]


# ---- images ------------------------------------------------------------------------


def load_image_processor(config: StrandsDeciderConfig) -> Any:
    """The image processor of `config.base_model`, pinned to its PIL backend.

    The torchvision backends (picked automatically when torchvision is installed) change
    pixel values enough to flip some answers; the PIL backend gives the same answers
    everywhere, and is what every result was measured on. Read at the base's revision, as
    the torso is.
    """
    import transformers

    rev: dict[str, Any] = {"revision": config.base_revision} if config.base_revision else {}
    cfg = transformers.AutoConfig.from_pretrained(config.base_model, **rev)
    if cfg.model_type == GEMMA:
        return transformers.Gemma4ImageProcessorPil.from_pretrained(config.base_model, **rev)
    image_tokens(cfg.model_type)  # refuses anything but the two families
    return transformers.Qwen2VLImageProcessorPil.from_pretrained(config.base_model, **rev)


def process_images(processor: Any, images: Sequence[Image.Image]) -> tuple[list[int], dict[str, torch.Tensor]]:
    """Pixels for the vision tower, and the number of placeholder tokens each image takes.

    Qwen: `pixel_values` and `image_grid_thw`, t*h*w // merge_size**2 tokens per image.
    Gemma 4: `pixel_values` and `image_position_ids` (patches padded to the processor's
    budget), and the processor's own `num_soft_tokens_per_image`.
    """
    out = processor(images=list(images), return_tensors="pt")
    if "num_soft_tokens_per_image" in out:
        counts = [int(n) for n in torch.as_tensor(out["num_soft_tokens_per_image"]).view(-1).tolist()]
        return counts, {"pixel_values": out["pixel_values"], "image_position_ids": out["image_position_ids"]}
    counts = image_tokens_for_grid(out["image_grid_thw"].tolist(), processor.merge_size)
    return counts, {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}


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
    from PIL import ImageOps

    try:
        img.load()
        # Phone photos store their rotation in EXIF. A malformed EXIF block raises
        # SyntaxError or struct.error here, which would otherwise surface as HTTP 500.
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise ValueError(f"image could not be decoded: {e}") from e
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
    """0 for text, 1 for image tokens: what the Qwen and Gemma 4 processors return beside
    input_ids (Gemma reads it only for bidirectional image attention, which E2B and E4B do
    not use; audio tokens stay 0, as no mask reads them)."""
    cfg: Any = torso.config
    is_image: torch.Tensor = input_ids == cfg.image_token_id
    return is_image.to(torch.int32)


def mm_base(torso: nn.Module) -> nn.Module:
    """The multimodal base (`Qwen3_5Model` or `Gemma4Model`) under any PEFT wrapping
    (PeftModel -> LoraModel -> model)."""
    base = getattr(torso, "base_model", torso)
    found: nn.Module = getattr(base, "model", base)
    return found


qwen_base = mm_base  # the name it had when Qwen3.5 was the only family


class VisionDeciderModel(StrandsDeciderModel):
    """A Strands Decider whose torso keeps its media encoders (Qwen3.5 or Gemma 4).
    Readout inherited."""

    # The media tensors of the batch being forwarded. The parent's forward and KL
    # reference call `self.encode(input_ids, attention_mask, past_key_values=...)`,
    # which knows nothing of media; `forward` sets this for the duration of the call.
    # Qwen: pixel_values, image_grid_thw. Gemma 4: pixel_values, image_position_ids,
    # input_features, input_features_mask.
    _mm: dict[str, torch.Tensor] = {}  # noqa: RUF012 - never mutated, only rebound

    @property
    def family(self) -> str:
        """`qwen3_5` or `gemma4`: the multimodal checkpoint's model_type."""
        cfg: Any = self.torso.config
        return str(cfg.model_type)

    @property
    def has_audio(self) -> bool:
        """True when the torso was built with its audio encoder (`load(..., audio=True)`)."""
        return getattr(mm_base(self.torso), "audio_tower", None) is not None

    @staticmethod
    def hidden_size(torso: nn.Module) -> int:
        # The multimodal config is composite; the decoder width is on text_config.
        cfg: Any = torso.config
        return int(cfg.get_text_config().hidden_size)

    @staticmethod
    def is_hybrid(torso: nn.Module) -> bool:
        # Recurrent (linear-attention) layers, as StrandsDeciderModel.is_hybrid reads them;
        # Gemma 4's sliding-window layers are attention, not recurrence.
        cfg = getattr(torso, "config", None)
        text = cfg.get_text_config() if cfg is not None and hasattr(cfg, "get_text_config") else cfg
        return "linear_attention" in (getattr(text, "layer_types", None) or [])

    @staticmethod
    def _load_torso(
        config: StrandsDeciderConfig,
        device_map: str | None,
        attn_implementation: str | None,
        audio: bool = False,
    ) -> nn.Module:
        _check_transformers()
        import transformers

        base_cfg = transformers.AutoConfig.from_pretrained(
            config.base_model, revision=config.base_revision
        )
        if base_cfg.model_type not in IMAGE_TOKENS:
            raise ValueError(
                f"image input needs a multimodal qwen3_5 or gemma4 checkpoint, got {base_cfg.model_type!r}"
            )
        if audio and getattr(base_cfg, "audio_config", None) is None:
            raise ValueError(f"{config.base_model} has no audio encoder (Gemma 4 E2B and E4B have one)")
        kwargs: dict[str, Any] = {
            "dtype": getattr(torch, config.torch_dtype),
            "revision": config.base_revision,
        }
        if device_map:
            kwargs["device_map"] = device_map
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        torso: Any
        if base_cfg.model_type == GEMMA:
            # Without `audio`, `audio_config` is unset, so the model is built without an
            # audio encoder and the checkpoint's audio weights are left unread.
            if not audio:
                base_cfg.audio_config = None
            full: Any = transformers.Gemma4ForConditionalGeneration.from_pretrained(
                config.base_model, config=base_cfg, **kwargs
            )
            torso = full.model  # .vision_tower, .embed_vision, [.audio_tower, .embed_audio], .language_model
            frozen = [torso.vision_tower, torso.embed_vision]
            if audio:
                frozen += [torso.audio_tower, torso.embed_audio]
            if config.host_embeddings:
                table = getattr(torso.language_model, "embed_tokens_per_layer", None)
                if table is not None:
                    torso.language_model.embed_tokens_per_layer = HostEmbedding(table)
        else:
            full = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
                config.base_model, **kwargs
            )
            torso = full.model  # .visual (ViT + merger) and .language_model
            frozen = [torso.visual]
        torso.config.use_cache = True
        for module in frozen:
            for p in module.parameters():
                p.requires_grad_(False)
        loaded: nn.Module = torso
        return loaded

    def attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        cfg = self.config
        # PEFT full-matches a string `target_modules` against each module name. Scoped
        # to the language model, so the encoders' own projections are never adapted.
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
        position_ids: torch.Tensor | None = None,
        **mm: torch.Tensor | None,
    ) -> torch.Tensor:
        media = {k: v for k, v in mm.items() if v is not None} or self._mm
        extra: dict[str, Any] = dict(media)
        if "pixel_values" in media:
            extra["mm_token_type_ids"] = mm_token_type_ids(self.torso, input_ids)
        if position_ids is not None:
            extra["position_ids"] = position_ids
        out = self.torso(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
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
        **mm: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        self._mm = {k: v for k, v in mm.items() if v is not None}
        try:
            return super().forward(
                input_ids, attention_mask, n_slots,
                labels=labels, label_dist=label_dist, weights=weights,
                past_key_values=past_key_values, temperature=temperature, opt_idx=opt_idx,
            )
        finally:
            self._mm = {}

    def frozen_slot_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_slots: torch.Tensor,
        **mm: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The untouched torso's option-number reading, with the media in the prompt."""
        self._mm = {k: v for k, v in mm.items() if v is not None}
        try:
            return super().frozen_slot_log_probs(input_ids, attention_mask, n_slots)
        finally:
            self._mm = {}

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device_map: str | None = None,
        attn_implementation: str | None = None,
        audio: bool = False,
    ) -> VisionDeciderModel:
        """Load a Strands Decider checkpoint (e.g. v19) onto the multimodal torso; with
        `audio`, a Gemma 4 E2B / E4B checkpoint keeps its audio encoder too.

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
        config.base_revision = base_revision(path, config)
        if config.head_type != "pointer":
            raise ValueError("image input needs a pointer-head checkpoint")
        lora_file = os.path.join(path, "lora", "adapter_model.safetensors")
        if config.use_lora and not os.path.exists(lora_file):
            raise FileNotFoundError(f"{lora_file}: missing, but the checkpoint config sets use_lora")
        head_state = load_head_state(path)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if config.force_bos:  # saved with the tokenizer; this only guards a lossy save
            ensure_bos(tok)

        torso: Any = cls._load_torso(config, device_map, attn_implementation, audio=audio)
        # As StrandsDeciderModel.load: Gemma 4 E2B is served with eager attention on CUDA.
        serving = serving_attn_implementation(torso.config.get_text_config())
        if attn_implementation is None and serving:
            torso.set_attn_implementation(serving)
        model = cls(config, torso, tok)
        cfg: Any = model.torso.config
        markers = [image_tokens(model.family)[1]] + ([audio_tokens(model.family)[1]] if audio else [])
        for marker, want in zip(markers, [cfg.image_token_id, getattr(cfg, "audio_token_id", None)], strict=False):
            got = tok.convert_tokens_to_ids(marker)
            if got != want:
                raise ValueError(
                    f"the checkpoint's tokeniser maps {marker} to {got}, but the base expects {want}"
                )
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
            for p in model.torso.parameters():  # served, as StrandsDeciderModel.load's is_trainable=False
                p.requires_grad_(False)
        model.head.load_state_dict(head_state)
        model.head.to(torch.float32)
        model.eval()
        return model


# ---- engine ------------------------------------------------------------------------


@dataclass
class VisionEngineConfig(EngineConfig):
    # Longest image side after resizing, fixed per deployment. None: the family's default
    # (DEFAULT_LONG_SIDE: 448 px for Qwen, where a 448 px square is 196 tokens; none for
    # Gemma 4, whose processor fixes the token count itself).
    image_long_side: int | None = None
    max_images: int = 4
    # Refused before decoding: 4096 x 4096 covers a 12 MP phone photo and bounds memory.
    max_image_pixels: int = 4096 * 4096
    # Audio (Gemma 4 E2B / E4B with `--audio`): clips per request, and seconds per clip
    # (30 s, Gemma 4's own cap, is 750 tokens).
    max_audio: int = 2
    max_audio_seconds: float = 30.0


class VisionEngine(SystemOneEngine):
    """SystemOneEngine that also answers over images and audio. Text-only requests take
    the unchanged text path."""

    def __init__(
        self,
        model: VisionDeciderModel,
        config: VisionEngineConfig | None = None,
        image_processor: Any = None,
        feature_extractor: Any = None,
    ):
        super().__init__(model, config or VisionEngineConfig())
        self.vcfg: VisionEngineConfig = (
            self.cfg if isinstance(self.cfg, VisionEngineConfig) else VisionEngineConfig()
        )
        if image_processor is None:
            # Pinned to the PIL backend (Qwen2VLImageProcessorPil, Gemma4ImageProcessorPil);
            # see load_image_processor.
            image_processor = load_image_processor(model.config)
        self.image_processor = image_processor
        if feature_extractor is None and model.has_audio:
            from .audio import load_feature_extractor

            feature_extractor = load_feature_extractor(model.config.base_model, model.config.base_revision)
        self.feature_extractor = feature_extractor
        self.long_side = (self.vcfg.image_long_side if self.vcfg.image_long_side is not None
                          else DEFAULT_LONG_SIDE[model.family])

    def _reset_positions(self) -> None:
        # The text path leaves positions to transformers, which would read a
        # `rope_deltas` stored on a Qwen torso by any earlier plain image forward (a
        # training or evaluation step on the same model). The image path below passes
        # positions explicitly and stores none. Gemma 4 stores none.
        base: Any = mm_base(self.model.torso)
        if hasattr(base, "rope_deltas"):
            base.rope_deltas = None

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        self._reset_positions()
        if not request.images and not request.audio:
            return super().evaluate(request)
        if len(request.images) > self.vcfg.max_images:
            raise ValueError(f"{len(request.images)} images; this server takes at most {self.vcfg.max_images}")
        clips: list[Any] = []
        if request.audio:
            if self.feature_extractor is None:
                raise ValueError("this engine has no audio encoder; serve a Gemma 4 E2B / E4B checkpoint "
                                 "with --audio for audio")
            if len(request.audio) > self.vcfg.max_audio:
                raise ValueError(f"{len(request.audio)} audio clips; this server takes at most {self.vcfg.max_audio}")
            from .audio import decode_audio

            clips = [decode_audio(b, self.vcfg.max_audio_seconds) for b in request.audio]
        images = [fit_image(decode_image(b, self.vcfg.max_image_pixels), self.long_side)
                  for b in request.images]
        names = list(request.questions)
        rendered = [render_question(request.questions[n]) for n in names]
        answers: dict[str, Answer] = {}
        total = 0
        for start in range(0, len(names), self.cfg.max_batch):
            chunk = rendered[start : start + self.cfg.max_batch]
            try:
                probs, ntok = self._image_probs(request.state, images, chunk, clips)
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
        self, state: Content, questions: dict[str, Question], images: Sequence[str],
        audio: Sequence[str] = (),
    ) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions, images=list(images),
                                              audio=list(audio)))

    def media_prompt(
        self, state: Content, images: Sequence[Image.Image], clips: Sequence[Any] = ()
    ) -> tuple[str, dict[str, torch.Tensor]]:
        """The state with every placeholder expanded, and the encoder inputs it needs."""
        family = cast(VisionDeciderModel, self.model).family
        mm: dict[str, torch.Tensor] = {}
        image_counts: list[int] = []
        if images:
            image_counts, mm = process_images(self.image_processor, images)
        audio_counts: list[int] = []
        if clips:
            from .audio import process_audio

            audio_counts, amm = process_audio(self.feature_extractor, clips)
            mm.update(amm)
        text = render_image_state(state, len(images), family, n_audio=len(clips))
        if images:
            text = expand_placeholders(text, image_tokens(family)[1], image_counts)
        if clips:
            text = expand_placeholders(text, audio_tokens(family)[1], audio_counts)
        dtype = next(p.dtype for p in self.model.torso.parameters())
        moved = {}
        for k, v in mm.items():
            # Gemma 4's encoders take their inputs in the model's dtype (as the processor's
            # `.to(model.dtype)` gives them); Qwen's ViT casts float32 pixels itself.
            cast_to = dtype if family == GEMMA and v.is_floating_point() else None
            moved[k] = v.to(self.device, dtype=cast_to) if cast_to else v.to(self.device)
        return text, moved

    def _fit_images(
        self, state: Content, images: list[Image.Image], question_texts: list[str],
        clips: Sequence[Any] = (),
    ) -> tuple[list[int], list[list[int]], dict[str, torch.Tensor]]:
        """Question reserve first, then the media block whole, then state text.

        As in `_fit`: with `strict_window` an over-long prompt is refused (HTTP 422);
        otherwise the questions keep their reserve and the state text loses its end, the
        same side `_fit` and training cut. Images and audio are never cut.
        """
        max_len = self.model.config.max_length
        enc = self.tok(question_texts, add_special_tokens=False, return_offsets_mapping=True)
        q, offs = enc["input_ids"], enc["offset_mapping"]
        longest = max(len(x) for x in q)

        family = cast(VisionDeciderModel, self.model).family
        text, mm = self.media_prompt(state, images, clips)
        s = self.tok(text, add_special_tokens=True)["input_ids"]
        ends = {self.tok.convert_tokens_to_ids(image_tokens(family)[2])}
        if clips:
            ends.add(self.tok.convert_tokens_to_ids(audio_tokens(family)[2]))
        keep = max(i for i, t in enumerate(s) if t in ends) + 1  # through the last image or clip

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
                f"the images and audio take {keep} tokens of a {budget}-token state budget; "
                "send fewer or smaller images, or shorter clips"
            )
        s = s[:budget]
        return s, q, mm

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def _image_probs(
        self, state: Content, images: list[Image.Image], rendered: list[RenderedQuestion],
        clips: Sequence[Any] = (),
    ) -> tuple[torch.Tensor, int]:
        m = len(rendered)
        s, q, mm = self._fit_images(state, images, [rq.text for rq in rendered], clips)
        prefix_ids = torch.tensor([s], device=self.device)
        n = prefix_ids.size(1)
        tt = mm_token_type_ids(self.model.torso, prefix_ids)
        model = cast(VisionDeciderModel, self.model)
        suffix_ids, suffix_mask = self._pad(q)
        prefix_pos: torch.Tensor | None = None
        suffix_pos: torch.Tensor | None = None
        if model.family == QWEN:
            base: Any = mm_base(self.model.torso)
            mpos, delta_t = base.get_rope_index(prefix_ids, tt, image_grid_thw=mm["image_grid_thw"])
            delta = int(delta_t.view(-1)[0])
            text_pos = torch.arange(n, device=self.device).view(1, 1, -1)
            prefix_pos = torch.cat([text_pos, mpos], dim=0)  # [4, 1, n]: text + 3 M-RoPE rows
            st = (torch.arange(suffix_ids.size(1), device=self.device) + n).view(1, 1, -1).expand(1, m, -1)
            suffix_pos = torch.cat([st, (st + delta).expand(3, m, -1)], dim=0)
        # Gemma 4: 1-D RoPE, one position per token; the defaults (0..n-1 for the prefix,
        # n.. for the suffix, from the cache's length) are the full forward's positions.
        prefix_out = self.model.torso(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            mm_token_type_ids=tt,
            use_cache=True,
            return_dict=True,
            **({"position_ids": prefix_pos} if prefix_pos is not None else {}),
            **mm,
        )
        cache = prefix_out.past_key_values if m == 1 else _expand_cache(prefix_out.past_key_values, m)

        full_mask = torch.cat(
            [torch.ones(m, n, dtype=suffix_mask.dtype, device=self.device), suffix_mask], dim=1
        )
        hidden = model.encode(suffix_ids, full_mask, past_key_values=cache, position_ids=suffix_pos)
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
    image_long_side: int | None = None,
    audio: bool = False,
) -> VisionEngine:
    """A vision engine configured as `create_app` configures a text one (device, prefix
    cache, `strict_window`, `max_batch`, model name), plus the image settings; with
    `audio`, a Gemma 4 E2B / E4B checkpoint also takes audio clips."""
    base = config or EngineConfig()
    names = {f.name for f in fields(EngineConfig)}
    cfg = VisionEngineConfig(**{k: v for k, v in asdict(base).items() if k in names},
                             image_long_side=image_long_side)
    model = VisionDeciderModel.load(checkpoint, attn_implementation=attn_implementation, audio=audio)
    return VisionEngine(model, cfg)
