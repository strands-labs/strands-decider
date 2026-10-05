"""Eyes for a text-only torso: a frozen SigLIP2 encoder and a trained projector, LLaVA-style.

MiniCPM5-2B is a plain Llama decoder with no vision tower of its own. Here one is
grafted on:

    image -> [SigLIP2 so400m, frozen] -> 24x24 patch features (1152-d)
          -> 2x2 pixel-unshuffle -> 12x12 = 144 features (4608-d)
          -> [projector: Linear, GELU, Linear] -> 144 vectors of the decoder's width

and the 144 vectors replace the input embeddings of 144 slot tokens inside `<state>`:

    <state>\\n<image>SLOT x 144</image>\\n...state text...\\n</state>\\n

The markers are plain text (`<image>` reads as `<`, `image`, `>` in MiniCPM5's
vocabulary, as `<state>` does); the slot is `<unused_token_0>`, a single token whose own
embedding is never used. Positions stay 1-D: an image is 144 ordinary positions. So the
text path is the text checkpoint unchanged, and the shared-prefix cache works as it does
for text: the prefix (images and state) is forwarded once through `inputs_embeds`, and
the question suffixes are plain token ids against its cache.

Two stages train it (training/recipe_minicpm_vision.sh):

1. alignment (graft_align.py): only the projector, on COCO train2014 captions, as
   next-token loss of the frozen base LM. Writes `projector.safetensors` and
   `projector_config.json`.
2. decider (vision_train.py with `projector_from`): a text checkpoint's LoRA and head
   plus the projector, on the image rows of data/image/. The checkpoint holds the
   projector beside the adapter; the encoder is read from the Hub at its pinned commit.

The encoder is fixed-resolution: every image is resized to 384 x 384 (aspect not kept)
by SigLIP's own processor, pinned to its PIL backend, after `read_image`/`fit_image`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from .infer import _expand_cache
from .modeling import StrandsDeciderModel, checkpoint_dir, config_path
from .prompting import RenderedQuestion
from .schema import Content
from .vision import ImagePrompt, VisionEngine

if TYPE_CHECKING:
    from PIL import Image

PROJECTOR_CONFIG = "projector_config.json"
PROJECTOR_WEIGHTS = "projector.safetensors"

# google/siglip2-so400m-patch16-384: Apache-2.0 (Hugging Face card and API), fixed
# resolution, patch 16 -> a 24 x 24 grid of 1152-d features.
SIGLIP2_SO400M = "google/siglip2-so400m-patch16-384"
SIGLIP2_SO400M_REV = "dd658faac399427308559e2c3ac1e99cbe43845d"


@dataclass
class ProjectorConfig:
    encoder: str = SIGLIP2_SO400M
    encoder_revision: str | None = SIGLIP2_SO400M_REV
    encoder_dtype: str = "bfloat16"
    vision_hidden: int = 1152
    text_hidden: int = 2048
    image_size: int = 384
    patch_size: int = 16
    # k x k neighbouring patches are concatenated into one slot: k*k fewer tokens per image
    unshuffle: int = 2
    image_open: str = "<image>"
    image_slot: str = "<unused_token_0>"
    image_close: str = "</image>"

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size or self.grid % self.unshuffle:
            raise ValueError(f"a {self.image_size}px image in {self.patch_size}px patches does not "
                             f"unshuffle by {self.unshuffle}")

    @property
    def grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def tokens_per_image(self) -> int:
        return (self.grid // self.unshuffle) ** 2

    def save(self, path: str) -> None:
        with open(os.path.join(path, PROJECTOR_CONFIG), "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> ProjectorConfig:
        with open(os.path.join(path, PROJECTOR_CONFIG), encoding="utf-8") as fh:
            return cls(**json.load(fh))


class Projector(nn.Module):
    """Patch features [n, grid*grid, dv] -> slot embeddings [n, tokens_per_image, dt].

    The pixel-unshuffle keeps every feature (concatenated, not averaged) while cutting
    the token count by unshuffle**2. Kept in fp32, as the head is.
    """

    def __init__(self, cfg: ProjectorConfig):
        super().__init__()
        self.grid, self.k = cfg.grid, cfg.unshuffle
        self.mlp = nn.Sequential(
            nn.Linear(cfg.vision_hidden * cfg.unshuffle ** 2, cfg.text_hidden),
            nn.GELU(),
            nn.Linear(cfg.text_hidden, cfg.text_hidden),
        )

    def unshuffle(self, feats: torch.Tensor) -> torch.Tensor:
        n, p, d = feats.shape
        g, k = self.grid, self.k
        if p != g * g:
            raise ValueError(f"expected {g * g} patch features per image, got {p}")
        x = feats.reshape(n, g // k, k, g // k, k, d).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(n, (g // k) ** 2, k * k * d)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.mlp(self.unshuffle(feats.to(torch.float32)))
        return out


def load_encoder(cfg: ProjectorConfig, dtype: str | None = None) -> nn.Module:
    """The SigLIP vision tower at its pinned commit, without the pooling head, frozen.

    Reads the vision half of a full SigLIP(2) checkpoint; the text tower's weights are
    left unread.
    """
    import transformers

    rev = {"revision": cfg.encoder_revision} if cfg.encoder_revision else {}
    vcfg = transformers.SiglipVisionConfig.from_pretrained(cfg.encoder, **rev)
    vcfg.vision_use_head = False
    got = (vcfg.hidden_size, vcfg.image_size, vcfg.patch_size)
    if got != (cfg.vision_hidden, cfg.image_size, cfg.patch_size):
        raise ValueError(f"{cfg.encoder}: hidden/image/patch {got}, the projector expects "
                         f"{(cfg.vision_hidden, cfg.image_size, cfg.patch_size)}")
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()  # the text tower's keys, reported unread
    try:
        enc = transformers.SiglipVisionModel.from_pretrained(
            cfg.encoder, config=vcfg, dtype=getattr(torch, dtype or cfg.encoder_dtype), **rev)
    finally:
        transformers.logging.set_verbosity(verbosity)
    for p in enc.parameters():
        p.requires_grad_(False)
    loaded: nn.Module = enc.eval()
    return loaded


def load_image_processor(cfg: ProjectorConfig) -> Any:
    """SigLIP's image processor at the encoder's commit, pinned to the PIL backend (as
    `VisionDeciderModel.image_prompt` pins Qwen's)."""
    from transformers import SiglipImageProcessorPil

    rev = {"revision": cfg.encoder_revision} if cfg.encoder_revision else {}
    return SiglipImageProcessorPil.from_pretrained(cfg.encoder, **rev)


class SiglipImages(ImagePrompt):
    """A fixed-resolution SigLIP encoder's images: `tokens_per_image` slots each."""

    def __init__(self, processor: Any, cfg: ProjectorConfig):
        super().__init__(processor)
        self.cfg = cfg
        self.open_, self.slot, self.close = cfg.image_open, cfg.image_slot, cfg.image_close

    def process(self, images: Sequence[Image.Image]) -> tuple[list[int], dict[str, torch.Tensor]]:
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"]
        if tuple(pixels.shape[1:]) != (3, self.cfg.image_size, self.cfg.image_size):
            raise ValueError(f"processor gave {tuple(pixels.shape)}, expected "
                             f"{self.cfg.image_size}px square images")
        return [self.cfg.tokens_per_image] * len(images), {"pixel_values": pixels}


def slot_token_id(tokenizer: Any, slot: str) -> int:
    """The slot's token id; refuses a vocabulary where it is not one token."""
    ids = tokenizer.encode(slot, add_special_tokens=False)
    if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
        raise ValueError(f"{slot!r} is not a single token in this vocabulary ({ids})")
    return int(ids[0])


def encode_images(encoder: nn.Module, projector: Projector, pixel_values: torch.Tensor) -> torch.Tensor:
    """Images [n, 3, H, W] -> slot embeddings [n, tokens_per_image, dt], fp32. The encoder
    is frozen and runs without a graph; gradients reach the projector only."""
    first = next(encoder.parameters())
    with torch.no_grad():
        feats = encoder(pixel_values=pixel_values.to(first.device, first.dtype)).last_hidden_state
    return cast(torch.Tensor, projector(feats))


def place_images(embeds: torch.Tensor, input_ids: torch.Tensor, slot_id: int,
                 images: torch.Tensor | None) -> torch.Tensor:
    """`embeds` with the image slots' rows replaced, in order, by `images` [n, t, d].

    Not in place, so a graph through `images` survives. Refuses a count mismatch: a slot
    without an image (or an image without its slots) would read garbage.
    """
    mask = input_ids == slot_id
    have = int(mask.sum())
    want = 0 if images is None else images.shape[0] * images.shape[1]
    if have != want:
        raise ValueError(f"the prompt holds {have} image slots, the images fill {want}")
    if images is None:
        return embeds
    return embeds.masked_scatter(mask.unsqueeze(-1), images.reshape(-1, images.shape[-1]).to(embeds.dtype))


class GraftedDeciderModel(StrandsDeciderModel):
    """A Strands Decider on a text-only torso, with a frozen encoder and a projector.

    Text forwards (no `pixel_values`) are the parent's, token for token. Image forwards
    embed the prompt, place the projected images at the slots and forward
    `inputs_embeds`. The torso, its adapter and head load exactly as a text checkpoint's
    (`StrandsDeciderModel.load`); the encoder and projector sit beside the torso, so LoRA
    never reaches them.
    """

    _pixels: torch.Tensor | None = None

    def __init__(self, config: Any, torso: nn.Module, tokenizer: Any, encoder: nn.Module,
                 pcfg: ProjectorConfig):
        super().__init__(config, torso, tokenizer)
        self.attach_eyes(encoder, pcfg)

    def attach_eyes(self, encoder: nn.Module, pcfg: ProjectorConfig) -> None:
        if pcfg.text_hidden != self.hidden_size(self.torso):
            raise ValueError(f"projector width {pcfg.text_hidden} != torso width {self.hidden_size(self.torso)}")
        if self.config.head_type != "pointer":
            raise ValueError("image input needs a pointer-head checkpoint")
        self.pcfg = pcfg
        self.slot_id = slot_token_id(self.tokenizer, pcfg.image_slot)
        self.encoder = encoder
        self.projector = Projector(pcfg).to(torch.float32)

    def train(self, mode: bool = True) -> GraftedDeciderModel:
        super().train(mode)
        self.encoder.eval()  # frozen: never in training mode
        return self

    # ---- the vision-model interface (see vision.load_vision_model) ----------------

    def image_prompt(self, processor: Any = None) -> SiglipImages:
        return SiglipImages(processor if processor is not None else load_image_processor(self.pcfg), self.pcfg)

    def reset_positions(self) -> None:
        """1-D positions: nothing is stored between forwards."""

    def embed(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None) -> torch.Tensor:
        """Input embeddings of `input_ids`, the image slots filled from `pixel_values`."""
        embeds = self.torso.get_input_embeddings()(input_ids)
        images = None if pixel_values is None else encode_images(self.encoder, self.projector, pixel_values)
        return place_images(embeds, input_ids, self.slot_id, images)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pixel_values is None:
            pixel_values = self._pixels
        if pixel_values is None:
            return super().encode(input_ids, attention_mask, past_key_values=past_key_values)
        out = self.torso(
            inputs_embeds=self.embed(input_ids, pixel_values),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
            return_dict=True,
        )
        hidden: torch.Tensor = out.last_hidden_state
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_slots: torch.Tensor,
        *args: Any,
        pixel_values: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        self._pixels = pixel_values
        try:
            return super().forward(input_ids, attention_mask, n_slots, *args, **kwargs)
        finally:
            self._pixels = None

    def frozen_slot_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_slots: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The base torso's option-number reading (adapter off), with the image placed by
        the current projector."""
        self._pixels = pixel_values
        try:
            return super().frozen_slot_log_probs(input_ids, attention_mask, n_slots)
        finally:
            self._pixels = None

    # ---- checkpoints -----------------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        super().save_pretrained(path)
        save_projector(self.projector, self.pcfg, path)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        projector_from: str | None = None,
        device_map: str | None = None,
        attn_implementation: str | None = None,
        trainable: bool = False,
    ) -> GraftedDeciderModel:
        """A checkpoint with its projector (`path` holds both), or a text checkpoint plus
        a stage-1 projector directory (`projector_from`) to start stage 2 from. The
        adapter and projector are frozen unless `trainable`; the encoder always is."""
        path = checkpoint_dir(path)
        pdir = checkpoint_dir(projector_from) if projector_from else path
        if not os.path.exists(os.path.join(pdir, PROJECTOR_CONFIG)):
            raise FileNotFoundError(f"{pdir}: no {PROJECTOR_CONFIG}; pass projector_from (a stage-1 output)")
        if not os.path.exists(config_path(path)):
            raise FileNotFoundError(f"{path}: not a Strands Decider checkpoint")
        pcfg = ProjectorConfig.load(pdir)
        text = StrandsDeciderModel.load(path, device_map=device_map,
                                        attn_implementation=attn_implementation, trainable=trainable)
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.config, obj.torso, obj.tokenizer, obj.head = text.config, text.torso, text.tokenizer, text.head
        obj.attach_eyes(load_encoder(pcfg), pcfg)
        obj.projector.load_state_dict(load_projector_state(pdir))
        obj.projector.requires_grad_(trainable)
        obj.eval()
        return obj


def save_projector(projector: Projector, cfg: ProjectorConfig, path: str) -> None:
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    save_file({k: v.detach().to("cpu", torch.float32).contiguous() for k, v in projector.state_dict().items()},
              os.path.join(path, PROJECTOR_WEIGHTS))
    cfg.save(path)


def load_projector_state(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = load_file(os.path.join(path, PROJECTOR_WEIGHTS))
    return state


class GraftedVisionEngine(VisionEngine):
    """VisionEngine for a grafted checkpoint: 1-D positions, images placed by embedding."""

    def _upcast_torso_for_cpu(self) -> None:
        super()._upcast_torso_for_cpu()
        enc = cast(GraftedDeciderModel, self.model).encoder
        first = next(enc.parameters(), None)
        if first is not None and first.dtype in (torch.bfloat16, torch.float16):
            with torch.inference_mode():
                enc.to(torch.float32)

    def _image_probs(
        self, state: Content, images: list[Image.Image], rendered: list[RenderedQuestion]
    ) -> tuple[torch.Tensor, int]:
        with torch.inference_mode():
            m = len(rendered)
            s, q, mm = self._fit_images(state, images, [rq.text for rq in rendered])
            model = cast(GraftedDeciderModel, self.model)
            prefix_ids = torch.tensor([s], device=self.device)
            n = prefix_ids.size(1)
            prefix_out = model.torso(
                inputs_embeds=model.embed(prefix_ids, mm["pixel_values"]),
                attention_mask=torch.ones_like(prefix_ids),
                use_cache=True,
                return_dict=True,
            )
            cache = prefix_out.past_key_values if m == 1 else _expand_cache(prefix_out.past_key_values, m)
            suffix_ids, suffix_mask = self._pad(q)
            full_mask = torch.cat(
                [torch.ones(m, n, dtype=suffix_mask.dtype, device=self.device), suffix_mask], dim=1
            )
            # The suffixes are text: plain ids against the cache, positions continuing from n.
            hidden = model.encode(suffix_ids, full_mask, past_key_values=cache)
            return self._readout(hidden, full_mask, rendered), n + int(suffix_mask.sum().item())
