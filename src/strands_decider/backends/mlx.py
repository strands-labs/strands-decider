"""Decision checkpoints on Apple silicon through mlx-vlm (docs/spec.md, The runtime interface).

The checkpoint is the Gemma 4 base's own multimodal checkpoint with the fine-tuning merged in
(spec section 9), or an MLX artifact converted from it (`strands-decider convert-mlx`,
`mlx_convert.py`). mlx-vlm's Gemma 4 classes (`gemma4` for E2B, E4B, 26B-A4B and 31B,
`gemma4_unified` for the 12B) load either as they are. This module decides what they are given:

- The mask. A state token sees the whole state and no question; a question token sees the
  whole prompt (prefix_bidir), on every layer. mlx-vlm gives every layer the mask its caller
  passes, as it is, and builds its own (causal, a window on sliding layers, Gemma 4's
  bidirectional overlay on image tokens) only when none is passed, so every pass here passes
  one. The state is encoded once, every state token seeing every other, into plain KV caches
  (mlx-vlm's `make_cache` gives sliding layers a rotating window, which would drop state
  keys). Each question then runs over the cache, seeing every key, and the cache is cut back
  to the state. No state token ever reads a question, and no question reads another: exactly
  prefix_bidir, with no window anywhere. With `share_state=False` each question is one full
  pass under the dense mask instead, the reference's computation.
- KV sharing. The last layers of E2B and E4B compute no keys or values; they read those of an
  earlier layer. Through them no token reads another token's hidden state, so only the last
  token's path reaches the answer, and a question carries only its last token through them
  (mlx-vlm's `logits_to_keep=1`). That is exact. The other sizes share nothing, and run every
  layer over every question token.
- Images. Each image is decoded as v1 decodes it, resized with Pillow's bicubic filter to the
  size the spec gives, and prepared by the checkpoint's image processor with its own resize
  off. The vision encoder's N vectors replace the placeholders' embeddings at the image's span
  in the state. At those positions the per-layer embedding of E2B and E4B is token 0's, as in
  the base model, which mlx-vlm's `get_input_embeddings` does.
- The readout. `forward` returns the final-norm hidden state at each question's last token, as
  the bf16 decoder gives it (`outputs = "hidden"`), and the core's `Readout` applies the bf16
  GEMM, the soft-cap, the temperature and the softmax in the spec's precision regime.

Audio is not part of the interface yet (spec section 12), so the audio encoder and its
projection are dropped at load and never run.

mlx-vlm is pinned to the minor version tested (the `mlx` extra): this module calls its
internals, `get_input_embeddings`, the language model's `__call__` (`inputs_embeds`,
`per_layer_inputs`, `mask`, `cache`, `logits_to_keep`), `encode_image`, `get_image_features`
and its processors.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import tempfile
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..core import (
    HIDDEN,
    DecisionConfig,
    Readout,
    RenderedImage,
    attention_mask,
    evaluate,
)
from ..core.config import IMAGE, TEXT
from ..core.images import IMAGE_FORMATS, MAX_SOFT_TOKENS, PATCH_SIZE, POOLING_KERNEL_SIZE
from ..core.readout import READOUT_FILE
from ..core.request import NO_VISION
from . import checkpoint_path

# MLX keeps freed buffers for reuse, by default up to its memory limit (nearly all of RAM); v1's
# MLX engine caps the cache at 1 GiB, and so does this backend.
DEFAULT_CACHE_LIMIT = 1 << 30

# config.json's model_type, which names mlx-vlm's module: Gemma4ForConditionalGeneration (E2B,
# E4B, 26B-A4B, 31B) and Gemma4UnifiedForConditionalGeneration (12B).
MODEL_TYPES = ("gemma4", "gemma4_unified")
UNIFIED = "gemma4_unified"
PROCESSOR_CONFIG = "processor_config.json"
TOKENIZER = "tokenizer.json"
INDEX = "model.safetensors.index.json"
# The image processor settings N depends on (spec section 3.2), as every Gemma 4 base sets them.
IMAGE_SETTINGS = {
    "patch_size": PATCH_SIZE,
    "pooling_kernel_size": POOLING_KERNEL_SIZE,
    "max_soft_tokens": MAX_SOFT_TOKENS,
}
# Modules of the base the runtime never runs: audio is not part of the interface.
AUDIO_MODULES = ("audio_tower", "embed_audio")
VISION_MODULES = ("vision_tower", "vision_embedder", "embed_vision")


# ---- checkpoints -----------------------------------------------------------------------------


def model_type(path: Path) -> str:
    """config.json's model_type, if mlx-vlm's Gemma 4 classes load it; otherwise ValueError."""
    with open(path / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    kind = cfg.get("model_type")
    if kind not in MODEL_TYPES:
        raise ValueError(
            f"{path}: config.json has model_type {kind!r}; a decision checkpoint is the Gemma 4 "
            f"base's multimodal checkpoint ({', '.join(MODEL_TYPES)})"
        )
    return str(kind)


def load_mlx_model(path: Path, *, keep: Sequence[str] = ()) -> Any:
    """mlx-vlm's model for the checkpoint or MLX artifact at `path`, loaded strictly and lazily
    (no tensor is read until it is evaluated). Modules named in AUDIO_MODULES and
    VISION_MODULES are dropped unless `keep` names them.

    mlx-vlm reads the shards its index names or, with no index, every `*.safetensors` in the
    directory, which would take the readout for a weight file. A checkpoint without an index
    (E2B, E4B and the 12B ship one `model.safetensors`) is therefore loaded through a
    directory of links to everything but the readout."""
    from mlx_vlm.utils import load_model

    if (path / INDEX).exists():
        model = load_model(path, lazy=True, strict=True)
    else:
        # The shards stay open once loaded, so the links may go before they are read.
        with tempfile.TemporaryDirectory(prefix="strands-decider-mlx-") as view:
            for entry in os.listdir(path):
                if entry != READOUT_FILE:
                    os.symlink(path / entry, Path(view) / entry)
            model = load_model(Path(view), lazy=True, strict=True)
    for name in AUDIO_MODULES + VISION_MODULES:
        if name not in keep and getattr(model, name, None) is not None:
            setattr(model, name, None)
    return model


# ---- images ----------------------------------------------------------------------------------


def decode_image(data: bytes) -> Any:
    """An image file as v1 decodes it (`vision.decode_image`): its first frame, its EXIF
    orientation applied, transparent areas composited onto white, as an RGB Pillow image. The
    file's format and size were checked from its header when the request was rendered; a
    file that fails here is refused with v1's messages."""
    from PIL import Image, ImageOps

    img: Any
    try:
        img = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
    except Exception as e:
        raise ValueError(f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {e}") from e
    try:
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise ValueError(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    return img.convert("RGB")


def resized_size(width: int, height: int) -> tuple[int, int]:
    """(W, H): the size Gemma 4's image processor resizes a `width` x `height` image to, as
    docs/spec.md section 3.2 gives it, in Python's double precision. Its sides are multiples
    of 48, and N = (H / 48) * (W / 48)."""
    side = PATCH_SIZE * POOLING_KERNEL_SIZE
    f = math.sqrt(MAX_SOFT_TOKENS * POOLING_KERNEL_SIZE**2 * PATCH_SIZE**2 / (height * width))
    h = math.floor(f * height / side) * side
    w = math.floor(f * width / side) * side
    if h == 0:
        h, w = side, min(math.floor(width / height) * side, MAX_SOFT_TOKENS * side)
    elif w == 0:
        w, h = side, min(math.floor(height / width) * side, MAX_SOFT_TOKENS * side)
    return w, h


def image_processor(path: Path, unified: bool) -> tuple[Any | None, str]:
    """The checkpoint's image processor, from its processor_config.json, with its resize off:
    mlx-vlm's `Gemma4ImageProcessor`, or `Gemma4UnifiedImageProcessor` for the 12B. Returns
    (None, why) when the checkpoint has none, or one whose settings give another N."""
    p = path / PROCESSOR_CONFIG
    if not p.exists():
        return None, f"it has no {PROCESSOR_CONFIG}"
    with open(p, encoding="utf-8") as f:
        settings = json.load(f).get("image_processor") or {}
    theirs = {k: settings.get(k) for k in IMAGE_SETTINGS}
    if theirs != IMAGE_SETTINGS:
        return None, f"its image processor has {theirs}, not {IMAGE_SETTINGS}"
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor
    from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
        Gemma4UnifiedImageProcessor,
    )

    processor: Any = Gemma4UnifiedImageProcessor if unified else Gemma4ImageProcessor
    kwargs = {k: v for k, v in settings.items() if k != "image_processor_type"}
    kwargs["do_resize"] = False
    return processor(**kwargs), ""


def prepare_image(image: RenderedImage, processor: Any) -> dict[str, np.ndarray]:
    """One image's pixels for the vision encoder (docs/spec.md section 11.1): decoded as v1
    decodes it, resized with Pillow's bicubic filter to the size of section 3.2, then
    prepared by the image processor with its own resize off. The processor's count of soft
    tokens must be the prompt's N."""
    from PIL import Image

    pil = decode_image(image.data)
    pil = pil.resize(resized_size(*pil.size), Image.Resampling.BICUBIC)
    data, counts = processor(pil)
    if [int(n) for n in counts] != [image.tokens]:
        raise ValueError(
            f"the image processor gives {list(counts)} soft tokens for a {pil.size[0]}x"
            f"{pil.size[1]} image whose prompt holds {image.tokens}"
        )
    return {k: np.asarray(v) for k, v in data.items() if k != "num_soft_tokens_per_image"}


# ---- the backend -----------------------------------------------------------------------------


def _everything(keys: int) -> Any:
    """A mask under which every query sees all `keys` keys, broadcast over the queries. A
    mask must always be passed: without one mlx-vlm builds a causal, windowed one."""
    return mx.ones((1, keys), dtype=mx.bool_)


class MLXBackend:
    """`strands_decider.core.Backend` on mlx-vlm: one row per question, the final-norm hidden
    state at its last token. Build one with `load_backend`.

    Every MLX call runs on one worker thread the backend owns: MLX's arrays and Metal
    streams belong to the thread that made them, and a server calls from a pool. Images are
    decoded and prepared on the caller's thread."""

    outputs = HIDDEN

    def __init__(
        self,
        model: Any,
        processor: Any | None,
        executor: ThreadPoolExecutor,
        *,
        unified: bool,
        share_state: bool = True,
        image_token_id: int,
    ) -> None:
        self.model = model
        self.processor = processor
        self.inputs: tuple[str, ...] = (TEXT, IMAGE) if processor is not None else (TEXT,)
        self.share_state = share_state
        self.unified = unified
        self.image_token_id = image_token_id
        self._executor = executor
        self._decoder = model.language_model.model
        # The layers that own keys and values; a KV-shared layer reads an earlier one's.
        self._owners = int(self._decoder.first_kv_shared_layer_idx)

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage] = (),
    ) -> np.ndarray:
        if images and IMAGE not in self.inputs:
            raise ValueError(NO_VISION)
        if not questions:
            raise ValueError("forward needs at least one question")
        for image in images:
            span = state[image.start : image.end]
            if len(span) != image.tokens or any(t != self.image_token_id for t in span):
                raise ValueError(
                    f"state ids [{image.start}, {image.end}) are not an image's placeholders"
                )
        pixels = [prepare_image(image, self.processor) for image in images]
        return self._executor.submit(self._forward, state, questions, images, pixels).result()

    def close(self) -> None:
        """Stop the worker thread. The backend cannot run after this."""
        self._executor.shutdown(wait=True)

    # -- on the worker thread --

    def _forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage],
        pixels: list[dict[str, np.ndarray]],
    ) -> np.ndarray:
        features = [self._encode(image, px) for image, px in zip(images, pixels, strict=True)]
        if self.share_state:
            rows = self._shared(state, questions, images, features)
        else:
            rows = [self._alone(state, q, images, features) for q in questions]
        return np.stack(rows)

    def _encode(self, image: RenderedImage, pixels: dict[str, np.ndarray]) -> Any:
        """The vision encoder's and its projection's N vectors for one image, in the vision
        encoder's dtype (bf16)."""
        projection = self.model.embed_vision.embedding_projection
        dtype = getattr(projection, "scales", projection.weight).dtype  # quantized: its scales'
        pv = mx.array(pixels["pixel_values"]).astype(dtype)
        if self.unified:
            out = self.model.get_image_features(pv, mx.array(pixels["image_position_ids"]))
        else:
            out = self.model.encode_image(pv)
        out = out.reshape(-1, out.shape[-1])
        if out.shape[0] != image.tokens:
            raise ValueError(
                f"the vision encoder gave {out.shape[0]} vectors for an image of "
                f"{image.tokens} soft tokens"
            )
        return out

    def _embed(self, ids: list[int], images: Sequence[RenderedImage], features: list[Any]) -> Any:
        """Token embeddings and per-layer inputs for `ids`, each image's vectors at its span."""
        f = self.model.get_input_embeddings(input_ids=mx.array([ids], dtype=mx.int32))
        embeds = f.inputs_embeds
        for image, vectors in zip(images, features, strict=True):
            embeds[0, image.start : image.end] = vectors.astype(embeds.dtype)
        return embeds, f.per_layer_inputs

    def _hidden(self, h: Any) -> np.ndarray:
        last = h[0, -1].astype(mx.float32)
        mx.eval(last)
        return np.array(last)

    def _shared(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage],
        features: list[Any],
    ) -> list[np.ndarray]:
        from mlx_vlm.models.cache import KVCache

        s = len(state)
        cache = [KVCache() for _ in range(self._owners)]
        embeds, per_layer = self._embed(state, images, features)
        # The state alone, every token seeing every state token. Only the caches are read:
        # MLX never computes what nothing reads, such as the KV-shared layers over the state.
        self._decoder(
            None, inputs_embeds=embeds, per_layer_inputs=per_layer, mask=_everything(s), cache=cache
        )
        mx.eval([c.state for c in cache])
        rows = []
        for q in questions:
            n = len(q)
            h = self._decoder(
                mx.array([q], dtype=mx.int32),
                mask=_everything(s + n),
                cache=cache,
                logits_to_keep=1,
            )
            rows.append(self._hidden(h))
            for c in cache:  # back to the state, for the next question
                c.trim(n)
        return rows

    def _alone(
        self,
        state: list[int],
        question: list[int],
        images: Sequence[RenderedImage],
        features: list[Any],
    ) -> np.ndarray:
        ids = state + question
        embeds, per_layer = self._embed(ids, images, features)
        mask = mx.array(attention_mask(len(state), len(ids)))
        h = self._decoder(
            None, inputs_embeds=embeds, per_layer_inputs=per_layer, mask=mask, logits_to_keep=1
        )
        return self._hidden(h)


def load_backend(
    checkpoint: str | os.PathLike[str],
    config: DecisionConfig | None = None,
    *,
    share_state: bool = True,
    cache_limit_bytes: int | None = DEFAULT_CACHE_LIMIT,
) -> MLXBackend:
    """The MLX backend for a decision checkpoint or MLX artifact (a local directory or a Hub
    repo id). `config` is its decision_config.json, read from the directory if not given.

    The image path is loaded when the checkpoint's `inputs` hold "image" and it carries a
    vision encoder and an image processor with the spec's settings; otherwise the backend
    takes text only, and image requests are refused. `share_state=False` runs each question
    as its own full pass. `cache_limit_bytes` sets MLX's process-wide buffer cache limit;
    None leaves it as it is."""
    path = checkpoint_path(checkpoint)
    cfg = config if config is not None else DecisionConfig.load(path)
    kind = model_type(path)
    unified = kind == UNIFIED
    processor, why = (None, "") if IMAGE not in cfg.inputs else image_processor(path, unified)
    if IMAGE in cfg.inputs and processor is None:
        warnings.warn(f"{path}: images are not supported: {why}", stacklevel=2)
    with open(path / "config.json", encoding="utf-8") as f:
        image_token_id = int(json.load(f)["image_token_id"])
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-backend")

    def build() -> MLXBackend:
        if cache_limit_bytes is not None:
            mx.set_cache_limit(cache_limit_bytes)
        model = load_mlx_model(path, keep=VISION_MODULES if processor is not None else ())
        mx.eval(model.parameters())
        return MLXBackend(
            model,
            processor,
            executor,
            unified=unified,
            share_state=share_state,
            image_token_id=image_token_id,
        )

    try:
        return executor.submit(build).result()
    except BaseException:
        executor.shutdown(wait=True)
        raise


# ---- serve and ask -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """What `/health` reports of the model (v1's `model.config` fields)."""

    base_model: str
    max_length: int
    temperature: float


@dataclass(frozen=True)
class _Model:
    config: ModelInfo


class MLXEngine:
    """A decision checkpoint on MLX behind v1's engine surface, for `serve` and `ask`:
    `evaluate(SystemOneRequest) -> SystemOneResponse`, `cfg` and `model.config`. Requests
    are answered by `strands_decider.core.evaluate` over an `MLXBackend`. Build one with
    `load_engine`."""

    def __init__(
        self,
        path: Path,
        config: DecisionConfig,
        readout: Readout,
        tokenizer: Any,
        backend: MLXBackend,
        engine_config: Any,
    ) -> None:
        self.path = path
        self.config = config
        self.readout = readout
        self.tokenizer = tokenizer
        self.backend = backend
        self.cfg = engine_config
        self.model = _Model(
            ModelInfo(
                base_model=config.provenance.base_model,
                max_length=config.max_length_validated,
                temperature=config.temperature,
            )
        )

    @property
    def takes_images(self) -> bool:
        return IMAGE in self.backend.inputs

    def answer(self, request: Any) -> dict[str, Any]:
        """One request (a JSON object, or a `SystemOneRequest`), answered as JSON."""
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        return evaluate(
            request,
            self.tokenizer,
            self.readout,
            self.config,
            self.backend,
            model_name=self.cfg.model_name,
        )

    def evaluate(self, request: Any) -> Any:
        from ..schema import SystemOneResponse

        return SystemOneResponse.model_validate(self.answer(request))

    def ask(self, state: Any, questions: dict[str, Any], images: Sequence[str] = ()) -> Any:
        """`state`, `questions` (name to a v1 `Question` or its JSON) and base64 `images`."""
        qs = {n: q.model_dump() if hasattr(q, "model_dump") else q for n, q in questions.items()}
        return self.evaluate({"state": state, "questions": qs, "images": list(images)})

    def warm_up(self) -> None:
        """Answer each question type once, and an image request if the checkpoint takes
        images, so the first request does not wait for MLX to build its kernels. Questions
        run one at a time over the shared state, so one request per type covers every
        number of questions."""
        state = "A warm-up request, answered before the server accepts requests."
        self.evaluate(
            {
                "state": state,
                "questions": {
                    "noul": {"type": "noul", "instructions": "Is this a warm-up?"},
                    "choice": {
                        "type": "choice",
                        "instructions": "Pick one.",
                        "criteria": {"a": None, "b": None},
                    },
                    "score": {
                        "type": "score",
                        "instructions": "How warm?",
                        "criteria": ["cold", "warm"],
                    },
                },
            }
        )
        if self.takes_images:
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "PNG")
            self.ask(
                state,
                {"q": {"type": "noul", "instructions": "Is the image grey?"}},
                [base64.b64encode(buf.getvalue()).decode("ascii")],
            )

    def close(self) -> None:
        self.backend.close()


def load_engine(
    checkpoint: str | os.PathLike[str],
    config: Any | None = None,
    *,
    cache_limit_bytes: int | None = DEFAULT_CACHE_LIMIT,
) -> MLXEngine:
    """`serve` and `ask`'s engine for a decision checkpoint on MLX. `config` is v1's
    `EngineConfig`: its `model_name` names the model in responses, and `use_prefix_cache`
    False runs each question as its own full pass instead of over the shared state."""
    from tokenizers import Tokenizer

    if config is None:
        from ..infer import EngineConfig

        config = EngineConfig(device="mlx")
    path = checkpoint_path(checkpoint)
    decision = DecisionConfig.load(path)
    readout = Readout.load(path, decision)
    tokenizer = Tokenizer.from_file(str(path / TOKENIZER))
    backend = load_backend(
        path,
        decision,
        share_state=bool(getattr(config, "use_prefix_cache", True)),
        cache_limit_bytes=cache_limit_bytes,
    )
    return MLXEngine(path, decision, readout, tokenizer, backend, config)
