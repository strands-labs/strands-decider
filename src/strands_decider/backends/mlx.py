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
  prefix_bidir, with no window anywhere. With `reuse_state=False` (`--no-prefix-cache`) each
  question is one full pass under the dense mask instead, the reference's computation.
- KV sharing. The last layers of E2B and E4B compute no keys or values; they read those of an
  earlier layer. Through them no token reads another token's hidden state, so only the last
  token's path reaches the answer, and a question carries only its last token through them
  (mlx-vlm's `logits_to_keep=1`). That is exact. The other sizes share nothing, and run every
  layer over every question token.
- Images. Each image is decoded as v1 decodes it, its decoded, oriented size held to its
  header's (the core's `check_decoded_size`, which gives the size to resize to), resized by the
  spec's bicubic resize (the core's `resize_image`: the width's pass, then the height's), and
  prepared by the checkpoint's image processor with its own resize off. The vision encoder's N
  vectors replace the placeholders' embeddings at the image's span in the state. At those positions the per-layer embedding of E2B and E4B is token 0's, as in
  the base model, which mlx-vlm's `get_input_embeddings` does.
- The readout. `forward` returns the final-norm hidden state at each question's last token, as
  the bf16 decoder gives it (`outputs = "hidden"`), and the core's `Readout` applies the bf16
  GEMM, the soft-cap, the temperature and the softmax in the spec's precision regime.

`ask` and `serve` reach this backend through `strands_decider.decision`, which registers
`load_backend` for `--device mlx` and puts the core and the request checks around it. Audio is
not part of the interface yet (spec section 12): the core refuses it, and the audio encoder and
its projection are dropped at load and never run.

mlx-vlm is pinned to the release tested (the `mlx` extra), and another is refused at load:
this module calls its internals, `get_input_embeddings`, the language model's `__call__`
(`inputs_embeds`, `per_layer_inputs`, `mask`, `cache`, `logits_to_keep`), `encode_image`,
`get_image_features` and its processors.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import threading
import warnings
from collections.abc import Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    from mlx.utils import tree_map_with_path
except ImportError as e:  # pragma: no cover - depends on the platform
    raise ImportError(
        "the MLX backend needs macOS on Apple silicon, Python 3.11 or later, and the mlx extra: "
        "pip install 'strands-decider[mlx]' (on another platform or Python the extra installs "
        "nothing)"
    ) from e

from ..core import (
    HIDDEN,
    DecisionConfig,
    RenderedImage,
    attention_mask,
    check_decoded_size,
    check_image_processor,
    resize_image,
)
from ..core.config import IMAGE, TEXT
from ..core.images import IMAGE_FORMATS
from ..core.request import NO_VISION
from ..refusals import RequestRefused
from .mlx_format import bases_form, check_mlx_vlm, conformance

# Requests whose images are decoded at once, on the callers' threads: one image may decode to
# tens of megabytes. How many requests reach the backend at once, and how long the others
# wait, is the decision engine's (`serve --max-concurrency`, `--queue-timeout`).
DECODE_SLOTS = 2

# MLX keeps freed buffers for reuse, by default up to its memory limit (nearly all of RAM); v1's
# MLX engine caps the cache at 1 GiB, and so does this backend.
DEFAULT_CACHE_LIMIT = 1 << 30

# config.json's model_type, which names mlx-vlm's module: Gemma4ForConditionalGeneration (E2B,
# E4B, 26B-A4B, 31B) and Gemma4UnifiedForConditionalGeneration (12B).
MODEL_TYPES = ("gemma4", "gemma4_unified")
UNIFIED = "gemma4_unified"
# A model shard's name in a checkpoint without an index.
_SHARD = re.compile(r"model(-\d+-of-\d+)?\.safetensors")
PROCESSOR_CONFIG = "processor_config.json"
TOKENIZER = "tokenizer.json"
INDEX = "model.safetensors.index.json"
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
    directory, which would take the readout, or any stray file, for a weight file. A
    checkpoint without an index (E2B, E4B and the 12B ship one `model.safetensors`), or whose
    config.json is not in the bases' form (`bases_form`), is therefore loaded through a
    directory of links to its other files and its model shards only (`model.safetensors`, or
    `model-*-of-*.safetensors`), with config.json rewritten in that form. A failure to load
    names the checkpoint, not the directory of links."""
    from mlx_vlm.utils import load_model

    # Absolute, so that the links below point at the files wherever they are resolved from:
    # a relative link target is read against the link's own directory.
    path = Path(path).resolve()
    with open(path / "config.json", encoding="utf-8") as f:
        rewritten = bases_form(json.load(f))
    try:
        if (path / INDEX).exists() and rewritten is None:
            model = load_model(path, lazy=True, strict=True)
        else:
            # The shards stay open once loaded, so the links may go before they are read.
            with tempfile.TemporaryDirectory(prefix="strands-decider-mlx-") as view:
                for entry in os.listdir(path):
                    if entry == "config.json" and rewritten is not None:
                        continue
                    if entry.endswith(".safetensors") and not _SHARD.fullmatch(entry):
                        continue  # the readout, or anything else that is not the model's
                    os.symlink(path / entry, Path(view) / entry)
                if rewritten is not None:
                    text = json.dumps(rewritten)
                    (Path(view) / "config.json").write_text(text, encoding="utf-8")
                model = load_model(Path(view), lazy=True, strict=True)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise ValueError(f"{path}: mlx-vlm cannot load this checkpoint: {e}") from e
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
        raise RequestRefused(f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {e}") from e
    try:
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise RequestRefused(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    return img.convert("RGB")


def image_processor(path: Path, unified: bool) -> tuple[Any | None, str]:
    """The checkpoint's image processor, from its processor_config.json, with its resize off:
    mlx-vlm's `Gemma4ImageProcessor`, or `Gemma4UnifiedImageProcessor` for the 12B. Returns
    (None, why) when the checkpoint has none, or one whose settings give another N (the
    core's `check_image_processor`)."""
    p = path / PROCESSOR_CONFIG
    if not p.exists():
        return None, f"it has no {PROCESSOR_CONFIG}"
    with open(p, encoding="utf-8") as f:
        config = json.load(f)
    try:
        check_image_processor(config)
    except ValueError as e:  # its message names the settings, never a path
        return None, str(e)
    settings = config.get("image_processor", config)
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
    decodes it; its decoded, oriented size held to its header's (`check_decoded_size`, which
    gives the size to resize to); resized by the spec's bicubic resize (`resize_image`: the
    width's pass, then the height's, the same on every Pillow); then prepared by the image
    processor with its own resize off. The processor's count of soft tokens must be the
    prompt's N."""
    pil = decode_image(image.data)
    try:
        width, height = check_decoded_size(image, *pil.size)
    except ValueError as e:  # the file's fault: v1's "image could not be decoded"
        raise RequestRefused(str(e)) from e
    pil = resize_image(pil, width, height)
    data, counts = processor(pil)
    if [int(n) for n in counts] != [image.tokens]:
        raise ValueError(
            f"the image processor gives {list(counts)} soft tokens for a {pil.size[0]}x"
            f"{pil.size[1]} image whose prompt holds {image.tokens}"
        )
    out = {k: v for k, v in data.items() if k != "num_soft_tokens_per_image"}
    # This runs off the worker thread, so it must be numpy: an MLX array here would be MLX
    # work on the caller's thread.
    wrong = {k: type(v).__name__ for k, v in out.items() if not isinstance(v, np.ndarray)}
    if wrong:
        raise TypeError(f"the image processor returned {wrong}, not numpy arrays")
    return out


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
    decoded and prepared, as numpy, on the caller's thread, `decode_slots` requests at a
    time: one image may decode to tens of megabytes. The requests that reach `forward` at once
    queue on the worker; the decision engine bounds how many, and for how long the rest wait
    (`serve --max-concurrency`, `--queue-timeout`). After `close`, queued requests are
    cancelled and `forward` refuses to run."""

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
        image_problem: str | None = None,
        description: dict[str, Any] | None = None,
        decode_slots: int = DECODE_SLOTS,
    ) -> None:
        self.model = model
        self.processor = processor
        self.inputs: tuple[str, ...] = (TEXT, IMAGE) if processor is not None else (TEXT,)
        # Why a checkpoint that lists images cannot take them here; the decision engine
        # refuses image requests with it.
        self.image_problem = image_problem if processor is None else None
        self.share_state = share_state
        self.unified = unified
        self.image_token_id = image_token_id
        self._executor = executor
        self._decoding = threading.BoundedSemaphore(decode_slots)
        self._closed = False
        self._decoder = model.language_model.model
        # The layers that own keys and values; a KV-shared layer reads an earlier one's.
        self._owners = int(self._decoder.first_kv_shared_layer_idx)
        # The decoder's width, which the decision engine checks the readout against.
        self.hidden_size = int(self._decoder.embed_tokens.weight.shape[-1])
        if hasattr(self._decoder.embed_tokens, "scales"):  # quantized: packed weight
            self.hidden_size = int(self._decoder.config.hidden_size)
        self._description = dict(description or {})

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage] = (),
    ) -> np.ndarray:
        if images and IMAGE not in self.inputs:
            raise ValueError(self.image_problem or NO_VISION)
        if not questions:
            raise ValueError("forward needs at least one question")
        for image in images:
            span = state[image.start : image.end]
            if len(span) != image.tokens or any(t != self.image_token_id for t in span):
                raise ValueError(
                    f"state ids [{image.start}, {image.end}) are not an image's placeholders"
                )
        if self._closed:
            raise RuntimeError("the MLX backend is closed")
        if images:
            with self._decoding:
                pixels = [prepare_image(image, self.processor) for image in images]
        else:
            pixels = []
        try:
            future = self._executor.submit(self._forward, state, questions, images, pixels)
        except RuntimeError as e:  # shut down between the check above and here
            raise RuntimeError("the MLX backend is closed") from e
        try:
            return future.result()
        except CancelledError as e:  # queued when the backend closed
            raise RuntimeError("the MLX backend closed before this request ran") from e

    def describe(self) -> dict[str, Any]:
        """What `serve`'s health check reports of the backend, computed once at load: no
        file is read and no MLX array touched when the health check calls it."""
        return {**self._description, "reuse_state": self.share_state}

    def close(self) -> None:
        """Stop the worker thread, cancelling the requests queued on it; `forward` refuses to
        run after this. Calling it again does nothing."""
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

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


DTYPES = ("bfloat16", "float32")


def load_backend(
    path: str | os.PathLike[str],
    *,
    device: str = "mlx",
    dtype: str | None = None,
    reuse_state: bool = True,
    max_batch: int = 32,
    max_batch_tokens: int | None = None,
    cache_limit_bytes: int | None = DEFAULT_CACHE_LIMIT,
) -> MLXBackend:
    """The loader `strands_decider.decision.BACKENDS` names for `--device mlx`: the MLX
    backend for the decision checkpoint or MLX artifact in the local directory `path`.

    mlx-vlm must be the release tested (`mlx_format.TESTED_MLX_VLM`). The image path is
    loaded when the checkpoint's `inputs` hold "image" and it carries an image processor with
    the spec's settings; otherwise the backend takes text only, and says why in
    `image_problem`. The weights run in the dtype they are stored in (bf16 for a decision
    checkpoint), or `dtype` ("bfloat16" or "float32") for every tensor but a quantized
    layer's scales and biases. `reuse_state=False` runs each question as its own full pass.
    Questions run one at a time over the state, so `max_batch` and `max_batch_tokens` do not
    apply. `cache_limit_bytes` sets MLX's process-wide buffer cache limit; None leaves it as
    it is. A quantized artifact whose format has no recorded conformance bound warns so, and
    `describe()` carries its claim."""
    from importlib.metadata import version

    del max_batch, max_batch_tokens  # one question at a time: nothing to batch
    kind_, _, index = str(device).partition(":")
    if kind_ != "mlx" or index not in ("", "0"):
        raise ValueError(f"the MLX backend runs on device 'mlx' (or 'mlx:0'), not {device!r}")
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"--dtype on mlx is one of {', '.join(DTYPES)}, not {dtype!r}")
    mlx_vlm = version("mlx-vlm")
    check_mlx_vlm(mlx_vlm)
    root = Path(path).resolve()
    cfg = DecisionConfig.load(root)
    kind = model_type(root)
    unified = kind == UNIFIED
    processor, why = (None, "") if IMAGE not in cfg.inputs else image_processor(root, unified)
    with open(root / "config.json", encoding="utf-8") as f:
        model_config = json.load(f)
    image_token_id = int(model_config["image_token_id"])
    try:
        claim = conformance(model_config, cfg.provenance.base_model)
    except ValueError as e:
        raise ValueError(f"{root}: {e}") from e
    if claim["notice"]:
        warnings.warn(f"{root}: {claim['notice']}", stacklevel=2)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-backend")

    def build() -> MLXBackend:
        if cache_limit_bytes is not None:
            mx.set_cache_limit(cache_limit_bytes)
        model = load_mlx_model(root, keep=VISION_MODULES if processor is not None else ())
        if dtype is not None:
            target = getattr(mx, dtype)

            def cast(name: str, a: Any) -> Any:
                if name.endswith((".scales", ".biases")) or not mx.issubdtype(a.dtype, mx.floating):
                    return a
                return a.astype(target)

            model.update(tree_map_with_path(cast, model.parameters()))
        mx.eval(model.parameters())
        decoder = model.language_model.model
        stored = str(decoder.norm.weight.dtype).removeprefix("mlx.core.")
        return MLXBackend(
            model,
            processor,
            executor,
            unified=unified,
            share_state=reuse_state,
            image_token_id=image_token_id,
            image_problem=f"images are not supported: {why}" if why else None,
            description={
                "backend": "mlx-vlm",
                "mlx_vlm": mlx_vlm,
                "device": "mlx",
                "dtype": stored,
                "conformance": {k: v for k, v in claim.items() if k != "notice"},
            },
        )

    try:
        return executor.submit(build).result()
    except BaseException:
        executor.shutdown(wait=True)
        raise


load_backend.dtypes = DTYPES  # type: ignore[attr-defined]  # checked by decision.check_dtype
