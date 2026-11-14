"""The transformers backend: decision checkpoints on PyTorch, on CPU, CUDA or Apple's MPS.

It is the reference runtime of docs/spec.md (section 10), the PyTorch model behind
`strands_decider.core.evaluate`, and implements the core's `Backend`: given a state's token
ids, its images with their spans, and each question's ids, it returns each question's final-norm hidden state at its last token
(`outputs = "hidden"`).

Loading (section 9). A decision checkpoint is the base model's own multimodal checkpoint,
loaded with transformers' class for it: `Gemma4ForConditionalGeneration` for E2B, E4B, 26B-A4B
and 31B, `Gemma4UnifiedForConditionalGeneration` for the 12B. `readout.safetensors` and
`decision_config.json` sit beside it, the latter validated by the core's `DecisionConfig`. The
audio encoder is not built: the interface refuses audio (section 1), and transformers'
audio tower has no SDPA path. A checkpoint made before that layout, a bare `Gemma4TextModel`
whose `decision_config.json` predates `interface` and `inputs`, loads as text only, with a
warning for each missing field. It names no prompt, so it runs on token ids rendered elsewhere;
`strands-decider ask` and `serve` refuse it.

The mask (section 6). Every layer gets the prefix_bidir mask as an explicit boolean mask, in a
dict keyed by layer type. The dict replaces Gemma 4's own mask construction: the causal rule,
the sliding windows, and the bidirectional image blocks of the larger sizes
(`use_bidirectional_attention: "vision"`) all go. The model runs SDPA, which applies the mask
it is given and no window of its own. An image's placeholders are state tokens like any other.

State reuse (section 6). By default a state with several questions is encoded once, its keys
and values kept for every layer, and the questions then run over them at positions S
onwards: the state never attends to a question, so its keys and values are the same for every
question. Questions run in batches (`max_batch`, `max_batch_tokens`), padded on the right,
each one seeing only the state and itself. A single question, and every question with
`reuse_state=False`, is one full prompt in one forward pass, the reference of section 10.
tests/test_backend_transformers.py checks that the two agree.

The readout (section 7.1). Rows are float32, which holds a bf16 decoder's values exactly; the
core's `Readout` does the bf16 GEMM and then the soft-cap and the temperature in float32.

Images (section 3.3). Each file is decoded as v1 decodes it (`vision.decode_image`), its
decoded size checked against its header (`check_decoded_size`), resized to the size the image
processor would choose for the decoded, oriented size by the spec's bicubic resize
(`resize_image`: the width's pass, then the height's), and given to the checkpoint's image
processor with its own resize off. For E2B, E4B, 26B-A4B and 31B that is
`Gemma4ImageProcessorPil`, whose pixels this reproduces wherever Pillow resizes the width
first (always before Pillow 13); the 12B's
`Gemma4UnifiedImageProcessor` has only a torchvision backend, so 12B images need torchvision.
The processor's soft-token count must equal the core's N. The model runs its vision encoder
in the pass that holds the state, and each image's N vectors take the place of its
placeholders; at those positions E2B's and E4B's per-layer embeddings are the pad token's, as
transformers looks them up.

Devices (`DEVICE_DTYPES`). CPU runs float32: bf16 is 3 to 4 times slower on CPUs without bf16
instructions. CUDA runs bf16, the reference precision. MPS runs bf16, which works there.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import json
import os
import threading
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from ..core import (
    CODES,
    CONTROL_TOKEN_IDS,
    HIDDEN,
    NO_VISION,
    DecisionConfig,
    Readout,
    RenderedImage,
    check_decoded_size,
    check_image_processor,
    load_tokenizer,
    resize_image,
)
from ..core.config import FILENAME, IMAGE, TEXT, check_float32_scale
from ..core.images import (
    IMAGE_CLOSE,
    IMAGE_OPEN,
    IMAGE_SOFT,
)
from ..core.readout import MODEL_CONFIG_FILE, READOUT_FILE, load_safetensors
from ..refusals import RequestRefused

if TYPE_CHECKING:
    from PIL import Image
    from tokenizers import Tokenizer

    from ..core import Backend

# The two layouts a checkpoint may have.
MULTIMODAL = "multimodal"  # the base's own multimodal checkpoint: the released layout
DECODER = "decoder"  # a bare Gemma 4 text decoder: the intermediate checkpoints, text only
ARCHITECTURE = "Gemma4ForConditionalGeneration"
UNIFIED_ARCHITECTURE = "Gemma4UnifiedForConditionalGeneration"
MODEL_TYPES = {"gemma4": (MULTIMODAL, False), "gemma4_unified": (MULTIMODAL, True)}
DECODER_MODEL_TYPES = {"gemma4_text": False, "gemma4_unified_text": True}
PROCESSOR_FILE = "processor_config.json"
TOKENIZER_FILE = "tokenizer.json"
# The image processor every Gemma 4 base names in its processor_config.json, by checkpoint class.
IMAGE_PROCESSOR_TYPES = {False: "Gemma4ImageProcessor", True: "Gemma4UnifiedImageProcessor"}
# Gemma 4's media placeholders (the bases' config.json): image, audio and video. A bare
# decoder's config does not carry them.
MEDIA_TOKEN_IDS = (258880, 258881, 258884)

# The dtype each device runs in, unless the caller names one. CPU: float32, since bf16 is 3 to
# 4 times slower on CPUs without bf16 instructions (E2B with an image on 16 EPYC 7R13 cores:
# 1.8 s against 6.5 s for the state). CUDA: bf16, the reference precision. MPS: bf16. On
# random-weight models with Gemma 4's head sizes it runs, SDPA with boolean masks included, and
# lands as far from float32 on CPU as bf16 on CPU does; float32 on MPS matches float32 on CPU
# to 1e-3 (measured with torch 2.15.0 on an M4 Pro; tests/test_backend_transformers.py,
# test_mps_*).
DEVICE_DTYPES = {"cpu": torch.float32, "cuda": torch.bfloat16, "mps": torch.bfloat16}
# The most tokens one question batch may hold: rows x (S + the longest question). Each row
# carries its own copy of the state's keys and values through every layer, so the budget
# bounds that copy.
MAX_BATCH_TOKENS = 65536
# Taken by every backend on MPS around its device work (re-entrant: `forward` holds it while it
# calls `prepare_pixels`).
MPS_TURNS = threading.RLock()


def default_dtype(device: str) -> torch.dtype:
    """The dtype `device` runs in (`DEVICE_DTYPES`)."""
    kind = torch.device(device).type
    if kind not in DEVICE_DTYPES:
        raise ValueError(f"the transformers backend runs on cpu, cuda or mps, not {device!r}")
    return DEVICE_DTYPES[kind]


# ---- the checkpoint -----------------------------------------------------------------------


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _layout(model_config: dict[str, Any], where: str) -> tuple[str, bool]:
    """(layout, unified) from config.json: the base's multimodal class, or a bare decoder."""
    model_type = model_config.get("model_type")
    architectures = model_config.get("architectures") or []
    if model_type in MODEL_TYPES:
        layout, unified = MODEL_TYPES[model_type]
        want = UNIFIED_ARCHITECTURE if unified else ARCHITECTURE
        if architectures != [want]:
            raise ValueError(
                f"{where}: config.json has model_type {model_type!r}, so its architectures "
                f"must be [{want!r}], not {architectures!r}"
            )
        return layout, unified
    if model_type in DECODER_MODEL_TYPES:
        return DECODER, DECODER_MODEL_TYPES[model_type]
    raise ValueError(
        f"{where}: not a Gemma 4 decision checkpoint: config.json has model_type "
        f"{model_type!r}; this backend loads {sorted(MODEL_TYPES)} (the base's multimodal "
        f"checkpoint) and {sorted(DECODER_MODEL_TYPES)} (a bare decoder)"
    )


def _text_softcap(model_config: dict[str, Any]) -> Any:
    for d in (model_config, model_config.get("text_config") or {}):
        if "final_logit_softcapping" in d:
            return d["final_logit_softcapping"]
    return None


def _intermediate(
    dc: dict[str, Any], model_config: dict[str, Any], where: str
) -> tuple[float, float, list[str]]:
    """(temperature, soft-cap, warnings) of a decoder checkpoint's decision_config.json made
    before the schema had `interface`, `inputs`, `prompt` and `provenance`. What the readout
    needs is checked as strictly as `DecisionConfig` checks it; each missing field is a
    warning, and the checkpoint takes text only."""
    notes = [
        f'{where}: {FILENAME} has no "interface" field: an intermediate checkpoint, read as '
        "a strands-decider decoder that takes text only"
    ]
    for key, want in (
        ("format_version", 1),
        ("attention_mode", "prefix_bidir"),
        ("pooling", "last"),
    ):
        value = dc.get(key)
        if value != want or isinstance(value, bool):
            raise ValueError(f"{where}: {FILENAME}: {key} must be {want!r}, not {value!r}")
    if list(dc.get("codes") or []) != list(CODES):
        raise ValueError(f"{where}: {FILENAME}: codes must be the 255 answer codes, A to IV")
    ids = dc.get("token_ids")
    if (
        not isinstance(ids, list)
        or len(ids) != len(CODES)
        or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(f"{where}: {FILENAME}: token_ids must be {len(CODES)} distinct ids")
    temperature = check_float32_scale("temperature", dc.get("temperature"))
    theirs = _text_softcap(model_config)
    ours = dc.get("final_logit_softcapping", theirs)
    if ours is None:
        raise ValueError(
            f"{where}: neither {FILENAME} nor config.json has final_logit_softcapping; the "
            "readout is soft-capped, and is never served without its cap"
        )
    if theirs is not None and theirs != ours:
        raise ValueError(
            f"{where}: config.json has final_logit_softcapping {theirs!r} and {FILENAME} "
            f"has {ours!r}; the checkpoint is inconsistent"
        )
    softcap = check_float32_scale("final_logit_softcapping", ours)
    if "inputs" in dc:
        if dc["inputs"] != [TEXT]:
            raise ValueError(
                f"{where}: {FILENAME} inputs are {dc['inputs']!r}, but the checkpoint is a bare "
                'text decoder with no vision encoder; it takes ["text"]'
            )
    else:
        notes.append(f'{where}: {FILENAME} has no "inputs" field: serving text only')
    return temperature, softcap, notes


@dataclass(frozen=True)
class Checkpoint:
    """A decision checkpoint's directory, read and checked, before any weight is loaded.

    `config` is the validated `decision_config.json`, or None for an intermediate checkpoint,
    which predates the schema; `readout` is `readout.safetensors` with the soft-cap. `inputs`
    is what the checkpoint takes: `config.inputs`, or ("text",) for a bare decoder."""

    path: str
    layout: str
    unified: bool
    config: DecisionConfig | None
    readout: Readout
    temperature: float
    inputs: tuple[str, ...]
    model_config: dict[str, Any] = field(repr=False)
    warnings: tuple[str, ...] = ()

    @classmethod
    def open(cls, checkpoint: str, *, revision: str | None = None) -> Checkpoint:
        """From a local directory, or a Hugging Face Hub repo id (downloaded to the Hub cache)."""
        from ..modeling import checkpoint_dir

        path = checkpoint_dir(checkpoint, revision=revision)
        model_config = _read_json(os.path.join(path, MODEL_CONFIG_FILE))
        layout, unified = _layout(model_config, path)
        dc = _read_json(os.path.join(path, FILENAME))
        if not isinstance(dc, dict):
            raise ValueError(f"{path}: {FILENAME} must hold a JSON object")
        if layout == DECODER and "interface" not in dc:
            temperature, softcap, notes = _intermediate(dc, model_config, path)
            weight = load_safetensors(os.path.join(path, READOUT_FILE))
            if set(weight) != {"weight"} or weight["weight"].shape[0] != len(CODES):
                raise ValueError(
                    f"{path}: {READOUT_FILE} must hold one tensor, `weight` [255, hidden]"
                )
            return cls(
                path, layout, unified, None, Readout(weight["weight"], softcap), temperature,
                (TEXT,), model_config, tuple(notes),
            )  # fmt: skip
        config = DecisionConfig.from_dict(dc)
        if layout == DECODER and IMAGE in config.inputs:
            raise ValueError(
                f'{path}: {FILENAME} inputs name "image", but the checkpoint is a bare text '
                "decoder with no vision encoder; an image checkpoint has the base's multimodal "
                "layout (docs/spec.md, section 9)"
            )
        readout = Readout.load(path, config)
        return cls(
            path, layout, unified, config, readout, config.temperature, config.inputs, model_config
        )

    def tokenizer(self) -> Tokenizer:
        """The checkpoint's tokenizer; for a decision checkpoint checked against the
        interface's control and code ids (the core's `load_tokenizer`). An intermediate
        checkpoint's is read as it is: it runs on ids rendered elsewhere."""
        if self.config is not None:
            return load_tokenizer(self.path, self.config)
        from tokenizers import Tokenizer

        return Tokenizer.from_file(os.path.join(self.path, TOKENIZER_FILE))

    def processor_config(self) -> dict[str, Any] | None:
        p = os.path.join(self.path, PROCESSOR_FILE)
        return _read_json(p) if os.path.exists(p) else None


def image_processor_problem(processor_config: Any, unified: bool) -> str | None:
    """Why images cannot be served with this `processor_config.json`, or None. The soft-token
    count N depends on the processor's patch size, pooling kernel and soft-token budget, so a
    processor with other values would put a different number of vectors where the prompt has
    N placeholders (docs/spec.md, sections 3.3 and 9)."""
    if not isinstance(processor_config, dict):
        return f"the checkpoint has no {PROCESSOR_FILE}, so image requests are refused"
    ip = processor_config.get("image_processor")
    if not isinstance(ip, dict):
        return f"{PROCESSOR_FILE} has no image_processor, so image requests are refused"
    want = IMAGE_PROCESSOR_TYPES[unified]
    if ip.get("image_processor_type") not in (want, want + "Pil"):
        return (
            f"{PROCESSOR_FILE} names image processor {ip.get('image_processor_type')!r}, not "
            f"{want!r}, so image requests are refused"
        )
    try:
        check_image_processor(processor_config)
    except ValueError as e:  # its message names the settings, never a path
        return f"{e}; image requests are refused"
    return None


# ---- pixels -------------------------------------------------------------------------------


class ImageRefused(RequestRefused):
    """An image file the runtime cannot decode: the caller's error, with v1's message."""


def decode(image: RenderedImage) -> Image.Image:
    """The image's file as v1 decodes it (`vision.decode_image`): the first frame, upright by
    its EXIF orientation, transparent areas on white, RGB. A file that fails is refused with
    v1's message (`ImageRefused`)."""
    from ..vision import decode_image

    try:
        return decode_image(base64.b64encode(image.data).decode("ascii"))
    except ValueError as e:
        raise ImageRefused(str(e)) from e


def resize(image: RenderedImage, img: Image.Image) -> Image.Image:
    """The decoded `img` of `image` resized as docs/spec.md section 3.3 says: to the size
    the processor would choose for the decoded, oriented size (which must be the header's or
    its transpose: `check_decoded_size`), by the spec's bicubic resize (`resize_image`, the
    width's pass, then the height's, the same on every Pillow)."""
    try:
        width, height = check_decoded_size(image, *img.size)
    except ValueError as e:
        raise ImageRefused(str(e)) from e
    resized: Image.Image = resize_image(img, width, height)
    return resized


PROCESSOR_FAILED = "the image processor could not be built, so image requests are refused"


def load_image_processor(path: str, unified: bool) -> Any:
    """The checkpoint's image processor from its processor_config.json: Gemma 4's PIL backend,
    or for the 12B its only one, which needs torchvision."""
    import transformers

    if unified:
        if importlib.util.find_spec("torchvision") is None:
            raise ImportError(
                "the 12B's image processor (Gemma4UnifiedImageProcessor) needs torchvision"
            )
        return transformers.Gemma4UnifiedImageProcessor.from_pretrained(path)
    return transformers.Gemma4ImageProcessorPil.from_pretrained(path)


# ---- the state's keys and values ----------------------------------------------------------


class StateCache:
    """A state's keys and values, layer by layer, behind the two methods of a transformers
    cache that Gemma 4's decoders call (`update` and `get_seq_length`).

    In the state pass it records each layer's keys and values and returns them as they are.
    Frozen, it serves question passes: `update` returns the state's keys and values, repeated
    across the batch, followed by the questions', and stores nothing, so one state serves any
    number of question batches. The KV-shared layers of E2B and E4B call nothing: they reuse
    the keys and values that the last layer of their type got from `update`, state and
    question both."""

    def __init__(self) -> None:
        self.layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.length = 0
        self.frozen = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *_: Any,
        **__: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.frozen:
            self.layers[layer_idx] = (key_states, value_states)
            return key_states, value_states
        k, v = self.layers[layer_idx]
        n = key_states.shape[0]
        k, v = k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)
        return torch.cat([k, key_states], dim=-2), torch.cat([v, value_states], dim=-2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def freeze(self, length: int) -> None:
        self.length, self.frozen = length, True


# ---- the backend --------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet_loading() -> Iterator[None]:
    """transformers' load report lists every tensor it does not use, the audio tower's
    hundreds included; the backend checks the loading info itself."""
    from transformers.utils import logging as hf_logging

    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(level)


# Tensors the backend does not build, and so does not read: the audio encoder and its
# projection always, the vision encoder and its projection for a text-only load.
_AUDIO_PREFIXES = ("model.audio_tower.", "model.embed_audio.")
_VISION_PREFIXES = ("model.vision_tower.", "model.vision_embedder.", "model.embed_vision.")


def _load_model(ckpt: Checkpoint, dtype: torch.dtype, *, vision: bool) -> torch.nn.Module:
    import transformers

    if ckpt.layout == DECODER:
        with _quiet_loading():
            loaded = transformers.AutoModel.from_pretrained(
                ckpt.path, dtype=dtype, attn_implementation="sdpa", output_loading_info=True
            )
        expected: tuple[str, ...] = ()
    else:
        conf = transformers.AutoConfig.from_pretrained(ckpt.path)
        conf.audio_config = None
        if not vision:
            conf.vision_config = None
        # Made redundant by prefix_bidir, and bypassed by the mask dict in any case (section 6).
        conf.get_text_config().use_bidirectional_attention = None
        cls = getattr(transformers, UNIFIED_ARCHITECTURE if ckpt.unified else ARCHITECTURE, None)
        if cls is None:
            raise RuntimeError(
                f"transformers {transformers.__version__} has no "
                f"{UNIFIED_ARCHITECTURE if ckpt.unified else ARCHITECTURE}; upgrade it"
            )
        with _quiet_loading():
            loaded = cls.from_pretrained(
                ckpt.path,
                config=conf,
                dtype=dtype,
                attn_implementation="sdpa",
                output_loading_info=True,
            )
        expected = _AUDIO_PREFIXES + (() if vision else _VISION_PREFIXES)
    model, info = loaded
    missing = sorted(info.get("missing_keys") or [])
    if missing:
        raise ValueError(
            f"{ckpt.path}: the checkpoint lacks {len(missing)} of the model's tensors, such as "
            f"{missing[:3]}"
        )
    unexpected = sorted(k for k in info.get("unexpected_keys") or [] if not k.startswith(expected))
    if unexpected:
        warnings.warn(
            f"{ckpt.path}: {len(unexpected)} tensors the model does not use, such as "
            f"{unexpected[:3]}",
            stacklevel=3,
        )
    return cast("torch.nn.Module", model)


def _check_readout_width(ckpt: Checkpoint) -> None:
    """The readout must have one row per code and the decoder's hidden size, read from
    config.json (`text_config.hidden_size`, or the top level for a bare decoder)."""
    cfg = ckpt.model_config
    hidden = (cfg.get("text_config") or cfg).get("hidden_size")
    if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1:
        raise ValueError(f"{ckpt.path}: config.json gives no hidden_size for the decoder")
    weight = ckpt.readout.weight
    shape = None if weight is None else tuple(weight.shape)
    if shape != (len(CODES), hidden):
        raise ValueError(
            f"{ckpt.path}: readout.safetensors `weight` is {shape}, not "
            f"[{len(CODES)}, {hidden}]: the decoder's hidden size"
        )


def _ids(name: str, seq: Sequence[int], vocab: int) -> list[int]:
    ids = [int(t) for t in seq]
    if not ids:
        raise ValueError(f"{name} has no tokens")
    bad = [t for t in ids if not 0 <= t < vocab]
    if bad:
        raise ValueError(f"{name} holds token ids outside the vocabulary of {vocab}: {bad[:3]}")
    return ids


class TransformersBackend:
    """`strands_decider.core.Backend` on a transformers Gemma 4 model (see the module
    docstring). Build one with `load`. `forward` may be called from several threads at once;
    on MPS, where two threads on the device abort the process, calls take turns."""

    outputs = HIDDEN

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint: Checkpoint,
        *,
        device: str,
        dtype: torch.dtype,
        image_processor: Any = None,
        image_problem: str | None = None,
        reuse_state: bool = True,
        max_batch: int = 32,
        max_batch_tokens: int | None = MAX_BATCH_TOKENS,
    ):
        if max_batch < 1 or (max_batch_tokens is not None and max_batch_tokens < 1):
            raise ValueError("max_batch and max_batch_tokens must be positive")
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype
        self.reuse_state = reuse_state
        self.max_batch = max_batch
        self.max_batch_tokens = max_batch_tokens
        # Metal aborts the process when two threads use the device at once, whichever backend
        # they run: one lock for every backend on MPS.
        self._turns: contextlib.AbstractContextManager[Any] = (
            MPS_TURNS if torch.device(device).type == "mps" else contextlib.nullcontext()
        )
        # The decoder, with the vision encoder where there is one: Gemma4Model or
        # Gemma4UnifiedModel inside the multimodal class, or the bare Gemma4TextModel.
        self.decoder: Any = model if checkpoint.layout == DECODER else cast(Any, model).model
        cfg: Any = self.decoder.config
        text: Any = cfg.get_text_config()
        self.layer_types = sorted(set(text.layer_types))
        self.vocab_size = int(text.vocab_size)
        self.pad_id = int(text.pad_token_id or 0)
        self.hidden_size = int(text.hidden_size)
        weight = checkpoint.readout.weight
        if weight is None or weight.shape != (len(CODES), self.hidden_size):
            shape = None if weight is None else weight.shape
            raise ValueError(
                f"{checkpoint.path}: readout.safetensors `weight` is {shape}, not "
                f"[{len(CODES)}, {self.hidden_size}]: the decoder's hidden size"
            )
        if checkpoint.layout == MULTIMODAL:
            self.media_ids = frozenset(
                int(i) for i in (cfg.image_token_id, cfg.audio_token_id, cfg.video_token_id)
            )
            self.image_ids = (int(cfg.boi_token_id), int(cfg.image_token_id), int(cfg.eoi_token_id))
        else:
            self.media_ids = frozenset(MEDIA_TOKEN_IDS)
            self.image_ids = (-1, MEDIA_TOKEN_IDS[0], -1)
        has_vision = getattr(self.decoder, "embed_vision", None) is not None
        self.image_processor = image_processor if has_vision else None
        # Why image requests are refused, None when the backend has an image path; the
        # decision engine refuses them with it before their files are read.
        self.image_problem = (
            None if self.image_processor is not None else (image_problem or NO_VISION)
        )

    @classmethod
    def load(
        cls,
        checkpoint: str | Checkpoint,
        *,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
        images: bool = True,
        reuse_state: bool = True,
        max_batch: int = 32,
        max_batch_tokens: int | None = MAX_BATCH_TOKENS,
        revision: str | None = None,
    ) -> TransformersBackend:
        """Load `checkpoint` (a directory, a Hub repo id, or an opened `Checkpoint`) onto
        `device`, in `dtype` (`DEVICE_DTYPES[device]` when None). The vision encoder is built
        when the checkpoint takes images and `images` is true; a checkpoint whose image
        processor or tokenizer does not fit the spec loads without it, and its image requests
        are refused with the reason (`image_problem`)."""
        ckpt = (
            checkpoint
            if isinstance(checkpoint, Checkpoint)
            else Checkpoint.open(checkpoint, revision=revision)
        )
        for note in ckpt.warnings:
            warnings.warn(note, stacklevel=2)
        _check_readout_width(ckpt)  # before the weights, which may be tens of GB
        default = default_dtype(device)  # checks the device
        dtype = dtype or default
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"the transformers backend runs float32 or bfloat16, not {dtype}")
        processor, problem = None, None
        if images and IMAGE in ckpt.inputs:
            problem = image_processor_problem(ckpt.processor_config(), ckpt.unified)
            if problem is None:
                problem = _image_ids_problem(ckpt)
            if problem is None:
                try:
                    processor = load_image_processor(ckpt.path, ckpt.unified)
                except (ImportError, AttributeError, OSError, ValueError) as e:
                    # A transformers without the class, or an unreadable processor config. The
                    # reason clients get is fixed; the error, which may name local paths, is
                    # only logged.
                    problem = PROCESSOR_FAILED
                    warnings.warn(
                        f"{ckpt.path}: the image processor could not be built: "
                        f"{type(e).__name__}: {e}",
                        stacklevel=2,
                    )
            if problem is not None and problem != PROCESSOR_FAILED:
                warnings.warn(f"{ckpt.path}: {problem}", stacklevel=2)
        model = _load_model(ckpt, dtype, vision=processor is not None)
        model.to(device).eval()
        return cls(
            model,
            ckpt,
            device=device,
            dtype=dtype,
            image_processor=processor,
            image_problem=problem,
            reuse_state=reuse_state,
            max_batch=max_batch,
            max_batch_tokens=max_batch_tokens,
        )

    def describe(self) -> dict[str, Any]:
        """What the backend runs, for logs and health checks."""
        return {
            "backend": "transformers",
            "device": self.device,
            "dtype": str(self.dtype).removeprefix("torch."),
            "layout": self.checkpoint.layout,
            "reuse_state": self.reuse_state,
        }

    # ---- images -------------------------------------------------------------------------

    def prepare_pixels(self, images: Sequence[RenderedImage]) -> dict[str, torch.Tensor]:
        """The vision encoder's inputs for `images` (section 3.3), on the device: each file
        decoded as v1 decodes it, its decoded size checked against its header, resized by the
        spec's bicubic resize (`resize` here, the core's `resize_image`), then through the
        checkpoint's image processor with its own resize off. Checks that the processor counts
        each image's N. The model runs its encoder on them in the pass that holds the state,
        and puts each image's N vectors at its placeholders, in order."""
        if self.image_processor is None:
            raise ValueError(self.image_problem or NO_VISION)
        pixels = [resize(im, decode(im)) for im in images]
        out = self.image_processor(images=pixels, do_resize=False, return_tensors="pt")
        counts = [int(n) for n in torch.as_tensor(out["num_soft_tokens_per_image"]).view(-1)]
        if counts != [im.tokens for im in images]:
            raise RuntimeError(
                f"the image processor gives {counts} soft tokens where the prompt has "
                f"{[im.tokens for im in images]} placeholders"
            )
        with self._turns:
            return {
                "pixel_values": out["pixel_values"].to(self.device, self.dtype),
                "image_position_ids": out["image_position_ids"].to(self.device),
            }

    # ---- the decoder ---------------------------------------------------------------------

    def _check_media(
        self, state: list[int], questions: list[list[int]], images: Sequence[RenderedImage]
    ) -> None:
        """Placeholders only where the images' spans say, and none in a question: the model
        puts the encoder's vectors wherever it finds the image token, in order."""
        boi, soft, eoi = self.image_ids
        expected: list[int] = []
        prev = 0
        for i, im in enumerate(images):
            start, end = im.start, im.end
            if (
                not (prev < start and end < len(state))
                or state[start - 1] != boi
                or state[end] != eoi
            ):
                raise ValueError(
                    f"image {i}: its span [{start}, {end}) is not between <|image> and <image|> "
                    "in the state, after the previous image"
                )
            expected += range(start, end)
            prev = end + 1
        found = [j for j, t in enumerate(state) if t in self.media_ids]
        if found != expected or any(state[j] != soft for j in found):
            raise ValueError(
                "the state's media placeholders are not exactly its images' spans: "
                f"{len(found)} placeholders for {len(expected)} image tokens"
            )
        for i, q in enumerate(questions):
            if any(t in self.media_ids for t in q):
                raise ValueError(f"question {i} holds a media placeholder; images go in the state")

    def _decode(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        cache: StateCache | None = None,
        pixels: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The final-norm hidden states of `ids` [rows, n], every layer under `mask` (boolean,
        [rows, 1, n, cached + n]), at `positions` [rows, n]. `pixels` (`prepare_pixels`) go to
        the pass whose ids hold the images' placeholders."""
        extra = pixels or {}
        out = self.decoder(
            input_ids=ids,
            attention_mask={t: mask for t in self.layer_types},
            position_ids=positions,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
            **extra,
        )
        hidden: torch.Tensor = out.last_hidden_state
        return hidden

    def _full_pass(
        self, state: list[int], question: list[int], pixels: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        """One prompt, one forward pass under prefix_bidir: the reference of section 10."""
        s, n = len(state), len(state) + len(question)
        mask = torch.ones(n, n, dtype=torch.bool, device=self.device)
        mask[:s, s:] = False
        ids = torch.tensor([state + question], device=self.device)
        positions = torch.arange(n, device=self.device).view(1, -1)
        return self._decode(ids, mask.view(1, 1, n, n), positions, None, pixels)[0, -1]

    def _batches(self, s: int, questions: list[list[int]]) -> list[list[int]]:
        """Consecutive runs of question indices: at most `max_batch` rows, and rows x (S + the
        longest question) at most `max_batch_tokens`, one question at least."""
        out: list[list[int]] = []
        cur: list[int] = []
        width = 0
        for i, q in enumerate(questions):
            w = max(width, len(q))
            full = len(cur) >= self.max_batch or (
                self.max_batch_tokens is not None
                and (len(cur) + 1) * (s + w) > self.max_batch_tokens
            )
            if cur and full:
                out.append(cur)
                cur, w = [], len(q)
            cur.append(i)
            width = w
        out.append(cur)
        return out

    def _over_state(
        self, state: list[int], questions: list[list[int]], pixels: dict[str, torch.Tensor] | None
    ) -> list[torch.Tensor]:
        """The state once, then the questions in batches over its keys and values."""
        s = len(state)
        cache = StateCache()
        mask = torch.ones(1, 1, s, s, dtype=torch.bool, device=self.device)
        ids = torch.tensor([state], device=self.device)
        self._decode(ids, mask, torch.arange(s, device=self.device).view(1, -1), cache, pixels)
        cache.freeze(s)
        rows: list[torch.Tensor] = [torch.empty(0)] * len(questions)
        for batch in self._batches(s, questions):
            qs = [questions[i] for i in batch]
            width = max(len(q) for q in qs)
            if width % 64 == 1:
                # torch 2.7's CUDA SDPA got such passes wrong for one key/value head (E2B).
                # One more masked column of padding sidesteps it on any version, at no cost
                # to the answers.
                width += 1
            ids = torch.full((len(qs), width), self.pad_id, dtype=torch.long)
            for r, q in enumerate(qs):
                ids[r, : len(q)] = torch.tensor(q)
            lengths = torch.tensor([len(q) for q in qs], device=self.device)
            # Each row sees the state and its own question, never the padding after it; padding
            # rows see the same, so no row is fully masked.
            cols = torch.arange(s + width, device=self.device)
            keep = cols.view(1, -1) < (s + lengths).view(-1, 1)
            mask = keep[:, None, None, :].expand(len(qs), 1, width, s + width)
            positions = (torch.arange(width, device=self.device) + s).expand(len(qs), -1)
            hidden = self._decode(ids.to(self.device), mask, positions, cache)
            for r, i in enumerate(batch):
                rows[i] = hidden[r, len(questions[i]) - 1]
        return rows

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage] = (),
    ) -> np.ndarray:
        """Each question run after the shared state under prefix_bidir, seeing no other
        question; returns the final-norm hidden state at each question's last token, float32,
        [len(questions), hidden]. `images` are the state's, in order, with their spans."""
        state = _ids("the state", state, self.vocab_size)
        if not questions:
            raise ValueError("forward needs at least one question")
        qs = [_ids(f"question {i}", q, self.vocab_size) for i, q in enumerate(questions)]
        images = tuple(images)
        if images and self.image_processor is None:
            raise ValueError(self.image_problem or NO_VISION)
        self._check_media(state, qs, images)
        # Decoded and resized on the CPU first; only the copy to the device takes turns on MPS.
        # The reference reruns the vision encoder in each prompt's pass, as one prompt per pass
        # does; with the state reused it runs once, in the state's pass.
        pixels = self.prepare_pixels(images) if images else None
        with self._turns, torch.inference_mode():
            # One question gains nothing from a shared state and would pay a second pass (as
            # v1's engine measured; here E2B on an H100 took 324 ms against 174 ms).
            if self.reuse_state and len(qs) > 1:
                rows = self._over_state(state, qs, pixels)
            else:
                rows = [self._full_pass(state, q, pixels) for q in qs]
            out: np.ndarray = torch.stack(rows).float().cpu().numpy()
        return out


def load_backend(
    path: str,
    *,
    device: str,
    dtype: str | None = None,
    reuse_state: bool = True,
    max_batch: int = 32,
    max_batch_tokens: int | None = None,
) -> TransformersBackend:
    """The loader `strands_decider.decision.BACKENDS` names for cpu, cuda and mps: `dtype` is
    one of `load_backend.dtypes`, or None for the device's own."""
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"the transformers backend runs {' or '.join(DTYPES)}, not {dtype!r}")
    return TransformersBackend.load(
        path,
        device=device,
        dtype=getattr(torch, dtype) if dtype else None,
        reuse_state=reuse_state,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens or MAX_BATCH_TOKENS,
    )


# The dtypes a caller may name (`--dtype`), checked before anything is loaded.
DTYPES = ("float32", "bfloat16")
load_backend.dtypes = DTYPES  # type: ignore[attr-defined]


def _image_ids_problem(ckpt: Checkpoint) -> str | None:
    """Why config.json's image token ids are not the interface's, or None: the model replaces
    the ids config.json names, and the prompt's placeholders are the interface's
    (`CONTROL_TOKEN_IDS`, which the checkpoint's tokenizer is held to by `load_tokenizer`)."""
    cfg = ckpt.model_config
    for name, key in (
        (IMAGE_OPEN, "boi_token_id"),
        (IMAGE_SOFT, "image_token_id"),
        (IMAGE_CLOSE, "eoi_token_id"),
    ):
        if cfg.get(key) != CONTROL_TOKEN_IDS[name]:
            return (
                f"config.json's {key} is {cfg.get(key)}, not the interface's id for {name}, "
                f"{CONTROL_TOKEN_IDS[name]}, so image requests are refused"
            )
    return None


if TYPE_CHECKING:

    def _implements_the_core_protocol(backend: TransformersBackend) -> Backend:
        return backend  # mypy checks TransformersBackend against the core's Backend


__all__ = [
    "DECODER",
    "DEVICE_DTYPES",
    "MULTIMODAL",
    "Checkpoint",
    "ImageRefused",
    "StateCache",
    "TransformersBackend",
    "decode",
    "default_dtype",
    "image_processor_problem",
    "load_backend",
    "load_image_processor",
    "resize",
]
