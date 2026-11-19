"""What the image build does to the stock converter's GGUFs: the mmproj without its audio tower,
the check that it preprocesses as the image processor does, and the decoder's lazily read byte
ranges. `gguf` is the llama.cpp tree's own gguf-py, which the caller puts on sys.path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The tensors llama.cpp reads lazily, a few rows per batch: Gemma 4 E2B's and E4B's per-layer
# embedding table, half of E2B's file.
_LAZY_TENSORS = ("per_layer_token_embd.weight",)


def is_audio_tensor(name: str) -> bool:
    return name.startswith(("a.", "mm.a."))


def is_audio_key(key: str) -> bool:
    return key == "clip.has_audio_encoder" or key.startswith("clip.audio.")


def vision_only(src: Path, dst: Path) -> None:
    """The converter's mmproj without its audio tower: the vision tower and its projection byte
    for byte, and every key but the audio ones, `general.size_label` recomputed from the tensors
    left, as the converter computes it. mtmd loads every encoder in the file, and E2B's 0.6 GB
    audio tower does not fit beside the vision tower on Lambda."""
    import gguf

    reader = gguf.GGUFReader(src)
    tensors = [t for t in reader.tensors if not is_audio_tensor(t.name)]
    writer = gguf.GGUFWriter(dst, arch="clip", endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()
    for field in reader.fields.values():
        # GGUFWriter writes the architecture and the GGUF.* header fields itself.
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if is_audio_key(field.name):
            continue
        value = field.contents()
        if field.name == gguf.Keys.General.SIZE_LABEL:
            value = gguf.size_label(sum(int(t.n_elements) for t in tensors), 0, 0, 0)
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, value, kind, sub_type=sub)
    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()


def check_mmproj(mmproj: Path, ckpt: Path) -> None:
    """That the mmproj normalises pixels and cuts patches as the checkpoint's image processor does.
    The converter writes them from the checkpoint, so a llama.cpp pin that writes them otherwise
    shows here, as a ValueError."""
    import gguf

    fields = gguf.GGUFReader(mmproj).fields
    got = [fields[f"clip.vision.{k}"].contents() for k in ("image_mean", "image_std", "patch_size")]
    settings = json.loads((ckpt / "processor_config.json").read_text("utf-8"))
    p = settings.get("image_processor", settings)
    normalise = p.get("do_normalize")
    want = [
        p["image_mean"] if normalise else [0, 0, 0],
        p["image_std"] if normalise else [1, 1, 1],
        p["patch_size"],
    ]
    if not (np.allclose(got[0], want[0]) and np.allclose(got[1], want[1]) and got[2] == want[2]):
        raise ValueError(
            f"the mmproj preprocesses with {got} (mean, std, patch), the processor with {want}"
        )


def lazy_ranges(path: Path) -> list[list[int]]:
    """The byte ranges, in the GGUF, of the tensors llama.cpp reads only a few rows of at a time."""
    import gguf

    return [
        [int(t.data_offset), int(t.data_offset) + int(t.n_bytes)]
        for t in gguf.GGUFReader(path).tensors
        if t.name in _LAZY_TENSORS
    ]
