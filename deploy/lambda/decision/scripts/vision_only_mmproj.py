"""Copy a Gemma 4 mmproj GGUF without its audio encoder.

    python vision_only_mmproj.py LLAMA_CPP_DIR IN.gguf OUT.gguf

llama.cpp's convert_hf_to_gguf.py --mmproj writes every encoder a Gemma 4 checkpoint has: for E2B
and E4B the vision tower and the audio tower, with their projections. mtmd loads every encoder
in the file into memory, and the audio tower's 0.6 GB does not fit beside the vision tower on
AWS Lambda. This copy keeps the vision tower and its projection byte for byte, and leaves out the
audio tensors (`a.*`, `mm.a.*`) and the audio metadata (`clip.has_audio_encoder`,
`clip.audio.*`): the file the converter writes for a checkpoint without an audio encoder.
`general.size_label` is recomputed from the tensors that remain, as the converter computes it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_audio_key(key: str) -> bool:
    return key == "clip.has_audio_encoder" or key.startswith("clip.audio.")


def is_audio_tensor(name: str) -> bool:
    return name.startswith(("a.", "mm.a."))


def main(llama_cpp: str, src: str, dst: str) -> None:
    sys.path.insert(0, str(Path(llama_cpp) / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(src)
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch is None or arch.contents() != "clip":
        raise SystemExit(
            f"{src}: not an mmproj GGUF (general.architecture {arch and arch.contents()!r})"
        )
    vision = reader.get_field("clip.has_vision_encoder")
    if vision is None or not vision.contents():
        raise SystemExit(f"{src}: has no vision encoder")
    tensors = [t for t in reader.tensors if not is_audio_tensor(t.name)]
    if not any(is_audio_tensor(t.name) for t in reader.tensors):
        raise SystemExit(f"{src}: has no audio tensors; copy it as it is")
    n_params = sum(int(t.n_elements) for t in tensors)

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
            value = gguf.size_label(n_params, 0, 0, 0)
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
    print(f"{dst}: {len(tensors)} of {len(reader.tensors)} tensors, {n_params:,} parameters")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
