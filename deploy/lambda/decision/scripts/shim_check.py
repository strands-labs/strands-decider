"""The llama.cpp shim, built, against transformers on a tiny random Gemma 4 decision checkpoint.

    python shim_check.py --llama-cpp SRC --lib LIB [--work DIR] [--json OUT]

SRC is the patched llama.cpp source (its convert_hf_to_gguf.py and gguf-py), LIB the shim
(`libdecision.so`) built against it. The check needs torch, transformers, safetensors,
sentencepiece, numpy, tokenizers, pydantic and Pillow, and no GPU and no download; it takes about
a minute after the build. tests/test_llama_cpp_shim.py runs it, and so does the Lambda image
build, on the image's own library.

It makes, from a fixed seed, a Gemma 4 checkpoint shaped like E2B but tiny: 15 decoder layers,
four sliding then one full, three times, as E2B alternates them, with KV sharing in the last five,
so that, as in E2B, two full-attention layers come before the shared ones (llama.cpp runs the
shared layers on the output row only, so only a full layer before them can show a full layer's
mask); per-layer embeddings, Gemma 4's whole 262,144-token vocabulary, a vision tower and an audio
tower, a random readout, and a synthetic tokenizer with Gemma 4's control-token ids. It converts it with the stock
convert_hf_to_gguf.py (the decoder in f32; --mmproj, then vision_only_mmproj.py), loads it in
LlamaCppBackend, and runs the same prompts in transformers' Gemma4ForConditionalGeneration in
float32 with the masks passed per layer type. It checks:

  convert   the converter's pplx-decider loader leaves a Strands Decider directory, and one whose
            decision_config.json is not an object, and still claims its own (patches/0002); the
            vision-only mmproj has no audio tensor or key left and the vision tower's tensors
            unchanged; and lazy_ranges finds the per-layer embedding table;
  mask      the shim matches the prefix_bidir reference, and for each question is several times
            further from each reference that differs from it in one way: a causal state, a causal question, causal throughout, a sliding window
            (8 tokens, shorter than the prompts), or causal full-attention layers with bidirectional
            sliding ones, which is what llama.cpp gives without patches/0001 (swa_only);
  reuse     several questions in one call give, bit for bit, what each gives alone, in any order;
  images    with two images in the state, the shim matches the reference, which places each
            image's vision-tower rows at its span; swapping the images changes the rows;
  golden    `strands_decider.core.evaluate`, rendering, the shim and the readout, gives the
            reference's top answer for every question, and probabilities within 0.01.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "strands_decider").is_dir():
        sys.path.insert(0, str(_parent / "src"))
        break

import numpy as np  # noqa: E402

from strands_decider.core import CODES, CONTROL_TOKEN_IDS  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 20261110
VOCAB = 262144
HIDDEN = 64
# A relative error of the shim's hidden state against the prefix_bidir reference within which
# the two agree (the KV cache is f16: 0.009 to 0.034 here), and the factor by which, for each
# question, every other mask's reference must be further away (14 to 31 times for swa_only, the
# nearest; a build without patches/0001 is 27 to 35 times nearer swa_only than prefix_bidir).
AGREE = 0.05
MISS_FACTOR = 5.0

TEXT_CONFIG = {
    "attention_bias": False, "attention_dropout": 0.0, "attention_k_eq_v": False, "bos_token_id": 2,
    "enable_moe_block": False, "eos_token_id": 1, "final_logit_softcapping": 30.0, "global_head_dim": 32,
    "head_dim": 16, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": HIDDEN,
    "hidden_size_per_layer_input": 16, "initializer_range": 0.1, "intermediate_size": 128,
    "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 3,
    "max_position_embeddings": 4096, "model_type": "gemma4_text", "num_attention_heads": 2,
    "num_hidden_layers": 15, "num_key_value_heads": 1, "num_kv_shared_layers": 5, "pad_token_id": 0,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    "sliding_window": 8, "tie_word_embeddings": True, "use_bidirectional_attention": None,
    "use_double_wide_mlp": True, "vocab_size": VOCAB, "vocab_size_per_layer_input": VOCAB,
}  # fmt: skip
VISION_CONFIG = {
    "attention_bias": False, "attention_dropout": 0.0, "default_output_length": 280, "global_head_dim": 16,
    "head_dim": 16, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 32, "initializer_range": 0.1,
    "intermediate_size": 64, "max_position_embeddings": 4096, "model_type": "gemma4_vision",
    "num_attention_heads": 2, "num_hidden_layers": 2, "num_key_value_heads": 2, "patch_size": 16,
    "pooling_kernel_size": 3, "position_embedding_size": 10240, "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}, "standardize": False,
    "use_clipped_linears": True,
}  # fmt: skip
AUDIO_CONFIG = {
    "attention_chunk_size": 12, "attention_context_left": 13, "attention_context_right": 0,
    "attention_invalid_logits_value": -1e9, "attention_logit_cap": 50.0, "conv_kernel_size": 5,
    "gradient_clipping": 1e10, "hidden_act": "silu", "hidden_size": 32, "initializer_range": 0.1,
    "model_type": "gemma4_audio", "num_attention_heads": 2, "num_hidden_layers": 1,
    "output_proj_dims": HIDDEN, "residual_weight": 0.5, "rms_norm_eps": 1e-6,
    "subsampling_conv_channels": [8, 4], "use_clipped_linears": True,
}  # fmt: skip
PROCESSOR_CONFIG = {
    "image_processor": {
        "do_convert_rgb": True, "do_normalize": False, "do_rescale": True, "do_resize": True,
        "image_mean": [0.0, 0.0, 0.0], "image_processor_type": "Gemma4ImageProcessor", "image_seq_length": 280,
        "image_std": [1.0, 1.0, 1.0], "max_soft_tokens": 280, "patch_size": 16, "pooling_kernel_size": 3,
        "resample": 3, "rescale_factor": 1 / 255,
    },
    "image_seq_length": 280, "processor_class": "Gemma4Processor",
}  # fmt: skip


# ---- the tiny checkpoint ------------------------------------------------------------------------


def tokenizer_json() -> dict:
    """A byte-fallback BPE tokenizer of Gemma 4's size with its special tokens at Gemma 4's ids,
    the printable ASCII characters, and every answer code one token (two-letter codes by merges)."""
    specials = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "<|turn>": 105,
                "<turn|>": 106, "<|image>": 255999, "<|audio>": 256000, "<|image|>": 258880,
                "<|audio|>": 258881, "<image|>": 258882, "<audio|>": 258883}  # fmt: skip
    assert all(specials[k] == v for k, v in CONTROL_TOKEN_IDS.items())
    vocab: dict[str, int] = dict(specials)
    ids = iter(i for i in range(VOCAB) if i not in specials.values())
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = next(ids)
    for c in [chr(i) for i in range(32, 127)] + ["\n"]:
        vocab.setdefault(c, next(ids))
    merges = []
    for code in CODES:
        if code not in vocab:
            vocab[code] = next(ids)
            merges.append([code[0], code[1]])
    for i in ids:
        vocab[f"<unused{i}>"] = i
    assert len(vocab) == VOCAB and sorted(vocab.values()) == list(range(VOCAB))
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for t, i in specials.items()]  # fmt: skip
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": None, "pre_tokenizer": None, "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": True,
                  "ignore_merges": False, "vocab": vocab, "merges": merges},
    }  # fmt: skip


def make_checkpoint(d: Path) -> Path:
    """The tiny checkpoint in the decision layout (docs/spec.md, section 9)."""
    import torch
    import transformers
    from safetensors.torch import save_file

    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    config = transformers.Gemma4Config(
        text_config=TEXT_CONFIG, vision_config=VISION_CONFIG, audio_config=AUDIO_CONFIG,
        boi_token_id=255999, eoi_token_id=258882, image_token_id=258880, boa_token_id=256000,
        eoa_token_id=258883, audio_token_id=258881, initializer_range=0.1, tie_word_embeddings=True,
    )  # fmt: skip
    model = transformers.Gemma4ForConditionalGeneration(config).eval()
    model.save_pretrained(d, safe_serialization=True)
    (d / "processor_config.json").write_text(json.dumps(PROCESSOR_CONFIG, indent=1))
    (d / "tokenizer.json").write_text(json.dumps(tokenizer_json()))
    # The other control tokens are extra special tokens, as in Gemma 4's own tokenizer_config.json,
    # so that the converter types them CONTROL and llama.cpp (and mtmd) match them.
    extra = [
        "<mask>",
        "<|turn>",
        "<turn|>",
        "<|image>",
        "<|image|>",
        "<image|>",
        "<|audio>",
        "<|audio|>",
        "<audio|>",
    ]
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<bos>", "eos_token": "<eos>",
        "pad_token": "<pad>", "unk_token": "<unk>", "model_max_length": 4096,
        "extra_special_tokens": extra,
    }))  # fmt: skip
    tok = tokenizer_json()["model"]["vocab"]
    readout = torch.randn(len(CODES), HIDDEN, generator=torch.Generator().manual_seed(SEED)) * 0.5
    save_file({"weight": readout.to(torch.bfloat16)}, str(d / "readout.safetensors"))
    (d / "decision_config.json").write_text(json.dumps({
        "interface": "strands-decider", "format_version": 1, "prompt": "strands-chat",
        "inputs": ["text", "image"], "codes": list(CODES), "token_ids": [tok[c] for c in CODES],
        "temperature": 1.0, "final_logit_softcapping": 30.0, "attention_mode": "prefix_bidir",
        "pooling": "last", "max_length_validated": 4096,
        "provenance": {"base_model": "google/gemma-4-E2B-it",
                       "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7"},
    }))  # fmt: skip
    return d


def convert(llama_cpp: Path, ckpt: Path, out: Path) -> tuple[Path, Path]:
    """The stock converter's decoder (f32) and vision-only mmproj (f32)."""
    out.mkdir(parents=True, exist_ok=True)
    convert_py = llama_cpp / "convert_hf_to_gguf.py"
    gguf, full, mmproj = out / "model.gguf", out / "mmproj-full.gguf", out / "mmproj.gguf"
    for args in ([ckpt, "--outtype", "f32", "--outfile", gguf],
                 [ckpt, "--mmproj", "--outtype", "f32", "--outfile", full]):  # fmt: skip
        r = subprocess.run(
            [sys.executable, convert_py, *map(str, args)], capture_output=True, text=True
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"convert_hf_to_gguf.py {' '.join(map(str, args[1:]))} failed:\n{r.stderr[-3000:]}"
            )
    subprocess.run(
        [sys.executable, HERE / "vision_only_mmproj.py", llama_cpp, full, mmproj], check=True,
        capture_output=True, text=True,
    )  # fmt: skip
    return gguf, mmproj


# Which directories the converter's pplx-decider loader claims: decision_config.json in each, beside
# a config.json and a readout. Run in llama.cpp's tree, as its own tests import it.
MATCHER = """
import json, sys, tempfile
from pathlib import Path
from conversion.pplx_decider import _is_pplx_decider_checkpoint
cases = {"pplx-decider": {"format_version": 1, "temperature": 1.0},
         "vllm-d3": {"format_id": "d3-code-readout-v1", "format_version": 1},
         "strands-decider": {"interface": "strands-decider", "format_version": 1},
         "not an object": [1, 2]}
out = {}
for name, decision in cases.items():
    with tempfile.TemporaryDirectory() as d:
        for f, text in (("config.json", "{}"), ("decision_config.json", json.dumps(decision)),
                        ("readout.safetensors", "")):
            (Path(d) / f).write_text(text)
        try:
            out[name] = _is_pplx_decider_checkpoint(Path(d))
        except Exception as e:
            out[name] = repr(e)
print(json.dumps(out))
"""
MATCHES = {"pplx-decider": True, "vllm-d3": False, "strands-decider": False, "not an object": False}


def check_conversion(llama_cpp: Path, gguf_path: Path, mmproj: Path) -> dict:
    """The converter's loader matcher, the vision-only copy against the converter's full mmproj,
    and the decoder's lazy ranges."""
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf

    sys.path.insert(0, str(HERE))
    from build_artifact import lazy_ranges

    failures = []
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(llama_cpp), str(llama_cpp / "gguf-py")]),
    }
    r = subprocess.run(
        [sys.executable, "-c", MATCHER], cwd=llama_cpp, env=env, capture_output=True, text=True
    )
    matches = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-500:]}
    if matches != MATCHES:
        failures.append(f"the converter's pplx-decider loader claims {matches}, not {MATCHES}")
    full, vision = gguf.GGUFReader(mmproj.with_name("mmproj-full.gguf")), gguf.GGUFReader(mmproj)
    audio_tensors = [t.name for t in vision.tensors if t.name.startswith(("a.", "mm.a."))]
    audio_keys = [
        k for k in vision.fields if k.startswith("clip.audio.") or k == "clip.has_audio_encoder"
    ]
    full_audio = sum(t.name.startswith(("a.", "mm.a.")) for t in full.tensors)
    by_name = {t.name: t for t in full.tensors}
    changed = [
        t.name
        for t in vision.tensors
        if bytes(t.data.tobytes()) != bytes(by_name[t.name].data.tobytes())
    ]
    if not full_audio:
        failures.append("the converter's mmproj has no audio tower to leave out")
    if audio_tensors or audio_keys:
        failures.append(f"the vision-only mmproj keeps {audio_tensors[:3]} {audio_keys[:3]}")
    if changed or len(vision.tensors) != len(full.tensors) - full_audio:
        failures.append(f"the vision-only mmproj changes the vision tower: {changed[:3]}")
    ranges = lazy_ranges(llama_cpp, gguf_path)
    size = gguf_path.stat().st_size
    if len(ranges) != 1 or not 0 < ranges[0][0] < ranges[0][1] <= size:
        failures.append(f"lazy_ranges gives {ranges} for a {size}-byte GGUF")
    return {
        "matcher": matches,
        "mmproj_tensors": [len(full.tensors), len(vision.tensors)],
        "lazy_ranges": ranges,
        "failures": failures,
    }


# ---- the reference --------------------------------------------------------------------------------


class Reference:
    """transformers' Gemma4ForConditionalGeneration on the tiny checkpoint, float32 on CPU, one
    prompt per pass, with the attention mask given per layer type (which bypasses Gemma 4's own)."""

    def __init__(self, ckpt: Path):
        import torch
        import transformers

        self.torch = torch
        conf = transformers.AutoConfig.from_pretrained(ckpt)
        conf.audio_config = None
        self.model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            ckpt, config=conf, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        self.processor = transformers.Gemma4ImageProcessorPil.from_pretrained(ckpt)

    def hidden(self, state, question, *, mask="prefix_bidir", window=None, pixels=()):
        torch = self.torch
        ids = list(state) + list(question)
        n, s = len(ids), len(state)
        i, j = torch.arange(n)[:, None], torch.arange(n)[None, :]
        state_rows, q_rows = i < s, i >= s
        if mask == "prefix_bidir":
            m = (state_rows & (j < s)) | q_rows
        elif mask == "causal_state":  # the state causal within itself
            m = (state_rows & (j <= i)) | q_rows
        elif mask == "causal_question":  # a question causal within itself, seeing the whole state
            m = (state_rows & (j < s)) | (q_rows & ((j < s) | (j <= i)))
        elif mask == "causal":
            m = j <= i
        elif mask == "swa_only":  # without patches/0001: full layers causal within each decode
            m = (state_rows & (j < s)) | q_rows
            full_m = (state_rows & (j <= i)) | (q_rows & ((j < s) | (j <= i)))
        else:
            raise ValueError(mask)
        full = full_m if mask == "swa_only" else m
        sliding = m & ((i - j).abs() < window) if window else m
        kw = {}
        if pixels:
            px = self.processor(images=list(pixels), do_resize=False, return_tensors="pt")
            kw = {k: v for k, v in px.items() if k != "num_soft_tokens_per_image"}
        with torch.inference_mode():
            out = self.model.model(
                input_ids=torch.tensor([ids]),
                attention_mask={"full_attention": full[None, None], "sliding_attention": sliding[None, None]},
                use_cache=False, return_dict=True, **kw,
            )  # fmt: skip
        return out.last_hidden_state[0, -1].float().numpy()


# ---- the checks -----------------------------------------------------------------------------------


def _rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def _png(color, size) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", size, "white")
    ImageDraw.Draw(im).rectangle(
        (size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4), fill=color
    )
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def run(llama_cpp: Path, lib: Path, work: Path) -> dict:
    from strands_decider.backends.llama_cpp import LlamaCppBackend, prepare_pixels
    from strands_decider.core import (
        DecisionConfig,
        Readout,
        evaluate,
        load_tokenizer,
        question_probabilities,
        render,
    )

    started = time.time()
    ckpt = make_checkpoint(work / "checkpoint")
    gguf, mmproj = convert(llama_cpp, ckpt, work / "gguf")
    timings = {"make_and_convert_s": round(time.time() - started, 1)}
    res_convert = check_conversion(llama_cpp, gguf, mmproj)
    backend = LlamaCppBackend(
        lib, gguf, mmproj=mmproj, processor_config=ckpt / "processor_config.json", threads=4
    )
    ref = Reference(ckpt)
    rng = np.random.default_rng(SEED)
    res: dict = {
        "timings": timings,
        "failures": list(res_convert.pop("failures")),
        "convert": res_convert,
    }

    def fail(msg: str) -> None:
        res["failures"].append(msg)

    # mask: a 40-token state and three 12- to 20-token questions, all longer than the window
    state = [2, *rng.integers(1000, VOCAB - 4000, 39).tolist()]
    questions = [rng.integers(1000, VOCAB - 4000, n).tolist() for n in (12, 16, 20)]
    rows = backend.forward(state, questions)
    errs: dict[str, list[float]] = {}
    for q, h in zip(questions, rows, strict=True):
        for name, kw in {"prefix_bidir": {}, "causal_state": {"mask": "causal_state"},
                         "causal_question": {"mask": "causal_question"}, "causal": {"mask": "causal"},
                         "window_8": {"window": 8}, "swa_only": {"mask": "swa_only"}}.items():  # fmt: skip
            errs.setdefault(name, []).append(_rel(h, ref.hidden(state, q, **kw)))
    res["mask"] = {k: [round(x, 4) for x in v] for k, v in errs.items()}  # per question
    agree = max(errs["prefix_bidir"])
    if agree > AGREE:
        fail(f"the shim misses the prefix_bidir reference by {agree:.4f} (> {AGREE})")
    for name in ("causal_state", "causal_question", "causal", "window_8", "swa_only"):
        for i, (other, own) in enumerate(zip(errs[name], errs["prefix_bidir"], strict=True)):
            if other < MISS_FACTOR * max(own, 1e-6):
                fail(
                    f"question {i}: the shim is {other:.4f} from the {name} reference and {own:.4f}"
                    f" from prefix_bidir's (under {MISS_FACTOR:g} times further)"
                )

    # reuse: together, alone, and in reverse order
    alone = np.stack([backend.forward(state, [q])[0] for q in questions])
    reverse = backend.forward(state, questions[::-1])[::-1]
    res["reuse_max_abs"] = float(max(np.abs(rows - alone).max(), np.abs(rows - reverse).max()))
    if res["reuse_max_abs"] != 0.0:
        fail(f"questions in one call differ from questions alone by {res['reuse_max_abs']}")

    # images: two images in the state, at their spans
    tok = load_tokenizer(ckpt)
    config = DecisionConfig.load(ckpt)
    files = [_png((220, 30, 30), (64, 48)), _png((30, 30, 220), (48, 96))]
    request = {
        "state": "two pictures",
        "images": [base64.b64encode(f).decode() for f in files],
        "questions": {
            "which": {
                "type": "choice",
                "instructions": "which?",
                "criteria": {"red": None, "blue": None},
            }
        },
    }
    rendered = render(request, tok, config.prompt)
    q = list(next(iter(rendered.questions.values())).question_ids)
    st = list(rendered.state_ids)
    h_img = backend.forward(st, [q], rendered.images)[0]
    pixels = [prepare_pixels(im) for im in rendered.images]
    h_ref = ref.hidden(st, q, pixels=pixels)
    res["images"] = {"spans": [im.span for im in rendered.images], "tokens": [im.tokens for im in rendered.images],
                     "rel_error": _rel(h_img, h_ref)}  # fmt: skip
    if res["images"]["rel_error"] > AGREE:
        fail(f"with images the shim misses the reference by {res['images']['rel_error']:.4f}")
    swapped = render({**request, "images": request["images"][::-1]}, tok, config.prompt)
    h_swap = backend.forward(list(swapped.state_ids), [q], swapped.images)[0]
    res["images"]["swap_rel_change"] = _rel(h_swap, h_img)
    if res["images"]["swap_rel_change"] < MISS_FACTOR * max(res["images"]["rel_error"], 1e-6):
        fail("swapping the two images barely changes the question's row")

    # golden: the whole runtime path against the reference's readout
    readout = Readout.load(ckpt, config)
    golden_request = {
        "state": {"ticket": 4182, "text": "The checkout page fails after the payment step."},
        "images": [request["images"][0]],
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            "team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "payments", "web": "pages", "mobile": None}},
            "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]},
        },
    }  # fmt: skip
    got = evaluate(golden_request, tok, readout, config, backend)
    rr = render(golden_request, tok, config.prompt)
    ref_rows = {n: ref.hidden(rr.state_ids, rq.question_ids, pixels=[prepare_pixels(i) for i in rr.images])
                for n, rq in rr.questions.items()}  # fmt: skip
    ref_p = question_probabilities(ref_rows, readout, config, rr, stage="hidden")
    shim_p = question_probabilities(dict(zip(rr.questions, backend.forward(
        list(rr.state_ids), [list(x.question_ids) for x in rr.questions.values()], rr.images), strict=True)),
        readout, config, rr, stage="hidden")  # fmt: skip
    res["golden"] = {n: {"reference": [round(float(x), 4) for x in ref_p[n]],
                         "shim": [round(float(x), 4) for x in shim_p[n]]} for n in rr.questions}  # fmt: skip
    for n in rr.questions:
        if int(np.argmax(ref_p[n])) != int(np.argmax(shim_p[n])):
            fail(
                f"golden {n!r}: top answer {int(np.argmax(shim_p[n]))}, the reference's {int(np.argmax(ref_p[n]))}"
            )
        if float(np.abs(ref_p[n] - shim_p[n]).max()) > 0.01:
            fail(
                f"golden {n!r}: probabilities differ by {float(np.abs(ref_p[n] - shim_p[n]).max()):.4f}"
            )
    res["golden_answers"] = got["answers"]
    backend.close()
    timings["total_s"] = round(time.time() - started, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        res = run(a.llama_cpp, a.lib, a.work or Path(tmp))
    text = json.dumps(res, indent=1, default=str)
    if a.json:
        a.json.write_text(text)
    print(text)
    if res["failures"]:
        raise SystemExit("shim check FAILED:\n  " + "\n  ".join(res["failures"]))
    print("shim check OK")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
