"""A tiny random-weight Llama base, the torso family of MiniCPM5, the way tests/tiny_qwen35.py
builds a Qwen3.5: written to a directory, nothing downloaded.

`save_llama`: a Llama decoder with an untied output head, as MiniCPM5 is, and a byte-level
BPE tokeniser over the 256 byte symbols (no merges) that prepends `<s>` through its
post-processor, as MiniCPM5's does. The tokeniser reloads as itself through AutoTokenizer
and tokenises every character of a state or question, as the real one does.
"""

from __future__ import annotations

import torch

ATTENTION_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def llama_tokenizer():
    import transformers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, **{c: i + 3 for i, c in enumerate(alphabet)}}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", 0)])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="</s>", unk_token="<unk>")


def save_llama(d: str) -> str:
    """2 layers, grouped-query attention, untied output head, saved to `d`."""
    import transformers

    tok = llama_tokenizer()
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=False,
        bos_token_id=0, eos_token_id=1, pad_token_id=1)
    torch.manual_seed(0)
    transformers.LlamaForCausalLM(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return d
