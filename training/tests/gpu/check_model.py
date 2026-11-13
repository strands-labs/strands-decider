"""The decision model on one GPU, with a base or merged checkpoint:

(1) encode() feeds the decoder exactly the core's ids (strands_decider.core.render), left-padded, with each row's
    prefix_bidir boundary at the end of its state part (the spec's S), and each row ends with the template's answer
    cue;
(2) the state's hidden states are the same whatever question follows (so a server may compute them once): bit for
    bit in bf16 where the GPU's kernels allow it, else to 1e-4 relative in float32;
(3) a padded batch gives each row the logits it gets alone, to within bf16 rounding (2 bf16 steps, or one more than
    the batch's unpadded row shows), and exactly in float32 up to float32 rounding;
(4) --attn-split's attention gives the masked attention's logits: to float32 rounding in float32 (in bf16 the two
    round differently; the gap is reported);
(5) the probabilities of every question sum to 1 over its own options.

  python -m training.tests.gpu.check_model <checkpoint dir> [strands-chat | strands-plain]
"""

import sys

import torch

from strands_decider import core
from training.model import DeciderModel

checkpoint = sys.argv[1]
template = sys.argv[2] if len(sys.argv) > 2 else "strands-chat"
CUE = {core.PLAIN: [107, 236820, 14433, 236813], core.CHAT: [106, 107, 105, 4368, 107]}
state = "Order #A-1182 shipped on 3 March; the customer reports the parcel arrived damaged and asks for a refund."
questions = [
    {
        "type": "noul",
        "instructions": "Should a refund be issued?",
        "criteria": {"true": "Issue a refund", "false": "Do not issue a refund"},
    },
    {
        "type": "choice",
        "instructions": "Which team handles this?",
        "criteria": {"billing": "Billing", "logistics": "Logistics", "sales": "Sales"},
    },
    {
        "type": "choice",
        "instructions": "Which option is correct?",
        "criteria": {"C": "4", "A": "3", "D": "2", "B": "1"},
    },
    {
        "type": "noul",
        "instructions": "Does the state mention a date?",
        "criteria": {"false": None, "true": None},
    },
]
rows = [{"state": state, "question": q} for q in questions]
rows.append(
    {
        "state": {"order": "A-1182", "items": [{"sku": "K-9", "qty": 2}]},
        "question": {
            "type": "choice",
            "instructions": "Which item?",
            "criteria": {"K-9": None, "none": None},
        },
    }
)


def bf16_steps(x: torch.Tensor) -> torch.Tensor:
    """One bf16 step at each value's magnitude: 0.125 for a logit near 27."""
    return torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(1e-30))) - 7)


model = DeciderModel(checkpoint, prompt_template=template, device="cuda")
print(
    f"{checkpoint}: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters on {torch.cuda.get_device_name(0)}"
)

# (1)
batch = model.encode(rows, max_length=16384)
ids, mask = batch.inputs["input_ids"].cpu(), batch.inputs["attention_mask"].cpu()
for i, row in enumerate(rows):
    q = model.render(row)
    pad = int((mask[i] == 0).sum())
    assert ids[i, pad:].tolist() == list(q.ids), f"row {i}: encode() ids differ from the core's"
    assert int(batch.prefix_end[i]) - pad == q.state_len, f"row {i}: boundary is not S"
    assert ids[i, -len(CUE[template]) :].tolist() == CUE[template], (
        f"row {i}: not the {template} answer cue"
    )
    assert batch.counts[i] == len(q.codes)
print(f"(1) ids, S and the answer cue equal the core's for {len(rows)} rows")

captured = {}
model.backbone.register_forward_hook(
    lambda _m, _a, out: captured.__setitem__("h", out.last_hidden_state)
)


def state_hidden(m, row):
    b = m.encode([row])
    with torch.inference_mode():
        logits = m(b)
    return captured["h"][0, : int(b.prefix_end[0])].float().cpu(), logits[
        0, : b.counts[0]
    ].float().cpu()


# (2)
(h1, l1), (h2, _l2) = state_hidden(model, rows[0]), state_hidden(model, rows[1])
diff, scale = (h1 - h2).abs().max().item(), h1.abs().max().item()
print(
    f"(2) bf16: max |hidden(q1) - hidden(q2)| over the state's {h1.shape[0]} tokens = {diff:.3g} (largest {scale:.1f})"
)
bitwise_state = diff == 0

# (3)
with torch.inference_mode():
    together = model(batch).float().cpu()
pads = (mask.shape[1] - mask.sum(1)).tolist()
worst_padded = worst_unpadded = 0.0
for i, row in enumerate(rows):
    _h, alone = state_hidden(model, row)
    steps = ((together[i, : len(alone)] - alone).abs() / bf16_steps(alone)).max().item()
    if pads[i]:
        worst_padded = max(worst_padded, steps)
    else:
        worst_unpadded = max(worst_unpadded, steps)
limit = max(2.0, worst_unpadded + 1)
print(
    f"(3) bf16: padded rows {worst_padded:.0f} bf16 steps from alone, the unpadded row {worst_unpadded:.0f}; bound {limit:.0f}"
)
assert worst_padded <= limit, (worst_padded, limit)

# (5)
probabilities = model.option_probabilities(rows, batch_size=len(rows), max_length=16384)
assert all(
    abs(sum(p) - 1) < 1e-5 and len(p) == c for p, c in zip(probabilities, batch.counts, strict=True)
)
print("(5) every question's probabilities sum to 1 over its own options")

# (4)
masked = together
model.install_split_attention(1)
with torch.inference_mode():
    split = model(model.encode(rows, max_length=16384)).float().cpu()
steps = max(
    ((split[i, :c] - masked[i, :c]).abs() / bf16_steps(masked[i, :c])).max().item()
    for i, c in enumerate(batch.counts)
)
print(
    f"(4) bf16: split attention vs masked: {steps:.1f} bf16 steps ({model._split['split_calls']} split calls; the "
    "two round differently in bf16, so the bound is checked in float32 below)"
)
assert model._split["split_calls"] > 0

# float32, where the model fits
weights = sum(p.numel() for p in model.parameters())
if weights * 4 < 0.6 * torch.cuda.get_device_properties(0).total_memory:
    del model
    torch.cuda.empty_cache()
    model = DeciderModel(checkpoint, prompt_template=template, device="cuda").float()
    model.backbone.register_forward_hook(
        lambda _m, _a, out: captured.__setitem__("h", out.last_hidden_state)
    )
    (f1, _), (f2, _) = state_hidden(model, rows[0]), state_hidden(model, rows[1])
    d = ((f1 - f2).abs().max() / f1.abs().max()).item()
    print(f"(2) float32: max |hidden(q1) - hidden(q2)| = {d:.2e} of the largest value")
    assert d < 1e-4, d
    with torch.inference_mode():
        together = model(model.encode(rows, max_length=16384)).float().cpu()
    gap = max(
        (together[i, : len(a)] - a).abs().max().item()
        for i, a in enumerate(state_hidden(model, r)[1] for r in rows)
    )
    print(f"(3) float32: padded batch vs alone, max |logit diff| = {gap:.2e}")
    assert gap < 1e-3, gap
    model.install_split_attention(1)
    with torch.inference_mode():
        split = model(model.encode(rows, max_length=16384)).float().cpu()
    counts = [len(model.render(r).codes) for r in rows]
    gap = max((split[i, :c] - together[i, :c]).abs().max().item() for i, c in enumerate(counts))
    print(f"(4) float32: split attention vs masked, max |logit diff| = {gap:.2e}")
    assert model._split["split_calls"] > 0 and gap < 1e-3, gap
else:
    print("float32 checks skipped: the model does not fit the GPU in float32")
    assert bitwise_state or diff < 0.02 * scale, (diff, scale)
print("MODEL CHECKS PASS")
