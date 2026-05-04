"""
Step 4: Verify CLT loading and inference.
This is the single most important verification — CLTs are needed for
MI User, MI Detective, and Believe it or Not.

Run: CUDA_VISIBLE_DEVICES=1 python diagnostics/step4_verify_clt.py
"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt, load_crosscoder
from src.hooks import gather_clt_activations, gather_crosscoder_activations
from src.metrics import compute_fvu, compute_l0

print("=" * 70)
print("STEP 4: VERIFY CLT AND CROSSCODER")
print("=" * 70)

# 1. Load model
model, tokenizer = load_gemma3_1b("pt", device="cuda")

prompt = "The law of conservation of energy states that energy cannot be created or destroyed, only transformed."
inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
print(f"\nPrompt: {prompt}")
print(f"Tokens: {inputs.shape}")

# ============================================================
# 2. TEST CROSSCODER (smaller, loads faster — good sanity check)
# ============================================================
print("\n" + "=" * 70)
print("TEST: Weakly Causal Crosscoder (layers 7,13,17,22 / 262k / medium)")
print("=" * 70)

cc = load_crosscoder(width="262k", l0="medium", device="cuda", half_precision=False)

cc_input = gather_crosscoder_activations(model, [7, 13, 17, 22], inputs)
cc_input_f = cc_input.float()

cc_features = cc.encode(cc_input_f)
cc_recon = cc.forward(cc_input_f)

# Overall metrics
fvu = compute_fvu(cc_recon, cc_input_f)
l0 = compute_l0(cc_features)
print(f"  Overall FVU: {fvu:.4f} ({fvu:.2%})")
print(f"  Overall L0:  {l0:.1f}")

# Per-layer FVU
print("  Per-layer FVU:")
for i, layer in enumerate([7, 13, 17, 22]):
    layer_fvu = compute_fvu(cc_recon[:, i, :], cc_input_f[:, i, :])
    print(f"    Layer {layer:2d}: {layer_fvu:.4f} ({layer_fvu:.2%})")

# Per-layer L0
print("  Per-layer L0:")
for i, layer in enumerate([7, 13, 17, 22]):
    layer_l0 = (cc_features[1:, i, :] > 0).float().sum(-1).mean()
    print(f"    Layer {layer:2d}: {layer_l0:.1f}")

del cc, cc_features, cc_recon, cc_input, cc_input_f
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 3. TEST CLT (the big one)
# ============================================================
print("\n" + "=" * 70)
print("TEST: CLT (262k / big / affine / fp16)")
print("=" * 70)

clt = load_clt(
    width="262k",
    l0="big",
    affine=True,
    device="cuda",
    half_precision=True,
)

# Print parameter shapes to confirm
print("\n  Parameter shapes:")
for name, p in clt.named_parameters():
    print(f"    {name:30s} {str(list(p.shape)):35s} {str(p.dtype):15s} {p.numel()*p.element_size()/(1024**3):.2f} GB")

# Gather activations
print("\n  Gathering activations from all 26 layers...")
clt_inputs, clt_targets = gather_clt_activations(model, 26, inputs)
print(f"    clt_inputs shape:  {clt_inputs.shape}")
print(f"    clt_targets shape: {clt_targets.shape}")

# Run CLT inference
print("\n  Running CLT inference...")
clt_inputs_h = clt_inputs.half()
clt_targets_h = clt_targets.half()

clt_features = clt.encode(clt_inputs_h)
clt_recon = clt.forward(clt_inputs_h)

# Overall metrics
fvu = compute_fvu(clt_recon, clt_targets_h)
l0 = compute_l0(clt_features)
print(f"\n  Overall FVU: {fvu:.4f} ({fvu:.2%})")
print(f"  Overall L0:  {l0:.1f}")

# Per-layer FVU (sample a few layers)
print("\n  Per-layer FVU (selected layers):")
for layer in [0, 5, 10, 13, 17, 20, 22, 25]:
    layer_fvu = compute_fvu(clt_recon[:, layer, :], clt_targets_h[:, layer, :])
    print(f"    Layer {layer:2d}: {layer_fvu:.4f} ({layer_fvu:.2%})")

# Per-layer L0
print("\n  Per-layer L0 (selected layers):")
for layer in [0, 5, 10, 13, 17, 20, 22, 25]:
    layer_l0 = (clt_features[1:, layer, :] > 0).float().sum(-1).mean()
    print(f"    Layer {layer:2d}: {layer_l0:.1f}")

# ============================================================
# 4. VERIFY CROSS-LAYER DECODER STRUCTURE
# ============================================================
print("\n" + "=" * 70)
print("VERIFY: Decoder causality (do later-layer encoders have zero weights for earlier layers?)")
print("=" * 70)

# For each encoder layer, check the norm of its decoder weights to each output layer
print("  Decoder weight norms: w_dec[enc_layer, :, dec_layer, :].norm()")
print("  (Should be near-zero above the diagonal, i.e., enc_layer > dec_layer)")
print()
print(f"  {'':8s}", end="")
for dec_l in range(0, 26, 5):
    print(f"  dec_{dec_l:02d}", end="")
print()

for enc_l in range(0, 26, 5):
    print(f"  enc_{enc_l:02d}", end="")
    for dec_l in range(0, 26, 5):
        norm = clt.w_dec[enc_l, :, dec_l, :].float().norm().item()
        marker = " ***" if enc_l > dec_l and norm > 1.0 else ""
        print(f"  {norm:6.1f}{marker}", end="")
    print()

print("\n  (*** = unexpected: encoder layer > decoder layer but large weight norm)")
print("  If causality holds, norms should be small when enc_layer > dec_layer")

# ============================================================
# 5. FEATURE INSPECTION: what does a CLT feature look like?
# ============================================================
print("\n" + "=" * 70)
print("INSPECT: Top CLT features on this prompt")
print("=" * 70)

# Find the feature with highest mean activation across all layers and positions
mean_acts = clt_features[1:].float().mean(0)  # (26, 10080)
top_vals, top_flat = mean_acts.flatten().topk(10)

str_tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())

for val, flat_idx in zip(top_vals, top_flat):
    layer = flat_idx.item() // clt.d_sae_per_layer
    feat = flat_idx.item() % clt.d_sae_per_layer
    # Get this feature's activation pattern across tokens
    feat_acts = clt_features[:, layer, feat].float()
    # Which tokens does it fire on?
    active_positions = (feat_acts > 0).nonzero(as_tuple=True)[0].tolist()
    active_tokens = [str_tokens[p] for p in active_positions if p < len(str_tokens)]
    
    # Where does this feature's decoder write most strongly?
    dec_norms = clt.w_dec[layer, feat].float().norm(dim=-1)  # (26,)
    top_dec_layers = dec_norms.topk(3)
    dec_info = ", ".join(f"L{l.item()}({v.item():.1f})" for v, l in zip(*top_dec_layers))
    
    print(f"  Layer {layer:2d} Feature {feat:5d} | mean_act={val.item():.2f} | fires on: {active_tokens[:5]} | decodes to: {dec_info}")

# ============================================================
# 6. MEMORY REPORT
# ============================================================
print("\n" + "=" * 70)
print("GPU MEMORY REPORT")
print("=" * 70)
allocated = torch.cuda.memory_allocated() / (1024**3)
reserved = torch.cuda.memory_reserved() / (1024**3)
print(f"  Allocated: {allocated:.2f} GB")
print(f"  Reserved:  {reserved:.2f} GB")
print(f"  Free:      {79.3 - allocated:.1f} GB (approximate)")

del clt, clt_features, clt_recon, clt_inputs, clt_targets
gc.collect()
torch.cuda.empty_cache()

print(f"\n  After cleanup:")
print(f"  Allocated: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

print("\n" + "=" * 70)
print("STEP 4 COMPLETE — CLT infrastructure verified")
print("=" * 70)
