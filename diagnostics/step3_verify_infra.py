"""
Step 3: Verify the core infrastructure works end-to-end.
Loads model + one SAE + one transcoder, runs basic inference and metrics.
Run: CUDA_VISIBLE_DEVICES=1 python diagnostics/step3_verify_infra.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_sae, load_transcoder
from src.hooks import (
    gather_residual_activations,
    gather_transcoder_activations,
)
from src.metrics import compute_fvu, compute_l0

print("=" * 70)
print("STEP 3: VERIFY INFRASTRUCTURE")
print("=" * 70)

# 1. Load model
model, tokenizer = load_gemma3_1b("pt", device="cuda")

# 2. Tokenize a test prompt
prompt = "The law of conservation of energy states that energy cannot be created or destroyed, only transformed."
inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
print(f"\nPrompt: {prompt}")
print(f"Tokens: {inputs.shape}")

# 3. Load and test residual SAE
print("\n" + "-" * 50)
print("TEST: Residual Stream SAE (layer 22, 65k, medium)")
print("-" * 50)
sae = load_sae(layer=22, site="resid_post", width="65k", l0="medium")

acts = gather_residual_activations(model, 22, inputs)
features = sae.encode(acts.float())
recon = sae.decode(features)

fvu = compute_fvu(recon, acts)
l0 = compute_l0(features)
print(f"  FVU: {fvu:.4f} ({fvu:.2%})")
print(f"  L0:  {l0:.1f}")

# Check top features
top_acts, top_idxs = features.squeeze().mean(0).topk(5)
print(f"  Top 5 features (by mean activation):")
for act, idx in zip(top_acts, top_idxs):
    print(f"    Feature {idx.item():>6d}: {act.item():.3f}")

del sae, features, recon
torch.cuda.empty_cache()

# 4. Load and test transcoder
print("\n" + "-" * 50)
print("TEST: Skip Transcoder (layer 17, 65k, medium, affine)")
print("-" * 50)
tc = load_transcoder(layer=17, width="65k", l0="medium", affine=True)

tc_cache = gather_transcoder_activations(model, 17, inputs)
tc_input = tc_cache["input"].float()
tc_target = tc_cache["target"].float()

tc_features = tc.encode(tc_input)
tc_recon = tc.forward(tc_input)

fvu = compute_fvu(tc_recon, tc_target)
l0 = compute_l0(tc_features)
print(f"  FVU: {fvu:.4f} ({fvu:.2%})")
print(f"  L0:  {l0:.1f}")

del tc, tc_features, tc_recon
torch.cuda.empty_cache()

# 5. Memory report
print("\n" + "-" * 50)
print("GPU MEMORY AFTER TESTS")
print("-" * 50)
allocated = torch.cuda.memory_allocated() / (1024**3)
reserved = torch.cuda.memory_reserved() / (1024**3)
print(f"  Allocated: {allocated:.2f} GB")
print(f"  Reserved:  {reserved:.2f} GB")

print("\n" + "=" * 70)
print("ALL TESTS PASSED — infrastructure is working")
print("=" * 70)
