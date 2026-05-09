"""
Step 7: Verify interventions, visualization, and autointerp modules.

Run: CUDA_VISIBLE_DEVICES=1 python diagnostics/step7_verify_pipelines.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_sae, load_clt
from src.hooks import gather_residual_activations, gather_clt_activations
from src.metrics import compute_fvu, compute_l0
from src.interventions import (
    ablate_sae_features, steer_with_feature, clamp_sae_features,
    ablate_clt_features, run_ablation_sweep, print_intervention_result,
)
from src.viz import (
    plot_feature_activations, plot_reconstruction_heatmap,
    plot_tool_comparison, highlight_tokens_html, save_html_report,
    render_attribution_graph_text,
)

CACHE = "/workspace/Gemma-Scope-2-Study/cache"
OUT = "/workspace/Gemma-Scope-2-Study/outputs"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("STEP 7: VERIFY PIPELINES")
print("=" * 70)

# Load model and SAE
model, tokenizer = load_gemma3_1b("pt", device="cuda")
sae = load_sae(22, site="resid_post", width="65k", l0="medium", cache_dir=CACHE)

prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
str_tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())

# ============================================================
# TEST 1: Feature ablation
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: Feature Ablation")
print(f"{'='*70}")

# Find top features first
acts = gather_residual_activations(model, 22, inputs)
features = sae.encode(acts.float())
top_vals, top_idxs = features[1:].mean(0).topk(5)
print(f"Top 5 features: {top_idxs.tolist()}")

# Ablate the strongest feature
result = ablate_sae_features(
    model, sae, tokenizer, inputs,
    layer=22,
    feature_indices=[top_idxs[0].item()],
)
print_intervention_result(result)

# Ablate top 3 features
result_multi = ablate_sae_features(
    model, sae, tokenizer, inputs,
    layer=22,
    feature_indices=top_idxs[:3].tolist(),
)
print_intervention_result(result_multi)

# ============================================================
# TEST 2: Feature clamping
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: Feature Clamping")
print(f"{'='*70}")

# Amplify the strongest feature by 2x
strongest_idx = top_idxs[0].item()
original_act = features[:, strongest_idx].max().item()

result_clamp = clamp_sae_features(
    model, sae, tokenizer, inputs,
    layer=22,
    clamp_map={strongest_idx: original_act * 2},
)
print_intervention_result(result_clamp)

# ============================================================
# TEST 3: Ablation sweep (find most impactful features)
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: Ablation Sweep (top 10 features)")
print(f"{'='*70}")

sweep_results = run_ablation_sweep(
    model, sae, tokenizer, inputs,
    layer=22,
    feature_indices=top_idxs[:10].tolist(),
)

print(f"  {'Feature':>8s}  {'DeltaLoss':>10s}  {'Changed':>8s}  {'Clean':>10s}  {'Ablated':>10s}")
print(f"  {'-'*50}")
for idx, dl, changed, clean_top, interv_top in sweep_results[:10]:
    print(f"  f{idx:>6d}  {dl:>+10.4f}  {'YES' if changed else 'no':>8s}  {clean_top:>10s}  {interv_top:>10s}")

# ============================================================
# TEST 4: Visualization
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: Visualization")
print(f"{'='*70}")

# Feature activation heatmap
plot_feature_activations(
    features, str_tokens,
    top_k=15,
    title=f"Feature Activations: '{prompt}'",
    save_path=f"{OUT}/feature_heatmap_france.png",
)

# Token highlighting HTML
feat_acts_at_last = features[:, top_idxs[0].item()].cpu().tolist()
html = highlight_tokens_html(
    str_tokens, feat_acts_at_last,
    title=f"Feature {top_idxs[0].item()} activations",
)
save_html_report([html], f"{OUT}/token_highlight_france.html",
                 title="Feature Activation Highlighting")

# Reconstruction error heatmap (per-layer using CLT)
print("  Loading CLT for reconstruction heatmap...")
clt = load_clt(width="262k", l0="big", affine=True, device="cuda",
               half_precision=True, cache_dir=CACHE)

clt_in, clt_tgt = gather_clt_activations(model, 26, inputs)
clt_in_h = clt_in.half()
clt_tgt_h = clt_tgt.half()
recon = clt.forward(clt_in_h)

# Per-position per-layer MSE
recon_error = (recon - clt_tgt_h).float().pow(2).mean(dim=-1)  # (seq, 26)
plot_reconstruction_heatmap(
    recon_error, str_tokens,
    title=f"CLT Reconstruction Error: '{prompt}'",
    save_path=f"{OUT}/recon_error_france.png",
)

# ============================================================
# TEST 5: CLT-based ablation
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: CLT Feature Ablation")
print(f"{'='*70}")

# Find top CLT features
clt_features = clt.encode(clt_in_h)
mean_acts = clt_features[1:].float().mean(0)
flat_top = mean_acts.flatten().topk(5)
d_sae = mean_acts.shape[1]

clt_top_specs = []
for flat_idx in flat_top.indices:
    layer = flat_idx.item() // d_sae
    feat = flat_idx.item() % d_sae
    clt_top_specs.append((layer, feat))
    print(f"  CLT feature: L{layer}/f{feat}")

# Ablate top CLT feature
result_clt = ablate_clt_features(
    model, clt, tokenizer, inputs,
    feature_specs=[clt_top_specs[0]],
)
print_intervention_result(result_clt)

# Ablate top 3 CLT features
result_clt_multi = ablate_clt_features(
    model, clt, tokenizer, inputs,
    feature_specs=clt_top_specs[:3],
)
print_intervention_result(result_clt_multi)

# ============================================================
# TEST 6: Steering (using IT model for generation)
# ============================================================
print(f"\n{'='*70}")
print("TEST 6: Feature Steering")
print(f"{'='*70}")

# Load IT model for generation
del model
torch.cuda.empty_cache()

model_it, _ = load_gemma3_1b("it", device="cuda")

user_prompt = "Tell me a fun fact."
it_prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
it_inputs = tokenizer.encode(it_prompt, return_tensors="pt", add_special_tokens=True).to("cuda")

# Steer with the thermodynamics feature from the tutorial (6524)
result_steer = steer_with_feature(
    model_it, sae, tokenizer, it_inputs,
    layer=22,
    feature_idx=6524,
    coeff=0.14,
    max_new_tokens=60,
    steering_layer=14,  # steer earlier for stronger effect
)
print_intervention_result(result_steer)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  GPU allocated: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
print(f"  Files generated:")
for f in os.listdir(OUT):
    if f.endswith(('.png', '.html', '.json')):
        size = os.path.getsize(os.path.join(OUT, f)) / 1024
        print(f"    {f}: {size:.1f} KB")

print(f"\n{'='*70}")
print("STEP 7 COMPLETE — All pipelines verified")
print(f"{'='*70}")
