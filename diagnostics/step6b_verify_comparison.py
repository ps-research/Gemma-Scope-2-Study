"""
Step 6b: Verify cross-tool comparison with fixes.
- Fix 1: Position-aware logit effects
- Fix 2: Matched width comparison
- Fix 3: Delta loss in table

Run: CUDA_VISIBLE_DEVICES=1 python diagnostics/step6b_verify_comparison.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt
from src.comparison import (
    run_full_comparison, run_matched_comparison, save_comparison,
)

print("=" * 70)
print("STEP 6b: CROSS-TOOL COMPARISON (with fixes)")
print("=" * 70)

CACHE_DIR = "/workspace/Gemma-Scope-2-Study/cache"

model, tokenizer = load_gemma3_1b("pt", device="cuda")

# Pre-load CLT
print("\nPre-loading CLT...")
clt = load_clt(
    width="262k", l0="big", affine=True,
    device="cuda", half_precision=True,
    cache_dir=CACHE_DIR,
)

prompt = "The capital of France is"

# ============================================================
# 1. Full comparison (with delta loss)
# ============================================================
print(f"\n{'='*70}")
print(f"FULL COMPARISON: '{prompt}'")
print(f"{'='*70}\n")

result = run_full_comparison(
    model, tokenizer, prompt,
    target_layer=17, width="65k", l0="medium",
    variant="pt", device="cuda",
    cache_dir=CACHE_DIR, preloaded_clt=clt,
)

print(f"\n--- COMPARISON TABLE (with delta loss) ---")
print(result.summary_table())

# Show top features with positions
print(f"\n--- TOP FEATURES WITH POSITIONS ---")
for r in result.tool_results:
    print(f"\n  {r.tool_name}:")
    for i, feat in enumerate(r.top_features[:3]):
        if len(feat) == 4:
            idx, act, pos, layer = feat
            tok = result.tokens[pos] if pos < len(result.tokens) else "?"
            print(f"    {i+1}. L{layer}/f{idx}: act={act:.0f} at pos {pos} ('{tok}')")
        elif len(feat) == 3:
            idx, act, pos = feat
            tok = result.tokens[pos] if pos < len(result.tokens) else "?"
            print(f"    {i+1}. f{idx}: act={act:.0f} at pos {pos} ('{tok}')")

    if r.top_logit_effects and r.top_logit_effects[0][0] != "(pre-W_O space, not directly comparable)":
        print(f"    Strongest feature promotes:")
        for tok, eff in r.top_logit_effects[:5]:
            print(f"      {tok:15s} {eff:+.4f}")

# ============================================================
# 2. Matched comparison (fair: all tools at 16k, small)
# ============================================================
print(f"\n{'='*70}")
print(f"MATCHED COMPARISON: all single-layer tools at 16k/small")
print(f"{'='*70}\n")

matched = run_matched_comparison(
    model, tokenizer, prompt,
    target_layer=17, variant="pt", device="cuda",
    cache_dir=CACHE_DIR,
)

print(f"\n--- MATCHED TABLE ---")
print(matched.summary_table())

# ============================================================
# 3. Key insights
# ============================================================
print(f"\n{'='*70}")
print("KEY INSIGHTS")
print(f"{'='*70}")

# Transcoder skip effect
tc_results = [r for r in result.tool_results if "Transcoder" in r.tool_name]
if len(tc_results) == 2:
    no_skip, skip = tc_results
    print(f"\n  1. Skip connection effect (layer 17):")
    print(f"     FVU: {no_skip.fvu:.4f} -> {skip.fvu:.4f} ({(1 - skip.fvu/no_skip.fvu)*100:.1f}% reduction)")
    if no_skip.delta_loss > 0 and skip.delta_loss > 0:
        print(f"     Delta loss: {no_skip.delta_loss:.4f} -> {skip.delta_loss:.4f} ({(1 - skip.delta_loss/no_skip.delta_loss)*100:.1f}% reduction)")

# Residual SAE vs transcoder delta loss
resid = next((r for r in result.tool_results if "Residual" in r.tool_name), None)
if resid and tc_results:
    skip_tc = tc_results[1]
    print(f"\n  2. Residual SAE vs Skip Transcoder:")
    print(f"     Residual SAE:    FVU={resid.fvu:.4f}, dL={resid.delta_loss:.4f}")
    print(f"     Skip Transcoder: FVU={skip_tc.fvu:.4f}, dL={skip_tc.delta_loss:.4f}")
    if resid.delta_loss > 0 and skip_tc.delta_loss > 0:
        print(f"     Residual SAE has {'HIGHER' if resid.delta_loss > skip_tc.delta_loss else 'LOWER'} delta loss")
        print(f"     (Paper predicts: residual SAE has higher delta loss despite lower FVU)")

# Attention SAE d_model
attn = next((r for r in result.tool_results if "Attn" in r.tool_name), None)
if attn:
    print(f"\n  3. Attention SAE operates in pre-W_O space:")
    print(f"     d_model=1024 (4 heads x 256 head_dim), not 1152")
    print(f"     L0={attn.l0_actual:.1f} — much higher than residual SAE ({resid.l0_actual:.1f})")
    print(f"     Logit effects not directly comparable (needs W_O projection)")

# Multi-layer tools note
print(f"\n  4. Multi-layer tools note:")
print(f"     Crosscoder: 262k width = 65,536 features/layer x 4 layers")
print(f"     CLT:        262k width = 10,080 features/layer x 26 layers")
print(f"     Single-layer: 65k width = 65,536 features (6.5x more per layer than CLT)")
print(f"     FVU comparison is NOT apples-to-apples — report this in the paper")

save_comparison(result, "/workspace/Gemma-Scope-2-Study/outputs/comparison_france_v2.json")

print(f"\n{'='*70}")
print(f"GPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
print("STEP 6b COMPLETE")
print(f"{'='*70}")
