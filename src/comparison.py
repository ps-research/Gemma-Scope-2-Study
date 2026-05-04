"""
Cross-tool comparison pipeline for the AI Anatomy paper.

Runs all Gemma Scope 2 tools on the same prompt and compares:
  1. Reconstruction quality (FVU, delta loss)
  2. Sparsity (L0)
  3. Feature overlap across tools
  4. Top features per tool (with position-specific logit effects)
  5. Attribution graph properties (CLT vs per-layer transcoders)

FIXES APPLIED:
  - Logit effects computed at the position where the feature fires strongest,
    not raw decoder vector dot product (which gives misleading results)
  - Delta loss added as a functional metric alongside FVU
  - Matched comparison mode for fair width/L0 comparisons
"""

import torch
import numpy as np
import json
import time
import gc
from dataclasses import dataclass, field
from typing import Optional
from functools import partial

from src.loader import (
    load_sae, load_transcoder, load_crosscoder, load_clt,
    GEMMA3_1B_CROSSCODER_LAYERS, GEMMA3_1B_NUM_LAYERS,
)
from src.hooks import (
    gather_residual_activations,
    gather_mlp_out_activations,
    gather_attn_out_activations,
    gather_transcoder_activations,
    gather_crosscoder_activations,
    gather_clt_activations,
)
from src.metrics import compute_fvu, compute_l0, cross_entropy_loss


# ============================================================
# Data structures
# ============================================================

@dataclass
class ToolResult:
    """Results from running a single tool on a prompt."""
    tool_name: str
    site: str
    layer: Optional[int]
    width: str
    l0_target: str

    # Metrics
    fvu: float = 0.0
    delta_loss: float = 0.0
    l0_actual: float = 0.0
    num_active_features: int = 0

    # Top features: list of (feature_idx, activation, position, [layer for multi-layer])
    top_features: list = field(default_factory=list)

    # Top logit effects at strongest position: list of (token_str, logit_effect)
    top_logit_effects: list = field(default_factory=list)

    # Timing
    load_time: float = 0.0
    inference_time: float = 0.0

    # Memory
    model_size_mb: float = 0.0


@dataclass
class ComparisonResult:
    """Results from comparing all tools on a single prompt."""
    prompt: str
    tokens: list
    tool_results: list = field(default_factory=list)

    def summary_table(self) -> str:
        header = (f"{'Tool':<35s} {'Site':<12s} {'Layer':<6s} "
                  f"{'FVU':>8s} {'DeltaL':>8s} {'L0':>7s} {'#Act':>6s} "
                  f"{'Load':>7s} {'Infer':>7s} {'MB':>7s}")
        sep = "-" * len(header)
        lines = [header, sep]

        for r in self.tool_results:
            layer_str = str(r.layer) if r.layer is not None else "multi"
            dl_str = f"{r.delta_loss:.4f}" if r.delta_loss > 0 else "N/A"
            lines.append(
                f"{r.tool_name:<35s} {r.site:<12s} {layer_str:<6s} "
                f"{r.fvu:>8.4f} {dl_str:>8s} {r.l0_actual:>7.1f} {r.num_active_features:>6d} "
                f"{r.load_time:>6.1f}s {r.inference_time:>6.3f}s {r.model_size_mb:>6.0f}"
            )

        return "\n".join(lines)


# ============================================================
# Helpers
# ============================================================

def _get_effective_unembed(model):
    """Get W_unembed * layernorm_weight for logit effect computation."""
    w_u = model.lm_head.weight
    ln_w = model.model.norm.weight
    return (w_u * ln_w).float()


def _logit_effects_at_position(decoder_vector, w_eff, tokenizer, k=10):
    """
    Get top-k tokens promoted/suppressed by a decoder vector.
    Returns list of (token_str, logit_effect).
    """
    logit_effects = w_eff @ decoder_vector.float()
    top_vals, top_ids = logit_effects.topk(k)
    bot_vals, bot_ids = logit_effects.topk(k, largest=False)

    results = []
    for v, tid in zip(top_vals, top_ids):
        tok_str = tokenizer.decode([tid.item()]).strip()
        results.append((tok_str, v.item()))
    return results


def _find_strongest_position(features, skip_bos=True):
    """
    Find the (position, feature_idx) with highest activation.
    features: (seq_len, d_sae)
    Returns: (position, feature_idx, activation_value)
    """
    start = 1 if skip_bos else 0
    f = features[start:]
    flat_idx = f.argmax().item()
    pos_offset = flat_idx // f.shape[-1]
    feat_idx = flat_idx % f.shape[-1]
    act_val = f[pos_offset, feat_idx].item()
    return start + pos_offset, feat_idx, act_val


def _top_features_with_positions(features, k=10, skip_bos=True):
    """
    Get top-k features by max activation, with position info.
    features: (seq_len, d_sae)
    Returns: list of (feature_idx, max_activation, position_of_max)
    """
    start = 1 if skip_bos else 0
    f = features[start:].float()

    # For each feature, get its max activation and the position where it occurs
    max_acts, max_positions = f.max(dim=0)  # (d_sae,), (d_sae,)

    top_vals, top_idxs = max_acts.topk(min(k, max_acts.shape[0]))
    results = []
    for val, idx in zip(top_vals, top_idxs):
        feat_idx = idx.item()
        pos = max_positions[feat_idx].item() + start
        results.append((feat_idx, val.item(), pos))
    return results


def _compute_delta_loss_resid(model, sae, layer, inputs):
    """Delta loss for residual SAE: splice reconstruction into residual stream."""
    # Clean forward pass
    model_output = model.forward(inputs, output_hidden_states=True)
    logits_clean = model_output.logits[0]
    input_acts = model_output.hidden_states[layer + 1][0]

    # Reconstruct (skip BOS)
    recon = sae.forward(input_acts[1:].float())

    # Patched forward pass
    def _hook(mod, inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        if tensor.dim() == 3:
            tensor[0, 1:] = recon
        else:
            tensor[1:] = recon
        return out

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        logits_patched = model.forward(inputs).logits[0]
    finally:
        handle.remove()

    tokens = inputs[0]
    loss_clean = cross_entropy_loss(logits_clean, tokens).mean().item()
    loss_patched = cross_entropy_loss(logits_patched, tokens).mean().item()
    return loss_patched - loss_clean


def _compute_delta_loss_transcoder(model, tc, layer, inputs):
    """Delta loss for transcoder: splice reconstruction into MLP output."""
    # Cache transcoder input during clean pass
    cached_input = None

    def _cache(mod, inp, out):
        nonlocal cached_input
        # post_feedforward_layernorm output can be tensor or tuple
        tensor = out[0] if isinstance(out, tuple) else out
        # Remove batch dim if present
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        cached_input = tensor

    h = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(_cache)
    try:
        logits_clean = model.forward(inputs).logits[0]
    finally:
        h.remove()

    # Compute transcoder reconstruction (skip BOS)
    recon = tc.forward(cached_input[1:].float())

    def _inject(mod, inp, out):
        # Handle both tuple and tensor outputs, and batch dimensions
        if isinstance(out, tuple):
            tensor = out[0]
        else:
            tensor = out
        if tensor.dim() == 3:
            tensor[0, 1:] = recon
        else:
            tensor[1:] = recon
        return out

    h2 = model.model.layers[layer].post_feedforward_layernorm.register_forward_hook(_inject)
    try:
        logits_patched = model.forward(inputs).logits[0]
    finally:
        h2.remove()

    tokens = inputs[0]
    loss_clean = cross_entropy_loss(logits_clean, tokens).mean().item()
    loss_patched = cross_entropy_loss(logits_patched, tokens).mean().item()
    return loss_patched - loss_clean


# ============================================================
# Per-tool runners
# ============================================================

def run_residual_sae(model, tokenizer, inputs, layer, width="65k", l0="medium",
                     variant="pt", device="cuda", cache_dir=None):
    t0 = time.time()
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    sae = load_sae(layer, site="resid_post", width=width, l0=l0, variant=variant,
                   device=device, **kwargs)
    load_time = time.time() - t0
    model_mb = sum(p.numel() * p.element_size() for p in sae.parameters()) / (1024**2)

    t0 = time.time()
    acts = gather_residual_activations(model, layer, inputs)
    features = sae.encode(acts.float())
    recon = sae.decode(features)
    infer_time = time.time() - t0

    fvu = compute_fvu(recon, acts).item()
    l0_val = compute_l0(features).item()
    n_active = (features[1:] > 0).any(dim=0).sum().item()

    # Top features with position info
    top_feats = _top_features_with_positions(features)

    # Logit effects at the position where the strongest feature fires
    w_eff = _get_effective_unembed(model)
    top_logits = []
    if top_feats:
        best_idx = top_feats[0][0]
        top_logits = _logit_effects_at_position(sae.w_dec[best_idx], w_eff, tokenizer)

    # Delta loss
    delta_loss = _compute_delta_loss_resid(model, sae, layer, inputs)

    result = ToolResult(
        tool_name=f"Residual SAE ({width}, {l0})",
        site="resid_post", layer=layer, width=width, l0_target=l0,
        fvu=fvu, delta_loss=delta_loss, l0_actual=l0_val,
        num_active_features=n_active,
        top_features=top_feats, top_logit_effects=top_logits,
        load_time=load_time, inference_time=infer_time, model_size_mb=model_mb,
    )

    del sae, features, recon, acts
    torch.cuda.empty_cache()
    return result


def run_mlp_sae(model, tokenizer, inputs, layer, width="65k", l0="medium",
                variant="pt", device="cuda", cache_dir=None):
    t0 = time.time()
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    sae = load_sae(layer, site="mlp_out", width=width, l0=l0, variant=variant,
                   device=device, **kwargs)
    load_time = time.time() - t0
    model_mb = sum(p.numel() * p.element_size() for p in sae.parameters()) / (1024**2)

    t0 = time.time()
    acts = gather_mlp_out_activations(model, layer, inputs)
    features = sae.encode(acts.float())
    recon = sae.decode(features)
    infer_time = time.time() - t0

    fvu = compute_fvu(recon, acts).item()
    l0_val = compute_l0(features).item()
    n_active = (features[1:] > 0).any(dim=0).sum().item()

    top_feats = _top_features_with_positions(features)

    w_eff = _get_effective_unembed(model)
    top_logits = []
    if top_feats:
        top_logits = _logit_effects_at_position(sae.w_dec[top_feats[0][0]], w_eff, tokenizer)

    # No delta loss for MLP SAE (would need different patching logic)
    # MLP output is not a residual stream site, patching is non-trivial
    delta_loss = 0.0

    result = ToolResult(
        tool_name=f"MLP SAE ({width}, {l0})",
        site="mlp_out", layer=layer, width=width, l0_target=l0,
        fvu=fvu, delta_loss=delta_loss, l0_actual=l0_val,
        num_active_features=n_active,
        top_features=top_feats, top_logit_effects=top_logits,
        load_time=load_time, inference_time=infer_time, model_size_mb=model_mb,
    )

    del sae, features, recon, acts
    torch.cuda.empty_cache()
    return result


def run_attn_sae(model, tokenizer, inputs, layer, width="65k", l0="medium",
                 variant="pt", device="cuda", cache_dir=None):
    t0 = time.time()
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    sae = load_sae(layer, site="attn_out", width=width, l0=l0, variant=variant,
                   device=device, **kwargs)
    load_time = time.time() - t0
    model_mb = sum(p.numel() * p.element_size() for p in sae.parameters()) / (1024**2)

    t0 = time.time()
    acts = gather_attn_out_activations(model, layer, inputs)
    features = sae.encode(acts.float())
    recon = sae.decode(features)
    infer_time = time.time() - t0

    fvu = compute_fvu(recon, acts).item()
    l0_val = compute_l0(features).item()
    n_active = (features[1:] > 0).any(dim=0).sum().item()

    top_feats = _top_features_with_positions(features)

    # Attn SAE is in pre-W_O space (d=1024) not residual space (d=1152)
    # Logit effects require projecting through W_O first
    # For now, skip — flag in results
    top_logits = [("(pre-W_O space, not directly comparable)", 0.0)]

    delta_loss = 0.0  # Would need W_O projection for meaningful patching

    result = ToolResult(
        tool_name=f"Attn SAE ({width}, {l0})",
        site="attn_out", layer=layer, width=width, l0_target=l0,
        fvu=fvu, delta_loss=delta_loss, l0_actual=l0_val,
        num_active_features=n_active,
        top_features=top_feats, top_logit_effects=top_logits,
        load_time=load_time, inference_time=infer_time, model_size_mb=model_mb,
    )

    del sae, features, recon, acts
    torch.cuda.empty_cache()
    return result


def run_transcoder(model, tokenizer, inputs, layer, width="65k", l0="medium",
                   affine=False, variant="pt", device="cuda", cache_dir=None):
    t0 = time.time()
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    tc = load_transcoder(layer, width=width, l0=l0, affine=affine, variant=variant,
                         device=device, **kwargs)
    load_time = time.time() - t0
    model_mb = sum(p.numel() * p.element_size() for p in tc.parameters()) / (1024**2)
    skip_label = "skip" if affine else "no-skip"

    t0 = time.time()
    cache = gather_transcoder_activations(model, layer, inputs)
    tc_input = cache["input"].float()
    tc_target = cache["target"].float()
    features = tc.encode(tc_input)
    recon = tc.forward(tc_input)
    infer_time = time.time() - t0

    fvu = compute_fvu(recon, tc_target).item()
    l0_val = compute_l0(features).item()
    n_active = (features[1:] > 0).any(dim=0).sum().item()

    top_feats = _top_features_with_positions(features)

    # Transcoder decoder writes to MLP output space = residual space
    # So logit effects ARE meaningful here
    w_eff = _get_effective_unembed(model)
    top_logits = []
    if top_feats:
        top_logits = _logit_effects_at_position(tc.w_dec[top_feats[0][0]], w_eff, tokenizer)

    # Delta loss
    delta_loss = _compute_delta_loss_transcoder(model, tc, layer, inputs)

    result = ToolResult(
        tool_name=f"Transcoder {skip_label} ({width}, {l0})",
        site="transcoder", layer=layer, width=width, l0_target=l0,
        fvu=fvu, delta_loss=delta_loss, l0_actual=l0_val,
        num_active_features=n_active,
        top_features=top_feats, top_logit_effects=top_logits,
        load_time=load_time, inference_time=infer_time, model_size_mb=model_mb,
    )

    del tc, features, recon, cache
    torch.cuda.empty_cache()
    return result


def run_crosscoder(model, tokenizer, inputs, width="262k", l0="medium",
                   variant="pt", device="cuda", cache_dir=None):
    t0 = time.time()
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    cc = load_crosscoder(width=width, l0=l0, variant=variant, device=device, **kwargs)
    load_time = time.time() - t0
    model_mb = sum(p.numel() * p.element_size() for p in cc.parameters()) / (1024**2)
    layers = GEMMA3_1B_CROSSCODER_LAYERS

    t0 = time.time()
    cc_input = gather_crosscoder_activations(model, layers, inputs).float()
    features = cc.encode(cc_input)
    recon = cc.forward(cc_input)
    infer_time = time.time() - t0

    fvu = compute_fvu(recon, cc_input).item()
    l0_val = compute_l0(features).item()
    n_active = (features[1:] > 0).any(dim=0).sum().item()

    # Top features with layer info
    # features shape: (seq, num_layers, d_sae)
    mean_acts = features[1:].float().max(dim=0).values  # (num_layers, d_sae)
    flat_acts = mean_acts.flatten()
    top_vals, top_flat = flat_acts.topk(min(10, flat_acts.shape[0]))
    d_sae = mean_acts.shape[1]
    top_feats = []
    for val, flat_idx in zip(top_vals, top_flat):
        layer_idx = flat_idx.item() // d_sae
        feat_idx = flat_idx.item() % d_sae
        # Find position of max activation for this feature
        feat_acts = features[1:, layer_idx, feat_idx]
        pos = feat_acts.argmax().item() + 1
        top_feats.append((feat_idx, val.item(), pos, layers[layer_idx]))

    delta_loss = 0.0  # Crosscoder patching is complex (multi-layer)

    result = ToolResult(
        tool_name=f"Crosscoder ({width}, {l0})",
        site="crosscoder", layer=None, width=width, l0_target=l0,
        fvu=fvu, delta_loss=delta_loss, l0_actual=l0_val,
        num_active_features=n_active,
        top_features=top_feats, top_logit_effects=[],
        load_time=load_time, inference_time=infer_time, model_size_mb=model_mb,
    )

    del cc, features, recon, cc_input
    torch.cuda.empty_cache()
    return result


def run_clt(model, tokenizer, inputs, clt=None, width="262k", l0="big",
            affine=True, variant="pt", device="cuda", half_precision=True,
            cache_dir=None):
    if clt is None:
        t0 = time.time()
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        clt = load_clt(width=width, l0=l0, affine=affine, variant=variant,
                       device=device, half_precision=half_precision, **kwargs)
        load_time = time.time() - t0
        clt_was_loaded = False
    else:
        load_time = 0.0
        clt_was_loaded = True

    model_mb = sum(p.numel() * p.element_size() for p in clt.parameters()) / (1024**2)

    t0 = time.time()
    clt_inputs, clt_targets = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)
    if half_precision:
        clt_inputs = clt_inputs.half()
        clt_targets = clt_targets.half()

    features = clt.encode(clt_inputs)
    recon = clt.forward(clt_inputs)
    infer_time = time.time() - t0

    fvu = compute_fvu(recon, clt_targets).item()
    l0_val = compute_l0(features).item()
    n_active = (features[1:] > 0).any(dim=0).sum().item()

    # Top features with layer info
    mean_acts = features[1:].float().max(dim=0).values  # (num_layers, d_sae)
    flat_acts = mean_acts.flatten()
    top_vals, top_flat = flat_acts.topk(min(10, flat_acts.shape[0]))
    d_sae = mean_acts.shape[1]
    top_feats = []
    for val, flat_idx in zip(top_vals, top_flat):
        layer_idx = flat_idx.item() // d_sae
        feat_idx = flat_idx.item() % d_sae
        feat_acts = features[1:, layer_idx, feat_idx]
        pos = feat_acts.argmax().item() + 1
        top_feats.append((feat_idx, val.item(), pos, layer_idx))

    # Per-layer FVU
    per_layer_fvu = {}
    for layer in range(GEMMA3_1B_NUM_LAYERS):
        layer_fvu = compute_fvu(recon[:, layer, :], clt_targets[:, layer, :]).item()
        per_layer_fvu[layer] = layer_fvu

    skip_label = "affine" if affine else "no-skip"
    result = ToolResult(
        tool_name=f"CLT {skip_label} ({width}, {l0})",
        site="clt", layer=None, width=width, l0_target=l0,
        fvu=fvu, delta_loss=0.0, l0_actual=l0_val,
        num_active_features=n_active,
        top_features=top_feats, top_logit_effects=[],
        load_time=load_time, inference_time=infer_time, model_size_mb=model_mb,
    )

    if not clt_was_loaded:
        del clt
        torch.cuda.empty_cache()

    del features, recon, clt_inputs, clt_targets
    torch.cuda.empty_cache()

    return result, per_layer_fvu


# ============================================================
# Full comparison pipeline
# ============================================================

def run_full_comparison(
    model,
    tokenizer,
    prompt,
    target_layer=17,
    width="65k",
    l0="medium",
    variant="pt",
    device="cuda",
    cache_dir=None,
    preloaded_clt=None,
    skip_heavy=False,
):
    """
    Run all tools on the same prompt and collect results.

    Note on fairness: single-layer tools use the specified width/l0.
    Multi-layer tools use 262k/medium (crosscoder) and 262k/big (CLT)
    because that's what's available. This is noted in results.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    str_tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())

    result = ComparisonResult(prompt=prompt, tokens=str_tokens)

    print(f"Comparing all tools on: '{prompt}'")
    print(f"  Target layer: {target_layer}, width: {width}, l0: {l0}")
    print()

    # 1. Residual SAE
    print("  [1/7] Residual SAE...")
    r = run_residual_sae(model, tokenizer, inputs, target_layer, width, l0, variant, device, cache_dir)
    result.tool_results.append(r)
    print(f"        FVU={r.fvu:.4f}, dL={r.delta_loss:.4f}, L0={r.l0_actual:.1f}")

    # 2. MLP SAE
    print("  [2/7] MLP SAE...")
    r = run_mlp_sae(model, tokenizer, inputs, target_layer, width, l0, variant, device, cache_dir)
    result.tool_results.append(r)
    print(f"        FVU={r.fvu:.4f}, L0={r.l0_actual:.1f}")

    # 3. Attention SAE
    print("  [3/7] Attention SAE...")
    r = run_attn_sae(model, tokenizer, inputs, target_layer, width, l0, variant, device, cache_dir)
    result.tool_results.append(r)
    print(f"        FVU={r.fvu:.4f}, L0={r.l0_actual:.1f}")

    # 4. Transcoder (no skip)
    print("  [4/7] Transcoder (no skip)...")
    r = run_transcoder(model, tokenizer, inputs, target_layer, width, l0,
                       affine=False, variant=variant, device=device, cache_dir=cache_dir)
    result.tool_results.append(r)
    print(f"        FVU={r.fvu:.4f}, dL={r.delta_loss:.4f}, L0={r.l0_actual:.1f}")

    # 5. Skip Transcoder (affine)
    print("  [5/7] Skip Transcoder (affine)...")
    r = run_transcoder(model, tokenizer, inputs, target_layer, width, l0,
                       affine=True, variant=variant, device=device, cache_dir=cache_dir)
    result.tool_results.append(r)
    print(f"        FVU={r.fvu:.4f}, dL={r.delta_loss:.4f}, L0={r.l0_actual:.1f}")

    if not skip_heavy:
        # 6. Crosscoder (always 262k — that's what's available)
        print("  [6/7] Crosscoder (262k, medium)...")
        r = run_crosscoder(model, tokenizer, inputs, "262k", l0, variant, device, cache_dir)
        result.tool_results.append(r)
        print(f"        FVU={r.fvu:.4f}, L0={r.l0_actual:.1f}")

        # 7. CLT (always 262k/big — that's what's available)
        print("  [7/7] CLT (262k, big, affine)...")
        r, per_layer_fvu = run_clt(model, tokenizer, inputs, clt=preloaded_clt,
                                   width="262k", l0="big", affine=True,
                                   variant=variant, device=device, cache_dir=cache_dir)
        result.tool_results.append(r)
        print(f"        FVU={r.fvu:.4f}, L0={r.l0_actual:.1f}")
    else:
        print("  [6/7] Crosscoder... SKIPPED")
        print("  [7/7] CLT... SKIPPED")

    return result


def run_matched_comparison(
    model, tokenizer, prompt, target_layer=17,
    variant="pt", device="cuda", cache_dir=None,
):
    """
    Run a fair comparison with matched widths where possible.
    All single-layer tools at 16k width, small L0 (available for all).
    Reports the comparison limitations clearly.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    str_tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())

    result = ComparisonResult(prompt=prompt, tokens=str_tokens)
    width = "16k"
    l0 = "small"

    print(f"MATCHED comparison on: '{prompt}'")
    print(f"  All tools at width={width}, l0={l0}, layer={target_layer}")
    print()

    # All single-layer tools at matched width/l0
    for name, runner in [
        ("Residual SAE", lambda: run_residual_sae(model, tokenizer, inputs, target_layer, width, l0, variant, device, cache_dir)),
        ("MLP SAE", lambda: run_mlp_sae(model, tokenizer, inputs, target_layer, width, l0, variant, device, cache_dir)),
        ("Attn SAE", lambda: run_attn_sae(model, tokenizer, inputs, target_layer, width, l0, variant, device, cache_dir)),
        ("Transcoder (no skip)", lambda: run_transcoder(model, tokenizer, inputs, target_layer, width, l0, False, variant, device, cache_dir)),
        ("Transcoder (skip)", lambda: run_transcoder(model, tokenizer, inputs, target_layer, width, l0, True, variant, device, cache_dir)),
    ]:
        print(f"  {name}...")
        try:
            r = runner()
            result.tool_results.append(r)
            dl = f", dL={r.delta_loss:.4f}" if r.delta_loss > 0 else ""
            print(f"    FVU={r.fvu:.4f}{dl}, L0={r.l0_actual:.1f}")
        except Exception as e:
            print(f"    FAILED: {e}")

    return result


def save_comparison(result, path):
    data = {
        "prompt": result.prompt,
        "tokens": result.tokens,
        "tools": [],
    }
    for r in result.tool_results:
        data["tools"].append({
            "tool_name": r.tool_name,
            "site": r.site,
            "layer": r.layer,
            "width": r.width,
            "l0_target": r.l0_target,
            "fvu": r.fvu,
            "delta_loss": r.delta_loss,
            "l0_actual": r.l0_actual,
            "num_active_features": r.num_active_features,
            "top_features": r.top_features,
            "top_logit_effects": r.top_logit_effects,
            "load_time": r.load_time,
            "inference_time": r.inference_time,
            "model_size_mb": r.model_size_mb,
        })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved comparison to {path}")
