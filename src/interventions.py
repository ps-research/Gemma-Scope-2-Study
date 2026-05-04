"""
Intervention experiments for causal validation of circuits.

Supports:
  1. Feature ablation — zero out specific features, measure downstream effects
  2. Feature steering — add scaled decoder vectors during generation
  3. Feature clamping — set feature activations to fixed values
  4. Feature swapping — replace features from one prompt with another
  5. Supernode interventions — group related features, intervene together
  6. Per-layer CLT patching — replace MLP output at specific layers

All interventions use PyTorch hooks on the Gemma 3 1B forward pass.
For CLT-based interventions, we modify the CLT's sparse feature activations
and reconstruct, then patch the result back into the model.

IMPORTANT: Interventions on the residual stream are "stronger" than on MLP
outputs because residual modifications propagate through ALL remaining layers.
Transcoder/CLT interventions only affect one MLP's contribution.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import partial
import copy

from src.hooks import (
    gather_residual_activations,
    gather_clt_activations,
    gather_transcoder_activations,
)
from src.metrics import cross_entropy_loss


# ============================================================
# Data structures
# ============================================================

@dataclass
class InterventionResult:
    """Result of an intervention experiment."""
    intervention_type: str
    description: str

    # Logit changes
    loss_clean: float = 0.0
    loss_intervened: float = 0.0
    delta_loss: float = 0.0

    # Top prediction changes
    top_tokens_clean: list = field(default_factory=list)    # [(token_str, logit)]
    top_tokens_intervened: list = field(default_factory=list)

    # Per-token losses (for detailed analysis)
    per_token_loss_clean: list = field(default_factory=list)
    per_token_loss_intervened: list = field(default_factory=list)

    # Generation outputs (for steering experiments)
    generation_clean: str = ""
    generation_intervened: str = ""


# ============================================================
# Core intervention helpers
# ============================================================

def _get_top_predictions(logits, tokenizer, k=10):
    """Get top-k predicted tokens from logits at last position."""
    top_vals, top_ids = logits[-1].topk(k)
    return [(tokenizer.decode([tid.item()]).strip(), v.item())
            for v, tid in zip(top_vals, top_ids)]


def _run_clean(model, inputs, tokenizer):
    """Run clean forward pass, return logits and top predictions."""
    with torch.no_grad():
        logits = model(inputs).logits[0]
    tokens = inputs[0]
    loss = cross_entropy_loss(logits, tokens)
    top_preds = _get_top_predictions(logits, tokenizer)
    return logits, loss, top_preds


# ============================================================
# 1. Feature ablation (SAE-based)
# ============================================================

def ablate_sae_features(
    model, sae, tokenizer, inputs,
    layer: int,
    feature_indices: list,
    site: str = "resid_post",
) -> InterventionResult:
    """
    Zero out specific SAE features and measure effect.

    Process:
      1. Run SAE encoder to get features
      2. Zero out specified features
      3. Decode back to activation space
      4. Patch the modified activations into the model
      5. Measure change in predictions

    Args:
        feature_indices: list of feature indices to ablate
    """
    # Clean pass
    logits_clean, loss_clean, top_clean = _run_clean(model, inputs, tokenizer)

    # Get activations and encode
    from src.hooks import gather_residual_activations
    acts = gather_residual_activations(model, layer, inputs)
    features = sae.encode(acts.float())

    # Ablate specified features
    features_modified = features.clone()
    for idx in feature_indices:
        features_modified[:, idx] = 0.0

    # Reconstruct with ablated features
    recon_clean = sae.decode(features)
    recon_ablated = sae.decode(features_modified)

    # The intervention = difference between clean and ablated reconstruction
    # We add this difference to the original activations
    delta = recon_ablated - recon_clean

    def _hook(mod, inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        if tensor.dim() == 3:
            tensor[0, 1:] += delta[1:].to(tensor.dtype)
        else:
            tensor[1:] += delta[1:].to(tensor.dtype)
        return out

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            logits_intervened = model(inputs).logits[0]
    finally:
        handle.remove()

    loss_intervened = cross_entropy_loss(logits_intervened, inputs[0])
    top_intervened = _get_top_predictions(logits_intervened, tokenizer)

    return InterventionResult(
        intervention_type="ablation",
        description=f"Ablated {len(feature_indices)} features at layer {layer}: {feature_indices[:5]}{'...' if len(feature_indices) > 5 else ''}",
        loss_clean=loss_clean.mean().item(),
        loss_intervened=loss_intervened.mean().item(),
        delta_loss=loss_intervened.mean().item() - loss_clean.mean().item(),
        top_tokens_clean=top_clean,
        top_tokens_intervened=top_intervened,
        per_token_loss_clean=loss_clean.tolist(),
        per_token_loss_intervened=loss_intervened.tolist(),
    )


# ============================================================
# 2. Feature steering (generation-time)
# ============================================================

def steer_with_feature(
    model, sae, tokenizer, inputs,
    layer: int,
    feature_idx: int,
    coeff: float = 0.1,
    max_new_tokens: int = 50,
    steering_layer: Optional[int] = None,
) -> InterventionResult:
    """
    Add a scaled decoder vector during generation to steer model behavior.

    The steering vector is the SAE decoder column for the specified feature,
    scaled by coeff * ||residual_stream_norm||.

    Uses the technique from the official Gemma Scope 2 tutorial:
    during each generation step, add coeff * norm * decoder_vector to the
    residual stream at the steering layer.

    Args:
        layer: SAE layer (for the decoder vector)
        feature_idx: which feature to steer with
        coeff: steering strength (0.1 = subtle, 0.5 = strong, >1.0 = may break coherence)
        steering_layer: which layer to apply steering at (default: same as SAE layer)
    """
    if steering_layer is None:
        steering_layer = layer

    decoder_vector = sae.w_dec[feature_idx].detach()

    # Clean generation
    with torch.no_grad():
        clean_output = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_clean = tokenizer.decode(clean_output[0], skip_special_tokens=True)

    # Steered generation
    def _steering_hook(mod, inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        if tensor.dim() == 3:
            # During generation, tensor shape is (1, seq_or_1, d_model)
            avg_norm = torch.norm(tensor[0, -1:], dim=-1, keepdim=True)
            tensor[0, -1:] += coeff * avg_norm * decoder_vector.to(tensor.dtype)
        else:
            avg_norm = torch.norm(tensor[-1:], dim=-1, keepdim=True)
            tensor[-1:] += coeff * avg_norm * decoder_vector.to(tensor.dtype)
        return out

    handle = model.model.layers[steering_layer].register_forward_hook(_steering_hook)
    try:
        with torch.no_grad():
            steered_output = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        handle.remove()

    gen_steered = tokenizer.decode(steered_output[0], skip_special_tokens=True)

    return InterventionResult(
        intervention_type="steering",
        description=f"Steered with feature {feature_idx} at layer {steering_layer}, coeff={coeff}",
        generation_clean=gen_clean,
        generation_intervened=gen_steered,
    )


# ============================================================
# 3. Feature clamping
# ============================================================

def clamp_sae_features(
    model, sae, tokenizer, inputs,
    layer: int,
    clamp_map: dict,
    site: str = "resid_post",
) -> InterventionResult:
    """
    Set specific features to fixed activation values.

    Args:
        clamp_map: {feature_idx: target_value}
            target_value=0 is equivalent to ablation
            target_value>0 amplifies the feature
            target_value<0 inverts the feature direction
    """
    logits_clean, loss_clean, top_clean = _run_clean(model, inputs, tokenizer)

    acts = gather_residual_activations(model, layer, inputs)
    features = sae.encode(acts.float())

    # Clamp features
    features_clamped = features.clone()
    for idx, value in clamp_map.items():
        features_clamped[:, idx] = value

    recon_clean = sae.decode(features)
    recon_clamped = sae.decode(features_clamped)
    delta = recon_clamped - recon_clean

    def _hook(mod, inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        if tensor.dim() == 3:
            tensor[0, 1:] += delta[1:].to(tensor.dtype)
        else:
            tensor[1:] += delta[1:].to(tensor.dtype)
        return out

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            logits_intervened = model(inputs).logits[0]
    finally:
        handle.remove()

    loss_intervened = cross_entropy_loss(logits_intervened, inputs[0])
    top_intervened = _get_top_predictions(logits_intervened, tokenizer)

    clamp_desc = ", ".join(f"f{k}={v:.1f}" for k, v in list(clamp_map.items())[:5])
    return InterventionResult(
        intervention_type="clamping",
        description=f"Clamped at layer {layer}: {clamp_desc}",
        loss_clean=loss_clean.mean().item(),
        loss_intervened=loss_intervened.mean().item(),
        delta_loss=loss_intervened.mean().item() - loss_clean.mean().item(),
        top_tokens_clean=top_clean,
        top_tokens_intervened=top_intervened,
        per_token_loss_clean=loss_clean.tolist(),
        per_token_loss_intervened=loss_intervened.tolist(),
    )


# ============================================================
# 4. Feature swapping
# ============================================================

def swap_features(
    model, sae, tokenizer,
    source_inputs, target_inputs,
    layer: int,
    feature_indices: list,
) -> InterventionResult:
    """
    Replace features from source prompt with features from target prompt.

    Example: swap "France" features from "The capital of France is"
    with "Germany" features from "The capital of Germany is"
    to see if the model now predicts "Berlin" instead of "Paris".

    Args:
        source_inputs: the prompt to modify (tokenized)
        target_inputs: the prompt to take features FROM (tokenized)
        feature_indices: which features to swap
    """
    # Clean pass on source
    logits_clean, loss_clean, top_clean = _run_clean(model, source_inputs, tokenizer)

    # Get features from both prompts
    acts_source = gather_residual_activations(model, layer, source_inputs)
    acts_target = gather_residual_activations(model, layer, target_inputs)

    feats_source = sae.encode(acts_source.float())
    feats_target = sae.encode(acts_target.float())

    # Swap specified features (use min seq length)
    min_seq = min(feats_source.shape[0], feats_target.shape[0])
    feats_swapped = feats_source.clone()
    for idx in feature_indices:
        feats_swapped[:min_seq, idx] = feats_target[:min_seq, idx]

    recon_clean = sae.decode(feats_source)
    recon_swapped = sae.decode(feats_swapped)
    delta = recon_swapped - recon_clean

    def _hook(mod, inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        if tensor.dim() == 3:
            tensor[0, 1:min_seq] += delta[1:min_seq].to(tensor.dtype)
        else:
            tensor[1:min_seq] += delta[1:min_seq].to(tensor.dtype)
        return out

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            logits_intervened = model(source_inputs).logits[0]
    finally:
        handle.remove()

    loss_intervened = cross_entropy_loss(logits_intervened, source_inputs[0])
    top_intervened = _get_top_predictions(logits_intervened, tokenizer)

    return InterventionResult(
        intervention_type="swap",
        description=f"Swapped {len(feature_indices)} features at layer {layer} from target prompt",
        loss_clean=loss_clean.mean().item(),
        loss_intervened=loss_intervened.mean().item(),
        delta_loss=loss_intervened.mean().item() - loss_clean.mean().item(),
        top_tokens_clean=top_clean,
        top_tokens_intervened=top_intervened,
    )


# ============================================================
# 5. CLT-based interventions
# ============================================================

def ablate_clt_features(
    model, clt, tokenizer, inputs,
    feature_specs: list,
    target_layers: Optional[list] = None,
) -> InterventionResult:
    """
    Ablate specific CLT features and patch reconstructions back.

    Args:
        feature_specs: list of (layer, feature_idx) tuples to ablate
        target_layers: which layers to patch (default: all layers where
                       ablated features have decoder weights)
    """
    from src.loader import GEMMA3_1B_NUM_LAYERS

    logits_clean, loss_clean, top_clean = _run_clean(model, inputs, tokenizer)

    # Get CLT activations
    clt_inputs, clt_targets = gather_clt_activations(model, GEMMA3_1B_NUM_LAYERS, inputs)
    if next(clt.parameters()).dtype == torch.float16:
        clt_inputs = clt_inputs.half()

    # Encode
    features = clt.encode(clt_inputs)

    # Ablate specified features
    features_ablated = features.clone()
    for layer_idx, feat_idx in feature_specs:
        features_ablated[:, layer_idx, feat_idx] = 0.0

    # Reconstruct both
    recon_clean = clt.forward(clt_inputs)
    # Manual decode of ablated features
    recon_ablated = clt.decode(features_ablated)
    if clt.affine_skip_connection is not None:
        import einops
        recon_ablated = recon_ablated + einops.einsum(
            clt_inputs, clt.affine_skip_connection,
            "... layer d_in, layer d_in d_out -> ... layer d_out"
        )

    delta = recon_ablated - recon_clean  # (seq, num_layers, d_model)

    if target_layers is None:
        # Patch all layers that are affected
        affected = set()
        for layer_idx, feat_idx in feature_specs:
            for out_layer in range(layer_idx, GEMMA3_1B_NUM_LAYERS):
                affected.add(out_layer)
        target_layers = sorted(affected)

    # Patch each target layer
    handles = []
    for tl in target_layers:
        def _hook(mod, inp, out, layer_idx=tl):
            tensor = out[0] if isinstance(out, tuple) else out
            d = delta[:, layer_idx].to(tensor.dtype)
            if tensor.dim() == 3:
                tensor[0, 1:] += d[1:]
            else:
                tensor[1:] += d[1:]
            return out

        h = model.model.layers[tl].post_feedforward_layernorm.register_forward_hook(_hook)
        handles.append(h)

    try:
        with torch.no_grad():
            logits_intervened = model(inputs).logits[0]
    finally:
        for h in handles:
            h.remove()

    loss_intervened = cross_entropy_loss(logits_intervened, inputs[0])
    top_intervened = _get_top_predictions(logits_intervened, tokenizer)

    specs_str = ", ".join(f"L{l}/f{f}" for l, f in feature_specs[:5])
    return InterventionResult(
        intervention_type="clt_ablation",
        description=f"Ablated CLT features: {specs_str}{'...' if len(feature_specs) > 5 else ''}",
        loss_clean=loss_clean.mean().item(),
        loss_intervened=loss_intervened.mean().item(),
        delta_loss=loss_intervened.mean().item() - loss_clean.mean().item(),
        top_tokens_clean=top_clean,
        top_tokens_intervened=top_intervened,
    )


# ============================================================
# 6. Supernode interventions
# ============================================================

def ablate_supernode(
    model, clt, tokenizer, inputs,
    supernode: list,
    name: str = "supernode",
) -> InterventionResult:
    """
    Ablate an entire supernode (group of related features).

    A supernode is a list of (layer, feature_idx) tuples that represent
    features grouped together based on semantic similarity or circuit
    connectivity (e.g., all "France-related" features across layers).

    This is just a convenience wrapper around ablate_clt_features
    with better naming for paper presentation.
    """
    result = ablate_clt_features(model, clt, tokenizer, inputs, supernode)
    result.description = f"Ablated supernode '{name}' ({len(supernode)} features)"
    return result


# ============================================================
# 7. Batch intervention runner
# ============================================================

def run_ablation_sweep(
    model, sae, tokenizer, inputs,
    layer: int,
    feature_indices: list,
    site: str = "resid_post",
) -> list:
    """
    Ablate features one at a time to find which have the largest effect.

    Returns list of (feature_idx, delta_loss, top_prediction_change)
    sorted by |delta_loss|.
    """
    results = []
    for idx in feature_indices:
        r = ablate_sae_features(model, sae, tokenizer, inputs, layer, [idx], site)
        # Check if top prediction changed
        clean_top = r.top_tokens_clean[0][0] if r.top_tokens_clean else ""
        interv_top = r.top_tokens_intervened[0][0] if r.top_tokens_intervened else ""
        changed = clean_top != interv_top
        results.append((idx, r.delta_loss, changed, clean_top, interv_top))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results


# ============================================================
# 8. Reconstruction differential (for Believe it or Not)
# ============================================================

def compute_reconstruction_differential(
    model_base, model_modified, clt,
    tokenizer, prompt, device="cuda",
) -> dict:
    """
    Compare CLT reconstruction quality between base and modified models.

    For each (layer, token_position), compute:
      recon_error_modified - recon_error_base

    Positive values = the modified model computes something new here
    that the pretrained CLT can't capture.

    Args:
        model_base: original Gemma 3 1B
        model_modified: GRPO-finetuned Gemma 3 1B
        clt: pretrained CLT (trained on base model)
        prompt: text to analyze

    Returns:
        dict with:
          - "differential": (seq_len, num_layers) tensor
          - "base_error": (seq_len, num_layers) tensor
          - "modified_error": (seq_len, num_layers) tensor
          - "tokens": list of token strings
    """
    from src.loader import GEMMA3_1B_NUM_LAYERS

    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    str_tokens = tokenizer.convert_ids_to_tokens(inputs[0].tolist())

    # Run CLT on base model
    clt_in_base, clt_tgt_base = gather_clt_activations(model_base, GEMMA3_1B_NUM_LAYERS, inputs)
    if next(clt.parameters()).dtype == torch.float16:
        clt_in_base = clt_in_base.half()
        clt_tgt_base = clt_tgt_base.half()

    recon_base = clt.forward(clt_in_base)
    error_base = (recon_base - clt_tgt_base).float().pow(2).mean(dim=-1)  # (seq, layers)

    # Run CLT on modified model
    clt_in_mod, clt_tgt_mod = gather_clt_activations(model_modified, GEMMA3_1B_NUM_LAYERS, inputs)
    if next(clt.parameters()).dtype == torch.float16:
        clt_in_mod = clt_in_mod.half()
        clt_tgt_mod = clt_tgt_mod.half()

    recon_mod = clt.forward(clt_in_mod)
    error_mod = (recon_mod - clt_tgt_mod).float().pow(2).mean(dim=-1)  # (seq, layers)

    differential = error_mod - error_base  # positive = new computation in modified model

    return {
        "differential": differential.cpu(),
        "base_error": error_base.cpu(),
        "modified_error": error_mod.cpu(),
        "tokens": str_tokens,
    }


# ============================================================
# Pretty printing
# ============================================================

def print_intervention_result(result: InterventionResult, tokenizer=None):
    """Print an intervention result in a readable format."""
    print(f"\n  Intervention: {result.description}")
    print(f"  Type: {result.intervention_type}")

    if result.loss_clean > 0:
        print(f"  Loss clean:     {result.loss_clean:.4f}")
        print(f"  Loss intervened: {result.loss_intervened:.4f}")
        print(f"  Delta loss:     {result.delta_loss:+.4f}")

    if result.top_tokens_clean:
        print(f"  Top predictions:")
        print(f"    Clean:      {', '.join(f'{t}({v:.2f})' for t, v in result.top_tokens_clean[:5])}")
    if result.top_tokens_intervened:
        print(f"    Intervened: {', '.join(f'{t}({v:.2f})' for t, v in result.top_tokens_intervened[:5])}")

    if result.generation_clean:
        print(f"  Generation (clean):     {result.generation_clean[:200]}")
    if result.generation_intervened:
        print(f"  Generation (intervened): {result.generation_intervened[:200]}")
