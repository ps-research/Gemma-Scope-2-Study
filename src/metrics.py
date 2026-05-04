"""
Evaluation metrics for Gemma Scope 2 models.

All metrics verified against the official tutorial implementation.

IMPORTANT: Always exclude the BOS token (position 0) from metrics.
SAEs are not trained on BOS and give nonsensical outputs for it.
The tutorial does this with [1:] slicing.
"""

import torch
from typing import Optional


def compute_fvu(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    skip_bos: bool = True,
) -> torch.Tensor:
    """
    Fraction of Variance Unexplained.

    FVU = MSE(recon, target) / Var(target)

    Lower is better. 0.0 = perfect reconstruction.
    FVU is purely a reconstruction metric — it does NOT measure
    downstream causal impact (use delta_loss for that).

    Args:
        reconstruction: (..., d_model) reconstructed activations
        target: (..., d_model) original activations
        skip_bos: if True, exclude position 0

    Returns:
        scalar FVU value
    """
    if skip_bos:
        reconstruction = reconstruction[1:]
        target = target[1:]

    target_f = target.float()
    recon_f = reconstruction.float()

    mse = torch.mean((recon_f - target_f) ** 2)
    var = target_f.var()

    return mse / var


def compute_l0(
    features: torch.Tensor,
    skip_bos: bool = True,
) -> torch.Tensor:
    """
    Average L0 norm (number of active features per token).

    Args:
        features: (..., d_sae) or (..., num_layers, d_sae) for multi-layer
        skip_bos: if True, exclude position 0

    Returns:
        scalar average L0
    """
    if skip_bos:
        features = features[1:]

    # For multi-layer, sum over both layer and feature dims
    if features.dim() >= 3:
        # (..., layers, features) — sum over all feature dims
        active = (features > 0).float().sum(dim=tuple(range(1, features.dim())))
    else:
        active = (features > 0).float().sum(dim=-1)

    return active.mean()


def cross_entropy_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Per-token cross-entropy loss (no reduction).

    Args:
        logits: (seq_len, vocab_size)
        tokens: (seq_len,) token ids

    Returns:
        (seq_len - 1,) per-token losses (shifted by 1 for next-token prediction)
    """
    log_probs = logits[:-1].log_softmax(dim=-1)
    target_tokens = tokens[1:]
    correct_log_probs = log_probs[torch.arange(len(target_tokens)), target_tokens]
    return -correct_log_probs


def compute_delta_loss(
    model,
    sae,
    layer: int,
    inputs: torch.Tensor,
    site: str = "resid_post",
) -> dict[str, float]:
    """
    Compute delta loss: how much model predictions degrade when
    splicing in SAE/transcoder reconstructions.

    This is the key FUNCTIONAL metric — it measures causal impact
    of reconstruction errors, not just raw reconstruction quality.

    Args:
        model: Gemma 3 1B model
        sae: loaded JumpReLUSAE (SAE or transcoder)
        layer: target layer
        inputs: tokenized input (1, seq_len)
        site: "resid_post", "mlp_out", or "transcoder"

    Returns:
        dict with "loss_clean", "loss_patched", "delta_loss"
    """
    from functools import partial

    # Clean forward pass with cached activations
    if site == "resid_post":
        model_output = model.forward(inputs, output_hidden_states=True)
        logits_clean = model_output.logits[0]
        input_acts = model_output.hidden_states[layer + 1][0]
        recon = sae.forward(input_acts.float())[1:]  # skip BOS

        def _hook(mod, inp, out):
            out[0][0, 1:] = recon
            return out

        hook_target = model.model.layers[layer]

    elif site == "transcoder":
        # Cache transcoder input
        cached = {}

        def _cache_hook(mod, inp, out):
            cached["input"] = out[0] if isinstance(out, tuple) else out

        h = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(_cache_hook)
        try:
            model_output = model.forward(inputs)
            logits_clean = model_output.logits[0]
        finally:
            h.remove()

        recon = sae.forward(cached["input"][1:].float())

        def _hook(mod, inp, out):
            out_t = out[0] if isinstance(out, tuple) else out
            out_t[1:] = recon
            return out

        hook_target = model.model.layers[layer].post_feedforward_layernorm

    else:
        raise ValueError(f"Unsupported site for delta_loss: {site}")

    # Patched forward pass
    handle = hook_target.register_forward_hook(_hook)
    try:
        model_output_patched = model.forward(inputs)
    finally:
        handle.remove()

    logits_patched = model_output_patched.logits[0]

    # Compute losses
    tokens = inputs[0]
    loss_clean = cross_entropy_loss(logits_clean, tokens).mean().item()
    loss_patched = cross_entropy_loss(logits_patched, tokens).mean().item()

    return {
        "loss_clean": loss_clean,
        "loss_patched": loss_patched,
        "delta_loss": loss_patched - loss_clean,
    }
