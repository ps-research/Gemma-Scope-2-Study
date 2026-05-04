"""
Activation extraction hooks for Gemma 3 1B.

Hook points verified from:
  - Official tutorial (gather_*_activations functions)
  - CLT config.json: hf_hook_point_in/out fields
  - diagnostics/step2_inspect_shapes.py

Gemma 3 1B module hierarchy (relevant paths):
  model.model.layers[L]                          → residual stream output (resid_post)
  model.model.layers[L].pre_feedforward_layernorm  → output = pre-MLP (transcoder/CLT input)
  model.model.layers[L].post_feedforward_layernorm → output = MLP output (transcoder/CLT target)
  model.model.layers[L].self_attn.o_proj           → input = attn output pre-W_O (attn SAE target)

IMPORTANT: SAEs are NOT trained on the BOS token. Always skip position 0
when computing metrics (the tutorial does this with [1:] indexing).
"""

import torch
from functools import partial
from typing import Optional


def _gather_hook(mod, inputs, outputs, cache: dict, key: str, use_input: bool):
    """Generic hook that stores activations."""
    if use_input:
        acts = inputs[0].squeeze(0) if inputs[0].dim() == 3 else inputs[0]
    else:
        acts = outputs[0] if isinstance(outputs, tuple) else outputs
        if acts.dim() == 3:
            acts = acts.squeeze(0)
    cache[key] = acts


def gather_residual_activations(
    model, layer: int, inputs: torch.Tensor
) -> torch.Tensor:
    """
    Get post-MLP residual stream activations at a specific layer.
    Site: resid_post
    Hook: model.model.layers[layer] output

    Returns: (seq_len, d_model)
    """
    cache = {}
    handle = model.model.layers[layer].register_forward_hook(
        partial(_gather_hook, cache=cache, key="acts", use_input=False)
    )
    try:
        model.forward(inputs)
    finally:
        handle.remove()
    return cache["acts"]


def gather_mlp_out_activations(
    model, layer: int, inputs: torch.Tensor
) -> torch.Tensor:
    """
    Get MLP output activations (post feedforward layernorm).
    Site: mlp_out
    Hook: model.model.layers[layer].post_feedforward_layernorm output

    Returns: (seq_len, d_model)
    """
    cache = {}
    handle = model.model.layers[layer].post_feedforward_layernorm.register_forward_hook(
        partial(_gather_hook, cache=cache, key="acts", use_input=False)
    )
    try:
        model.forward(inputs)
    finally:
        handle.remove()
    return cache["acts"]


def gather_attn_out_activations(
    model, layer: int, inputs: torch.Tensor
) -> torch.Tensor:
    """
    Get attention output activations BEFORE W_O projection.
    Site: attn_out
    Hook: model.model.layers[layer].self_attn.o_proj INPUT (not output!)

    Returns: (seq_len, d_model)
    """
    cache = {}
    handle = model.model.layers[layer].self_attn.o_proj.register_forward_hook(
        partial(_gather_hook, cache=cache, key="acts", use_input=True)
    )
    try:
        model.forward(inputs)
    finally:
        handle.remove()
    return cache["acts"]


def gather_transcoder_activations(
    model, layer: int, inputs: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Get both transcoder input (pre-MLP) and target (MLP output).
    Site: transcoder
    Input hook: model.model.layers[layer].pre_feedforward_layernorm output
    Target hook: model.model.layers[layer].post_feedforward_layernorm output

    Returns: dict with keys "input" and "target", each (seq_len, d_model)
    """
    cache = {}
    handle_in = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(
        partial(_gather_hook, cache=cache, key="input", use_input=False)
    )
    handle_out = model.model.layers[layer].post_feedforward_layernorm.register_forward_hook(
        partial(_gather_hook, cache=cache, key="target", use_input=False)
    )
    try:
        model.forward(inputs)
    finally:
        handle_in.remove()
        handle_out.remove()
    return cache


def gather_crosscoder_activations(
    model, layers: list[int], inputs: torch.Tensor
) -> torch.Tensor:
    """
    Get residual stream activations at multiple layers for crosscoder.
    Site: resid_post at each of the crosscoder layers (e.g. [7, 13, 17, 22])

    Returns: (seq_len, num_layers, d_model)
    """
    cache = {}
    handles = []
    for layer in layers:
        handle = model.model.layers[layer].register_forward_hook(
            partial(_gather_hook, cache=cache, key=f"acts_{layer}", use_input=False)
        )
        handles.append(handle)
    try:
        model.forward(inputs)
    finally:
        for h in handles:
            h.remove()

    return torch.stack(
        [cache[f"acts_{layer}"] for layer in layers], dim=-2
    )


def gather_clt_activations(
    model, num_layers: int, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get ALL pre-MLP inputs and MLP outputs for CLT inference.
    Site: all 26 layers, both pre_feedforward_layernorm and post_feedforward_layernorm

    Returns:
        clt_inputs:  (seq_len, num_layers, d_model) — pre-MLP at each layer
        clt_targets: (seq_len, num_layers, d_model) — MLP output at each layer
    """
    cache = {}
    handles = []
    for layer in range(num_layers):
        h_in = model.model.layers[layer].pre_feedforward_layernorm.register_forward_hook(
            partial(_gather_hook, cache=cache, key=f"in_{layer}", use_input=False)
        )
        h_out = model.model.layers[layer].post_feedforward_layernorm.register_forward_hook(
            partial(_gather_hook, cache=cache, key=f"out_{layer}", use_input=False)
        )
        handles.extend([h_in, h_out])
    try:
        model.forward(inputs)
    finally:
        for h in handles:
            h.remove()

    clt_inputs = torch.stack(
        [cache[f"in_{layer}"] for layer in range(num_layers)], dim=-2
    )
    clt_targets = torch.stack(
        [cache[f"out_{layer}"] for layer in range(num_layers)], dim=-2
    )
    return clt_inputs, clt_targets
