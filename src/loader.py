"""
Loading functions for Gemma 3 1B and all Gemma Scope 2 tools.

File path conventions (verified from HuggingFace repo):
  resid_post/layer_{L}_width_{W}_l0_{L0}/params.safetensors
  mlp_out/layer_{L}_width_{W}_l0_{L0}/params.safetensors
  attn_out/layer_{L}_width_{W}_l0_{L0}/params.safetensors
  transcoder/layer_{L}_width_{W}_l0_{L0}[_affine]/params.safetensors
  crosscoder/layer_7_13_17_22_width_{W}_l0_{L0}/params_layer_{0-3}.safetensors
  clt/width_{W}_l0_{L0}[_affine]/params_layer_{0-25}.safetensors

Width values: 16k, 65k, 262k, 1m
L0 values: small, medium, big

Actual feature counts (verified):
  SAE/transcoder 65k  = 65,536 features
  CLT 262k = 10,080 features/layer × 26 layers = 262,080 total
  Crosscoder 262k = 65,536 features/layer × 4 layers = 262,144 total
"""

import os
import json
import torch
from typing import Literal, Optional
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from src.architectures import JumpReLUSAE, JumpReLUMultiLayerSAE


# ============================================================
# Constants
# ============================================================

GEMMA3_1B_D_MODEL = 1152
GEMMA3_1B_NUM_LAYERS = 26
GEMMA3_1B_CROSSCODER_LAYERS = [7, 13, 17, 22]

DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cache"
)


# ============================================================
# Base model loading
# ============================================================

def load_gemma3_1b(
    variant: Literal["pt", "it"] = "pt",
    device: str = "cuda",
    dtype=torch.bfloat16,
) -> tuple:
    """
    Load Gemma 3 1B and its tokenizer.

    Args:
        variant: "pt" for pretrained, "it" for instruction-tuned
        device: target device
        dtype: model precision (bf16 recommended, ~1.9 GB)

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = f"google/gemma-3-1b-{variant}"
    print(f"Loading {model_name} in {dtype}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    params = sum(p.numel() for p in model.parameters())
    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    print(f"  Loaded: {params/1e6:.0f}M params, {mem_gb:.2f} GB on {device}")

    return model, tokenizer


# ============================================================
# Single-layer model loading
# ============================================================

def _load_single_layer(
    category: str,
    layer: int,
    width: str,
    l0: str,
    affine: bool = False,
    variant: Literal["pt", "it"] = "pt",
    device: str = "cuda",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> JumpReLUSAE:
    """Internal: load any single-layer SAE or transcoder."""
    affine_str = "_affine" if affine else ""
    repo_id = f"google/gemma-scope-2-1b-{variant}"
    filename = f"{category}/layer_{layer}_width_{width}_l0_{l0}{affine_str}/params.safetensors"

    print(f"Loading {repo_id} / {filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )

    params = load_file(path, device="cpu")
    d_model, d_sae = params["w_enc"].shape
    has_skip = "affine_skip_connection" in params

    sae = JumpReLUSAE(d_model, d_sae, has_skip=has_skip)
    sae.load_state_dict(params)
    sae = sae.to(device)
    sae.eval()

    mem_mb = sum(p.numel() * p.element_size() for p in sae.parameters()) / (1024**2)
    print(f"  Loaded: d_model={d_model}, d_sae={d_sae}, skip={has_skip}, {mem_mb:.1f} MB")

    return sae


def load_sae(
    layer: int,
    site: str = "resid_post",
    width: str = "65k",
    l0: str = "medium",
    variant: Literal["pt", "it"] = "pt",
    device: str = "cuda",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> JumpReLUSAE:
    """
    Load a single-layer SAE.

    Args:
        layer: transformer layer index (0-25)
        site: which activation site
        width: feature width (16k, 65k, 262k, 1m)
        l0: sparsity target (small, medium, big)
        variant: pt or it
        device: target device
    """
    return _load_single_layer(site, layer, width, l0, False, variant, device, cache_dir)


def load_transcoder(
    layer: int,
    width: str = "65k",
    l0: str = "medium",
    affine: bool = True,
    variant: Literal["pt", "it"] = "pt",
    device: str = "cuda",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> JumpReLUSAE:
    """
    Load a single-layer transcoder (with or without affine skip).

    Args:
        layer: transformer layer index (0-25)
        width: feature width
        l0: sparsity target
        affine: whether to load the affine skip variant
        variant: pt or it
        device: target device
    """
    return _load_single_layer("transcoder", layer, width, l0, affine, variant, device, cache_dir)


# ============================================================
# Multi-layer model loading
# ============================================================

def load_crosscoder(
    width: str = "262k",
    l0: str = "medium",
    variant: Literal["pt", "it"] = "pt",
    device: str = "cuda",
    half_precision: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> JumpReLUMultiLayerSAE:
    """
    Load 4-layer weakly causal crosscoder (layers 7, 13, 17, 22).

    Crosscoder features: 65,536 per layer for "262k" width.
    Total on GPU: ~6 GB fp32, ~3 GB fp16.
    """
    layers = GEMMA3_1B_CROSSCODER_LAYERS
    num_layers = len(layers)
    layer_str = "_".join(str(l) for l in layers)
    repo_id = f"google/gemma-scope-2-1b-{variant}"
    subcategory = f"layer_{layer_str}_width_{width}_l0_{l0}"

    print(f"Loading crosscoder: {repo_id} / crosscoder/{subcategory}")
    print(f"  Layers: {layers}")

    params_list = []
    for idx in range(num_layers):
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"crosscoder/{subcategory}/params_layer_{idx}.safetensors",
            cache_dir=cache_dir,
        )
        params_list.append(load_file(path, device="cpu"))

    # Stack along layer dimension
    params = {
        k: torch.stack([p[k] for p in params_list])
        for k in params_list[0].keys()
    }

    d_model, d_sae = params["w_enc"].shape[1:]
    has_skip = "affine_skip_connection" in params

    cc = JumpReLUMultiLayerSAE(d_model, d_sae, num_layers, has_skip)
    cc.load_state_dict(params)

    if half_precision:
        cc = cc.half()

    cc = cc.to(device)
    cc.eval()

    mem_gb = sum(p.numel() * p.element_size() for p in cc.parameters()) / (1024**3)
    print(f"  Loaded: d_model={d_model}, d_sae_per_layer={d_sae}, layers={num_layers}, {mem_gb:.2f} GB")

    return cc


def load_clt(
    width: str = "262k",
    l0: str = "medium",
    affine: bool = True,
    variant: Literal["pt", "it"] = "pt",
    device: str = "cuda",
    half_precision: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> JumpReLUMultiLayerSAE:
    """
    Load cross-layer transcoder (all 26 layers).

    Memory (verified from actual files):
      262k, fp32: ~30.4 GB
      262k, fp16: ~15.2 GB  ← recommended
      524k, fp32: ~60.7 GB
      524k, fp16: ~30.4 GB

    Args:
        width: "262k" (10,080 features/layer) or "524k" (20,160 features/layer)
        l0: "medium" (L0≈50) or "big" (L0≈150)
        affine: whether to load affine skip variant (recommended for cleaner circuits)
        variant: pt or it
        device: target device
        half_precision: load in fp16 (strongly recommended to fit on single GPU)
    """
    num_layers = GEMMA3_1B_NUM_LAYERS
    affine_str = "_affine" if affine else ""
    repo_id = f"google/gemma-scope-2-1b-{variant}"
    subcategory = f"width_{width}_l0_{l0}{affine_str}"

    print(f"Loading CLT: {repo_id} / clt/{subcategory}")
    print(f"  All {num_layers} layers, half_precision={half_precision}")

    params_list = []
    for idx in range(num_layers):
        path = hf_hub_download(
            repo_id=repo_id,
            filename=f"clt/{subcategory}/params_layer_{idx}.safetensors",
            cache_dir=cache_dir,
        )
        p = load_file(path, device="cpu")
        if half_precision:
            p = {k: v.half() for k, v in p.items()}
        params_list.append(p)
        if (idx + 1) % 5 == 0 or idx == num_layers - 1:
            print(f"    Loaded layer {idx + 1}/{num_layers}")

    # Stack along layer dimension
    params = {
        k: torch.stack([p[k] for p in params_list])
        for k in params_list[0].keys()
    }

    # Free the list to save RAM during transfer to GPU
    del params_list

    d_model, d_sae = params["w_enc"].shape[1:]
    has_skip = "affine_skip_connection" in params

    clt = JumpReLUMultiLayerSAE(d_model, d_sae, num_layers, has_skip)

    # Match precision for state dict loading
    if half_precision:
        clt = clt.half()

    clt.load_state_dict(params)
    del params

    clt = clt.to(device)
    clt.eval()

    mem_gb = sum(p.numel() * p.element_size() for p in clt.parameters()) / (1024**3)
    print(f"  Loaded: d_model={d_model}, d_sae_per_layer={d_sae}, {mem_gb:.2f} GB on {device}")

    return clt


def load_config(
    category: str,
    subcategory: str,
    variant: Literal["pt", "it"] = "pt",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict:
    """Load the config.json for any Gemma Scope 2 checkpoint."""
    repo_id = f"google/gemma-scope-2-1b-{variant}"
    path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{category}/{subcategory}/config.json",
        cache_dir=cache_dir,
    )
    with open(path) as f:
        return json.load(f)
