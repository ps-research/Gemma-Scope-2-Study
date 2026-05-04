"""
Gemma Scope 2 Study — Core Infrastructure
==========================================
Shared loading, inference, and evaluation code for all 4 workshop papers:
  1. AI Anatomy
  2. MI User
  3. MI Detective
  4. Believe it or Not

All architecture details verified from:
  - Official Gemma Scope 2 tutorial (Google DeepMind)
  - Direct inspection of HuggingFace checkpoint tensors
  - Gemma Scope 2 technical paper (Dec 2025)

Hardware target: 1x A100 80GB (CUDA_VISIBLE_DEVICES=1)
Base model: Gemma 3 1B (d_model=1152, 26 layers, vocab=262144)
"""

from src.architectures import JumpReLUSAE, JumpReLUMultiLayerSAE
from src.loader import (
    load_gemma3_1b,
    load_sae,
    load_transcoder,
    load_crosscoder,
    load_clt,
)
from src.hooks import (
    gather_residual_activations,
    gather_mlp_out_activations,
    gather_attn_out_activations,
    gather_transcoder_activations,
    gather_crosscoder_activations,
    gather_clt_activations,
)
from src.metrics import (
    compute_fvu,
    compute_l0,
    compute_delta_loss,
    cross_entropy_loss,
)
