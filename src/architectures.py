"""
SAE / Transcoder / CLT architectures.

Verified tensor shapes (from diagnostics/step2_inspect_shapes.py):

Single-layer (SAE, transcoder):
  w_enc:     (d_model, d_sae)           e.g. (1152, 65536)
  b_enc:     (d_sae,)
  threshold: (d_sae,)
  w_dec:     (d_sae, d_model)
  b_dec:     (d_model,)
  affine_skip_connection: (d_model, d_model)   [optional, transcoders only]

Multi-layer (crosscoder, CLT) — PER LAYER FILE:
  w_enc:     (d_model, d_sae_per_layer)   e.g. (1152, 10080) for 262k CLT
  b_enc:     (d_sae_per_layer,)
  threshold: (d_sae_per_layer,)
  w_dec:     (d_sae_per_layer, num_layers, d_model)  e.g. (10080, 26, 1152)
  b_dec:     (d_model,)
  affine_skip_connection: (d_model, d_model)   [optional]

After stacking all layers:
  w_enc:     (num_layers, d_model, d_sae_per_layer)
  w_dec:     (num_layers, d_sae_per_layer, num_layers, d_model)
  threshold: (num_layers, d_sae_per_layer)
  b_enc:     (num_layers, d_sae_per_layer)
  b_dec:     (num_layers, d_model)
"""

import torch
import torch.nn as nn
import einops


class JumpReLUSAE(nn.Module):
    """
    Single-layer JumpReLU SAE / Transcoder.

    Works for all single-layer models in Gemma Scope 2:
      - resid_post SAEs
      - mlp_out SAEs
      - attn_out SAEs
      - transcoders (with or without affine skip)

    The ONLY difference between an SAE and a transcoder is:
      - SAE: input = output target (reconstruct activations)
      - Transcoder: input = pre-MLP, output target = MLP output
      - Skip transcoder: same as transcoder + affine_skip_connection

    The architecture itself is identical.
    """

    def __init__(self, d_model: int, d_sae: int, has_skip: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.has_skip = has_skip

        self.w_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        if has_skip:
            self.affine_skip_connection = nn.Parameter(torch.zeros(d_model, d_model))
        else:
            self.affine_skip_connection = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse features.

        Args:
            x: (batch, seq, d_model) or (seq, d_model)
        Returns:
            features: same leading dims + (d_sae,), sparse and non-negative
        """
        pre_acts = x @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        return mask * torch.nn.functional.relu(pre_acts)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.

        Args:
            features: (..., d_sae)
        Returns:
            reconstructed: (..., d_model)
        """
        return features @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode → decode, optionally + skip.

        For SAEs: output should approximate x (the input).
        For transcoders: output should approximate MLP(x).
        """
        features = self.encode(x)
        recon = self.decode(features)
        if self.affine_skip_connection is not None:
            recon = recon + x @ self.affine_skip_connection
        return recon


class JumpReLUMultiLayerSAE(nn.Module):
    """
    Multi-layer JumpReLU model: Crosscoders and CLTs.

    Key architectural facts (verified):
      - Encoder is per-layer: each layer has its own (d_model, d_sae) encoder
      - Decoder is cross-layer: each layer's features decode to ALL layers
        w_dec shape per layer file: (d_sae_per_layer, num_layers, d_model)
      - For CLTs: num_layers = 26 (all layers of Gemma 3 1B)
      - For crosscoders: num_layers = 4 (layers 7, 13, 17, 22)
      - Causality is enforced in the WEIGHTS, not architecture
        (early-layer features should have near-zero decoder weights for earlier layers)
      - Affine skip connection is per-layer, same-layer only: (d_model, d_model)

    Width naming:
      - "262k" CLT = 262,080 total features = 10,080 per layer × 26 layers
      - "262k" crosscoder = 65,536 per layer × 4 layers = 262,144 total
    """

    def __init__(
        self,
        d_model: int,
        d_sae_per_layer: int,
        num_layers: int,
        has_skip: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae_per_layer = d_sae_per_layer
        self.num_layers = num_layers
        self.has_skip = has_skip

        self.w_enc = nn.Parameter(torch.zeros(num_layers, d_model, d_sae_per_layer))
        self.b_enc = nn.Parameter(torch.zeros(num_layers, d_sae_per_layer))
        self.threshold = nn.Parameter(torch.zeros(num_layers, d_sae_per_layer))
        # Decoder: (encoder_layer, features, decoder_layer, d_model)
        self.w_dec = nn.Parameter(torch.zeros(num_layers, d_sae_per_layer, num_layers, d_model))
        self.b_dec = nn.Parameter(torch.zeros(num_layers, d_model))

        if has_skip:
            self.affine_skip_connection = nn.Parameter(torch.zeros(num_layers, d_model, d_model))
        else:
            self.affine_skip_connection = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode multi-layer activations to sparse features.

        Args:
            x: (..., num_layers, d_model) — stacked activations from all layers
        Returns:
            features: (..., num_layers, d_sae_per_layer) — sparse features per layer
        """
        # Einstein summation: per-layer matrix multiply
        pre_acts = einops.einsum(
            x, self.w_enc,
            "... layer d_in, layer d_in d_sae -> ... layer d_sae"
        ) + self.b_enc
        mask = pre_acts > self.threshold
        return mask * torch.nn.functional.relu(pre_acts)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features to multi-layer reconstructions.

        Args:
            features: (..., num_layers_in, d_sae_per_layer)
        Returns:
            reconstructed: (..., num_layers_out, d_model)

        The decoder aggregates features from all encoder layers to reconstruct
        each output layer: y_hat[layer_out] = sum over layer_in of (W_dec[layer_in] @ features[layer_in])
        """
        return einops.einsum(
            features, self.w_dec,
            "... layer_in d_sae, layer_in d_sae layer_out d_model -> ... layer_out d_model"
        ) + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode → decode, optionally + skip.

        For crosscoders: output should approximate the input activations.
        For CLTs: output should approximate the MLP outputs at each layer.
        """
        features = self.encode(x)
        recon = self.decode(features)
        if self.affine_skip_connection is not None:
            # Skip is per-layer, same-layer only
            recon = recon + einops.einsum(
                x, self.affine_skip_connection,
                "... layer d_in, layer d_in d_out -> ... layer d_out"
            )
        return recon

    def encode_single_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Encode activations from a single layer only."""
        pre_acts = x @ self.w_enc[layer_idx] + self.b_enc[layer_idx]
        mask = pre_acts > self.threshold[layer_idx]
        return mask * torch.nn.functional.relu(pre_acts)

    def get_feature_decoder(self, enc_layer: int, feature_idx: int) -> torch.Tensor:
        """
        Get the decoder vector for a specific feature.

        Returns: (num_layers, d_model) — this feature's contribution to each output layer.
        Useful for understanding cross-layer effects of individual features.
        """
        return self.w_dec[enc_layer, feature_idx]  # (num_layers, d_model)
