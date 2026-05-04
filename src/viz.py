"""
Visualization utilities for Gemma Scope 2 analysis.

Generates:
  1. Feature activation heatmaps (which features fire on which tokens)
  2. Reconstruction error heatmaps (per-layer, per-token)
  3. Cross-tool comparison matrices
  4. Attribution graph diagrams (text-based and exportable)
  5. Token-level activation highlighting

All plots saved as PNG/PDF for paper inclusion.
Uses matplotlib (no interactive dependencies).
"""

import torch
import numpy as np
import json
import os
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")


# ============================================================
# 1. Feature activation heatmap
# ============================================================

def plot_feature_activations(
    features: torch.Tensor,
    tokens: list,
    top_k: int = 20,
    title: str = "Feature Activations",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    """
    Plot heatmap of top-k feature activations across token positions.

    Args:
        features: (seq_len, d_sae) — feature activations
        tokens: list of token strings
        top_k: number of top features to show
        title: plot title
        save_path: if provided, save figure here
    """
    if not HAS_MPL:
        print("matplotlib required for plotting")
        return

    features = features.float().cpu()

    # Find top-k features by max activation
    max_acts = features[1:].max(dim=0).values  # skip BOS
    top_vals, top_idxs = max_acts.topk(min(top_k, max_acts.shape[0]))

    # Build heatmap data
    data = features[:, top_idxs].numpy()  # (seq_len, top_k)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_idxs)))
    ax.set_yticklabels([f"f{idx.item()}" for idx in top_idxs], fontsize=8)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Feature Index")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Activation")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 2. Reconstruction error heatmap (per-layer, per-token)
# ============================================================

def plot_reconstruction_heatmap(
    recon_error: torch.Tensor,
    tokens: list,
    layer_labels: Optional[list] = None,
    title: str = "Reconstruction Error",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    skip_bos: bool = True,
):
    """
    Plot per-token, per-layer reconstruction error heatmap.

    Args:
        recon_error: (seq_len, num_layers) — MSE per position per layer
        tokens: list of token strings
        layer_labels: labels for y-axis (default: Layer 0, 1, ...)
        title: plot title
        save_path: if provided, save figure here
    """
    if not HAS_MPL:
        return

    data = recon_error.float().cpu().numpy()
    if skip_bos:
        data = data[1:]
        tokens = tokens[1:]

    n_layers = data.shape[1]
    if layer_labels is None:
        layer_labels = [f"L{i}" for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.T, aspect="auto", cmap="hot", interpolation="nearest")

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=6)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="MSE")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 3. Reconstruction differential heatmap (Believe it or Not)
# ============================================================

def plot_reconstruction_differential(
    differential: torch.Tensor,
    tokens: list,
    title: str = "Reconstruction Differential (Modified - Base)",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    """
    Plot reconstruction error differential between base and modified model.
    Positive values (red) = modified model computes something new here.
    Negative values (blue) = modified model computes less here.

    Args:
        differential: (seq_len, num_layers) — error_modified - error_base
    """
    if not HAS_MPL:
        return

    data = differential.float().cpu().numpy()[1:]  # skip BOS
    tokens_display = tokens[1:]

    vmax = np.abs(data).max()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.T, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(tokens_display)))
    ax.set_xticklabels(tokens_display, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(data.shape[1]))
    ax.set_yticklabels([f"L{i}" for i in range(data.shape[1])], fontsize=6)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Error Differential (+ = new computation)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 4. Cross-tool comparison bar chart
# ============================================================

def plot_tool_comparison(
    comparison_result,
    metric: str = "fvu",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """
    Bar chart comparing a metric across tools.

    Args:
        comparison_result: ComparisonResult object
        metric: "fvu", "l0_actual", "delta_loss", "num_active_features"
    """
    if not HAS_MPL:
        return

    names = [r.tool_name for r in comparison_result.tool_results]
    values = [getattr(r, metric, 0.0) for r in comparison_result.tool_results]

    # Filter out N/A (0.0 for delta_loss when not computed)
    if metric == "delta_loss":
        valid = [(n, v) for n, v in zip(names, values) if v > 0]
        if not valid:
            print("No valid delta_loss values to plot")
            return
        names, values = zip(*valid)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(names)), values, color="steelblue", edgecolor="black")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper().replace("_", " "))

    if title is None:
        title = f"{metric.upper()} Comparison: '{comparison_result.prompt}'"
    ax.set_title(title)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 5. Attribution graph text rendering
# ============================================================

def render_attribution_graph_text(graph, max_edges: int = 20) -> str:
    """
    Render attribution graph as formatted text for terminal/paper.
    Shows the most important paths from features to outputs.
    """
    lines = [
        f"Attribution Graph: '{graph.prompt}'",
        f"Tokens: {' | '.join(graph.tokens)}",
        f"Features: {len(graph.feature_nodes)}, Outputs: {len(graph.output_nodes)}, Edges: {len(graph.edges)}",
        "",
    ]

    # Group edges by output node
    output_edges = {}
    for e in graph.edges:
        if e.target_id in graph.output_nodes:
            output_edges.setdefault(e.target_id, []).append(e)

    # Sort outputs by total incoming weight
    sorted_outputs = sorted(
        output_edges.items(),
        key=lambda x: sum(abs(e.weight) for e in x[1]),
        reverse=True,
    )

    for out_id, edges in sorted_outputs[:5]:
        out_node = graph.output_nodes[out_id]
        total_w = sum(e.weight for e in edges)
        lines.append(f"  -> '{out_node.token_str}' (logit={out_node.logit_value:.2f}, total_weight={total_w:+.0f})")

        edges.sort(key=lambda e: abs(e.weight), reverse=True)
        for e in edges[:5]:
            src = graph.feature_nodes.get(e.source_id)
            if src:
                lines.append(f"     <- L{src.layer:02d}/f{src.feature_idx} "
                           f"(act={src.activation:.0f}) w={e.weight:+.0f}")

        # Check for multi-hop paths to this output
        ff_edges = [e2 for e2 in graph.edges
                    if e2.target_id in graph.feature_nodes
                    and any(e3.source_id == e2.target_id for e3 in edges[:5])]
        if ff_edges:
            ff_edges.sort(key=lambda e: abs(e.weight), reverse=True)
            for ff in ff_edges[:2]:
                mid = graph.feature_nodes.get(ff.target_id)
                src = graph.feature_nodes.get(ff.source_id)
                if src and mid:
                    lines.append(f"        <- L{src.layer:02d}/f{src.feature_idx} "
                               f"(act={src.activation:.0f}) -> L{mid.layer:02d}/f{mid.feature_idx} w={ff.weight:+.0f}")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# 6. Per-layer FVU line plot
# ============================================================

def plot_per_layer_fvu(
    per_layer_fvu: dict,
    title: str = "CLT FVU by Layer",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
):
    """Plot FVU across layers for CLT or crosscoder."""
    if not HAS_MPL:
        return

    layers = sorted(per_layer_fvu.keys())
    fvus = [per_layer_fvu[l] for l in layers]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(layers, fvus, "o-", color="steelblue", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("FVU")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 7. Token highlighting (HTML output)
# ============================================================

def highlight_tokens_html(
    tokens: list,
    values: list,
    title: str = "",
    colormap: str = "activation",
) -> str:
    """
    Generate HTML with color-highlighted tokens.

    Args:
        tokens: list of token strings
        values: list of floats (same length as tokens)
        colormap: "activation" (green), "error" (red), "divergence" (blue-red)
    """
    max_val = max(abs(v) for v in values) if values else 1.0
    if max_val == 0:
        max_val = 1.0

    spans = []
    for tok, val in zip(tokens, values):
        norm_val = val / max_val

        if colormap == "activation":
            bg = f"rgba(0,200,0,{abs(norm_val):.3f})"
        elif colormap == "error":
            bg = f"rgba(255,0,0,{abs(norm_val):.3f})"
        elif colormap == "divergence":
            if norm_val >= 0:
                bg = f"rgba(255,0,0,{norm_val:.3f})"
            else:
                bg = f"rgba(0,0,255,{-norm_val:.3f})"
        else:
            bg = f"rgba(128,128,128,{abs(norm_val):.3f})"

        tok_escaped = tok.replace("<", "&lt;").replace(">", "&gt;")
        spans.append(f'<span style="background:{bg};padding:2px 4px;">{tok_escaped}</span>')

    html = f"<div style='font-family:monospace;line-height:2;'>"
    if title:
        html += f"<b>{title}</b><br>"
    html += " ".join(spans)
    html += "</div>"
    return html


def save_html_report(
    sections: list,
    save_path: str,
    title: str = "Gemma Scope 2 Analysis",
):
    """
    Save multiple HTML sections to a single report file.

    Args:
        sections: list of HTML strings
        save_path: output file path
        title: page title
    """
    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
  .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
  h2 {{ color: #333; }}
</style>
</head><body>
<h1>{title}</h1>
"""
    for section in sections:
        html += f'<div class="section">{section}</div>\n'

    html += "</body></html>"

    with open(save_path, "w") as f:
        f.write(html)
    print(f"Saved report: {save_path}")
