"""
Professional Research Figure Templates.

Consistent style across all 3 papers. 600 dpi PDF export.

Figure types:
  1. Diverging heatmap (10×10 grid, relevance matrix)
  2. Layer landscape heatmap (tasks × layers)
  3. Scatter plot with regression (FVU vs delta loss)
  4. Grouped bar chart (skip fraction, behavioral metrics)
  5. Circuit / supernode diagram (attribution graphs)
  6. Text comparison panel (before/after ablation)
  7. Line plot (formation timeline)
  8. Formatted verdict table
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os


# ============================================================
# Global Style
# ============================================================

STYLE = {
    "font_family": "serif",
    "title_size": 13,
    "label_size": 11,
    "tick_size": 9,
    "legend_size": 9,
    "cell_text_size": 7,
    "dpi": 600,
    "fig_bg": "white",
    "grid_alpha": 0.15,
}

# Color palettes
PALETTE = {
    "tools": ["#2c3e50", "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
              "#9b59b6", "#1abc9c", "#e67e22", "#34495e", "#16a085"],
    "diverging": "RdBu_r",
    "sequential": "YlOrRd",
    "sequential_r": "YlOrRd_r",
    "binary": ["#f0f0f0", "#3498db", "#2c3e50"],
    "properties": ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"],
    "behaviors": ["#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad"],
}


def _apply_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent style to an axes."""
    ax.set_facecolor(STYLE["fig_bg"])
    if title:
        ax.set_title(title, fontsize=STYLE["title_size"], fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=STYLE["label_size"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=STYLE["label_size"])
    ax.tick_params(labelsize=STYLE["tick_size"])


def _save(fig, path):
    """Save figure as PDF (600 dpi) and PNG (150 dpi preview)."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", format="pdf",
                facecolor="white", edgecolor="none")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# 1. Diverging Heatmap (10×10 grid)
# ============================================================

def plot_diverging_heatmap(
    matrix, row_labels, col_labels,
    title="", label="Value",
    save_path="figure.pdf",
    figsize=None, annotate=True,
    cmap=None, center_zero=True,
):
    """
    Diverging heatmap — red for positive, blue for negative, white for zero.
    Used for: relevance grid, feature fingerprinting.
    """
    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (max(10, n_cols * 1.3), max(6, n_rows * 0.7))

    fig, ax = plt.subplots(figsize=figsize)

    if cmap is None:
        cmap = PALETTE["diverging"]

    if center_zero:
        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=STYLE["tick_size"])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=STYLE["tick_size"])

    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                if center_zero:
                    color = "white" if abs(val) > vmax * 0.5 else "black"
                else:
                    color = "white" if val > matrix.max() * 0.6 else "black"
                text = f"{val:+.2f}" if val != 0 else "0"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=STYLE["cell_text_size"], fontweight="bold", color=color)

    _apply_style(ax, title=title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(label, fontsize=STYLE["label_size"])
    cbar.ax.tick_params(labelsize=STYLE["tick_size"])

    fig.tight_layout()
    _save(fig, save_path)


# ============================================================
# 2. Layer Landscape Heatmap (tasks × layers)
# ============================================================

def plot_layer_landscape(
    matrix, task_labels,
    title="Feature Relevance Across Layers",
    label="|Max Logit Effect on Target|",
    save_path="figure.pdf",
    figsize=(16, 7),
    annotate_peaks=True,
):
    """
    Tasks × Layers heatmap with Gaussian smoothing look.
    Peak layer annotated per task.
    """
    n_tasks, n_layers = matrix.shape
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, aspect="auto", cmap=PALETTE["sequential"],
                   interpolation="bilinear")

    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xticklabels([str(i) for i in range(0, n_layers, 2)], fontsize=STYLE["tick_size"])
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_labels, fontsize=STYLE["tick_size"])

    if annotate_peaks:
        for i in range(n_tasks):
            peak = np.argmax(matrix[i])
            peak_val = matrix[i, peak]
            if peak_val > 0:
                ax.plot(peak, i, "k*", markersize=8)
                ax.text(peak + 0.3, i, f"L{peak}", fontsize=6, va="center",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8))

    _apply_style(ax, title=title, xlabel="Layer", ylabel="Cognitive Task")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(label, fontsize=STYLE["label_size"])
    cbar.ax.tick_params(labelsize=STYLE["tick_size"])

    fig.tight_layout()
    _save(fig, save_path)


# ============================================================
# 3. Scatter Plot with Regression (FVU vs Delta Loss)
# ============================================================

def plot_scatter_with_regression(
    data_groups,
    xlabel="FVU", ylabel="Delta Loss",
    title="FVU vs Delta Loss by Tool Type",
    save_path="figure.pdf",
    figsize=(10, 7),
):
    """
    Scatter plot with per-group regression lines.

    Args:
        data_groups: list of dicts with keys:
            "label", "x", "y", "color" (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, group in enumerate(data_groups):
        color = group.get("color", PALETTE["tools"][i % len(PALETTE["tools"])])
        x = np.array(group["x"])
        y = np.array(group["y"])

        ax.scatter(x, y, c=color, s=60, alpha=0.7, edgecolors="white",
                   linewidth=0.5, label=group["label"], zorder=3)

        # Regression line
        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, p(x_line), color=color, linestyle="--", alpha=0.5, linewidth=1.5)

    ax.legend(fontsize=STYLE["legend_size"], framealpha=0.9)
    ax.grid(True, alpha=STYLE["grid_alpha"])
    _apply_style(ax, title=title, xlabel=xlabel, ylabel=ylabel)

    fig.tight_layout()
    _save(fig, save_path)


# ============================================================
# 4. Grouped Bar Chart
# ============================================================

def plot_grouped_bars(
    categories, groups, values,
    ylabel="Value", title="",
    save_path="figure.pdf",
    figsize=None, colors=None,
    sort_by_first_group=False,
    show_values=True,
):
    """
    Grouped bar chart.

    Args:
        categories: list of category labels (x-axis)
        groups: list of group names (legend)
        values: dict {group_name: [val_per_category]}
        sort_by_first_group: sort categories by first group's values
    """
    n_cats = len(categories)
    n_groups = len(groups)
    if figsize is None:
        figsize = (max(10, n_cats * 1.2), 6)
    if colors is None:
        colors = PALETTE["tools"][:n_groups]

    if sort_by_first_group:
        first_vals = values[groups[0]]
        order = np.argsort(first_vals)[::-1]
        categories = [categories[i] for i in order]
        for g in groups:
            values[g] = [values[g][i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.8 / n_groups
    x = np.arange(n_cats)

    for i, group in enumerate(groups):
        offset = (i - n_groups / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values[group], bar_width,
                      label=group, color=colors[i], edgecolor="white", linewidth=0.5)

        if show_values:
            for bar, val in zip(bars, values[group]):
                if val != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=6, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=STYLE["tick_size"])
    ax.legend(fontsize=STYLE["legend_size"], framealpha=0.9)
    ax.grid(True, axis="y", alpha=STYLE["grid_alpha"])
    _apply_style(ax, title=title, ylabel=ylabel)

    fig.tight_layout()
    _save(fig, save_path)


# ============================================================
# 5. Circuit / Supernode Diagram
# ============================================================

def plot_circuit_diagram(
    nodes, edges,
    title="Attribution Circuit",
    save_path="figure.pdf",
    figsize=(14, 8),
):
    """
    Render an attribution circuit as a node-edge diagram.

    Args:
        nodes: list of dicts with keys:
            "id", "label", "layer", "type" (input/feature/output),
            "activation" (optional), "x" (optional), "y" (optional)
        edges: list of dicts with keys:
            "source", "target", "weight", "label" (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")

    # Color by node type
    type_colors = {
        "input": "#3498db",
        "feature": "#2ecc71",
        "output": "#e74c3c",
        "hub": "#f39c12",
    }

    # Auto-layout: arrange by layer
    if not any("x" in n for n in nodes):
        layers = sorted(set(n.get("layer", 0) for n in nodes))
        layer_map = {l: i for i, l in enumerate(layers)}
        n_layers = len(layers)

        layer_counts = {}
        for n in nodes:
            l = n.get("layer", 0)
            layer_counts[l] = layer_counts.get(l, 0) + 1

        layer_current = {l: 0 for l in layers}
        for n in nodes:
            l = n.get("layer", 0)
            n["x"] = layer_map[l] / max(n_layers - 1, 1)
            count = layer_counts[l]
            idx = layer_current[l]
            n["y"] = (idx + 0.5) / count
            layer_current[l] += 1

    # Build node lookup
    node_lookup = {n["id"]: n for n in nodes}

    # Draw edges first (behind nodes)
    max_weight = max((abs(e["weight"]) for e in edges), default=1)
    for e in edges:
        src = node_lookup.get(e["source"])
        tgt = node_lookup.get(e["target"])
        if not src or not tgt:
            continue

        weight = e["weight"]
        norm_w = abs(weight) / max_weight
        color = "#c0392b" if weight > 0 else "#2980b9"
        alpha = max(0.2, min(0.9, norm_w))
        linewidth = max(0.5, norm_w * 4)

        ax.annotate("",
            xy=(tgt["x"], tgt["y"]),
            xytext=(src["x"], src["y"]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                connectionstyle="arc3,rad=0.1",
            ))

    # Draw nodes
    for n in nodes:
        ntype = n.get("type", "feature")
        color = type_colors.get(ntype, "#95a5a6")
        size = 800 if ntype == "hub" else 400

        ax.scatter(n["x"], n["y"], s=size, c=color, edgecolors="white",
                   linewidth=1.5, zorder=5)
        ax.text(n["x"], n["y"] - 0.04, n["label"],
                ha="center", va="top", fontsize=7, fontweight="bold",
                zorder=6)

    # Layer labels at top
    layers = sorted(set(n.get("layer", 0) for n in nodes))
    layer_map = {l: i for i, l in enumerate(layers)}
    n_layers = len(layers)
    for l in layers:
        x_pos = layer_map[l] / max(n_layers - 1, 1)
        ax.text(x_pos, 1.05, f"L{l}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#7f8c8d")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=type_colors["input"], label="Input"),
        mpatches.Patch(facecolor=type_colors["feature"], label="Feature"),
        mpatches.Patch(facecolor=type_colors["output"], label="Output"),
        mpatches.Patch(facecolor=type_colors["hub"], label="Hub"),
        mpatches.Patch(facecolor="#c0392b", label="Promotes (+)"),
        mpatches.Patch(facecolor="#2980b9", label="Suppresses (-)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=STYLE["legend_size"], framealpha=0.9)

    ax.set_title(title, fontsize=STYLE["title_size"], fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, save_path)


def attribution_graph_to_circuit(graph, top_n_features=15, top_n_edges=30):
    """
    Convert an AttributionGraph to nodes/edges format for plot_circuit_diagram.
    Selects the most important features and edges.
    """
    # Select top features by activation
    sorted_features = sorted(
        graph.feature_nodes.values(),
        key=lambda f: f.activation, reverse=True
    )[:top_n_features]

    feature_ids = {f.node_id for f in sorted_features}

    # Select top edges involving these features
    relevant_edges = [e for e in graph.edges
                      if e.source_id in feature_ids or e.target_id in feature_ids]
    relevant_edges.sort(key=lambda e: abs(e.weight), reverse=True)
    relevant_edges = relevant_edges[:top_n_edges]

    # Identify hub features (many connections)
    edge_counts = {}
    for e in relevant_edges:
        edge_counts[e.source_id] = edge_counts.get(e.source_id, 0) + 1
        edge_counts[e.target_id] = edge_counts.get(e.target_id, 0) + 1

    nodes = []
    for f in sorted_features:
        is_hub = edge_counts.get(f.node_id, 0) >= 4
        nodes.append({
            "id": f.node_id,
            "label": f"L{f.layer}/f{f.feature_idx}\n({f.activation:.0f})",
            "layer": f.layer,
            "type": "hub" if is_hub else "feature",
            "activation": f.activation,
        })

    # Add output nodes
    for nid, o in graph.output_nodes.items():
        if any(e.target_id == nid for e in relevant_edges):
            nodes.append({
                "id": nid,
                "label": f"'{o.token_str}'\n({o.logit_value:.1f})",
                "layer": 26,
                "type": "output",
            })

    edges = [{"source": e.source_id, "target": e.target_id, "weight": e.weight}
             for e in relevant_edges]

    return nodes, edges


# ============================================================
# 6. Text Comparison Panel
# ============================================================

def plot_text_comparison(
    pairs,
    title="Intervention Effect on Generation",
    save_path="figure.pdf",
    figsize=None,
):
    """
    Side-by-side text comparison panels.

    Args:
        pairs: list of dicts with keys:
            "label", "clean", "intervened", "highlight_words" (optional)
    """
    n_pairs = len(pairs)
    if figsize is None:
        figsize = (14, max(4, n_pairs * 3))

    fig, axes = plt.subplots(n_pairs, 2, figsize=figsize)
    if n_pairs == 1:
        axes = axes.reshape(1, 2)

    for i, pair in enumerate(pairs):
        for j, (variant, text) in enumerate([("Clean", pair["clean"]),
                                              ("Ablated", pair["intervened"])]):
            ax = axes[i, j]
            ax.axis("off")

            # Wrap text
            wrapped = _wrap_text(text, 60)

            header_color = "#2c3e50" if j == 0 else "#c0392b"
            ax.text(0.5, 1.0, f"{variant}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, fontweight="bold",
                    color=header_color)

            ax.text(0.05, 0.85, wrapped, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, fontfamily="monospace",
                    wrap=True, linespacing=1.4)

            # Border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(header_color)
                spine.set_linewidth(2)

        # Row label
        axes[i, 0].text(-0.05, 0.5, pair["label"], transform=axes[i, 0].transAxes,
                         ha="right", va="center", fontsize=10, fontweight="bold",
                         rotation=90)

    fig.suptitle(title, fontsize=STYLE["title_size"], fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_path)


def _wrap_text(text, width):
    """Simple text wrapping."""
    words = text.split()
    lines = []
    current = []
    length = 0
    for word in words:
        if length + len(word) + 1 > width:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


# ============================================================
# 7. Line Plot (Formation Timeline)
# ============================================================

def plot_formation_timeline(
    timelines, tokens,
    feature_labels=None,
    title="Feature Activation Across Token Positions",
    save_path="figure.pdf",
    figsize=(14, 5),
):
    """
    Line plot showing feature activation vs token position.

    Args:
        timelines: dict {feature_label: [activation_per_position]}
        tokens: list of token strings for x-axis
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, acts) in enumerate(timelines.items()):
        color = PALETTE["tools"][i % len(PALETTE["tools"])]
        ax.plot(range(len(acts)), acts, "-o", color=color, markersize=3,
                linewidth=1.5, label=label, alpha=0.8)

    # Token labels on x-axis (show every Nth to avoid crowding)
    n_tokens = len(tokens)
    step = max(1, n_tokens // 20)
    ax.set_xticks(range(0, n_tokens, step))
    ax.set_xticklabels([tokens[i][:12] for i in range(0, n_tokens, step)],
                        rotation=45, ha="right", fontsize=7)

    ax.legend(fontsize=STYLE["legend_size"], framealpha=0.9)
    ax.grid(True, alpha=STYLE["grid_alpha"])
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    _apply_style(ax, title=title, xlabel="Token Position", ylabel="Feature Activation")

    fig.tight_layout()
    _save(fig, save_path)


# ============================================================
# 8. Verdict Table
# ============================================================

def plot_verdict_table(
    rows,
    title="Mechanistic Verdict Table",
    save_path="figure.pdf",
    figsize=(14, 5),
):
    """
    Publication-quality formatted table.

    Args:
        rows: list of dicts with keys:
            "behavior", "expected", "verdict", "confidence",
            "evidence_score", "key_finding"
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    col_labels = ["Behavior", "Expected", "Verdict", "Conf.", "Evidence", "Key Finding"]
    cell_text = []
    cell_colors = []

    verdict_colors = {
        "Genuine": "#e8f5e9",
        "Confused": "#fff3e0",
        "Emergent Artifact": "#fce4ec",
    }

    for row in rows:
        verdict = row["verdict"]
        bg = verdict_colors.get(verdict, "#ffffff")
        cell_text.append([
            row["behavior"],
            row["expected"],
            verdict,
            row["confidence"],
            f"{row['evidence_score']}/4",
            row["key_finding"],
        ])
        cell_colors.append([bg] * 6)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)

    # Style cells
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(range(len(col_labels)))
    table.scale(1, 1.8)

    ax.set_title(title, fontsize=STYLE["title_size"], fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, save_path)


# ============================================================
# 9. Multi-panel figure helper
# ============================================================

def plot_multi_panel_bars(
    panels,
    title="",
    save_path="figure.pdf",
    figsize=None,
    n_cols=2,
):
    """
    Multi-panel bar charts (e.g., 4 user properties, 4 behaviors).

    Args:
        panels: list of dicts with keys:
            "title", "categories", "values", "colors" (optional),
            "ylabel" (optional)
    """
    n_panels = len(panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (6 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, panel in enumerate(panels):
        ax = axes[i]
        cats = panel["categories"]
        vals = panel["values"]
        colors = panel.get("colors", [PALETTE["tools"][j % len(PALETTE["tools"])]
                                       for j in range(len(cats))])

        bars = ax.barh(range(len(cats)), vals, color=colors[:len(cats)],
                       edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=7)
        ax.invert_yaxis()

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=7)

        _apply_style(ax, title=panel.get("title", ""),
                     xlabel=panel.get("ylabel", ""))
        ax.grid(True, axis="x", alpha=STYLE["grid_alpha"])

    # Hide unused axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=STYLE["title_size"] + 1, fontweight="bold", y=1.02)

    fig.tight_layout()
    _save(fig, save_path)
