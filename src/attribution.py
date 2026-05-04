"""
Attribution graph computation for CLTs.

ARCHITECTURE:
  1. Run model to get actual top-k predictions at target positions
  2. Run CLT encoder to find all active features
  3. Compute every active feature's logit effect on each target token
     -> These are feature->output edges
  4. Compute feature->feature virtual weights (decoder-encoder dot products)
     -> These are feature->feature edges
  5. Prune backward from output nodes through features to find circuits

THEORY (same-position, Stage 1):
  Virtual weight from feature s (layer_s) to feature t (layer_t > layer_s):
    w(s->t) = sum over l from layer_s to layer_t-1 of:
              w_dec[layer_s, idx_s, l, :] dot w_enc[layer_t, :, idx_t]

  Feature s's effect on output logit for token v:
    logit_effect(s->v) = a_s * (sum over l>=layer_s of w_dec[s.layer, s.idx, l, :]) dot W_eff_unembed[v, :]

VERIFIED FACTS:
  - Decoder causality is architectural: w_dec[l, :, l2, :] = 0 for l > l2
  - CLT 262k has 10,080 features per layer x 26 layers
  - "Paris" is predicted through collective action of 15+ features, not any single one
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json


# ============================================================
# Data structures
# ============================================================

@dataclass
class FeatureNode:
    """A single active feature in the attribution graph."""
    position: int
    layer: int
    feature_idx: int
    activation: float

    @property
    def node_id(self) -> str:
        return f"feat_p{self.position}_L{self.layer}_f{self.feature_idx}"


@dataclass
class InputNode:
    """Token embedding input node."""
    position: int
    token_id: int
    token_str: str

    @property
    def node_id(self) -> str:
        return f"input_p{self.position}"


@dataclass
class OutputNode:
    """Output logit node for a specific token prediction."""
    position: int
    token_id: int
    token_str: str
    logit_value: float

    @property
    def node_id(self) -> str:
        return f"output_p{self.position}_t{self.token_id}"


@dataclass
class Edge:
    """Directed edge in the attribution graph."""
    source_id: str
    target_id: str
    weight: float
    virtual_weight: float


@dataclass
class AttributionGraph:
    """Complete attribution graph for a prompt."""
    prompt: str
    tokens: list
    feature_nodes: dict = field(default_factory=dict)
    input_nodes: dict = field(default_factory=dict)
    output_nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)

    @property
    def all_nodes(self) -> dict:
        return {**self.feature_nodes, **self.input_nodes, **self.output_nodes}

    @property
    def num_nodes(self) -> int:
        return len(self.feature_nodes) + len(self.input_nodes) + len(self.output_nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_incoming_edges(self, node_id):
        return [e for e in self.edges if e.target_id == node_id]

    def get_outgoing_edges(self, node_id):
        return [e for e in self.edges if e.source_id == node_id]

    def summary(self) -> str:
        lines = [
            f"AttributionGraph for: '{self.prompt}'",
            f"  Tokens: {len(self.tokens)}",
            f"  Feature nodes: {len(self.feature_nodes)}",
            f"  Input nodes: {len(self.input_nodes)}",
            f"  Output nodes: {len(self.output_nodes)}",
            f"  Edges: {len(self.edges)}",
        ]
        if self.edges:
            weights = [abs(e.weight) for e in self.edges]
            lines.append(f"  Edge |weight| range: [{min(weights):.2f}, {max(weights):.2f}]")
        return "\n".join(lines)


# ============================================================
# Feature extraction
# ============================================================

def extract_active_features(clt, clt_inputs, min_activation=0.0, skip_bos=True):
    """Run CLT encoder and extract all active features."""
    features = clt.encode(clt_inputs)
    active = []
    start_pos = 1 if skip_bos else 0

    for pos in range(start_pos, features.shape[0]):
        for layer in range(features.shape[1]):
            feat_acts = features[pos, layer]
            nonzero_mask = feat_acts > min_activation
            nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]
            for idx in nonzero_indices:
                active.append(FeatureNode(
                    position=pos,
                    layer=layer,
                    feature_idx=idx.item(),
                    activation=feat_acts[idx].item(),
                ))
    return active


# ============================================================
# Virtual weight computation
# ============================================================

def compute_feature_to_feature_edges(clt, active_features, min_edge_weight=0.01, same_position_only=True):
    """
    Compute virtual weight edges between active features.

    Virtual weight from s to t (same position, t.layer > s.layer):
      w(s->t) = sum_{l=s.layer}^{t.layer-1} w_dec[s.layer, s.idx, l, :] dot w_enc[t.layer, :, t.idx]
    Edge weight: A(s->t) = s.activation * w(s->t)
    """
    edges = []
    w_dec = clt.w_dec.data
    w_enc = clt.w_enc.data

    features_by_pos = {}
    for f in active_features:
        features_by_pos.setdefault(f.position, []).append(f)

    for pos, pos_features in features_by_pos.items():
        for s in pos_features:
            s_dec_vectors = w_dec[s.layer, s.feature_idx]  # (n_layers, d_model)

            for t in pos_features:
                if t.layer <= s.layer:
                    continue

                t_enc_vector = w_enc[t.layer, :, t.feature_idx]
                dec_sum = s_dec_vectors[s.layer:t.layer].sum(dim=0)
                virtual_w = (dec_sum * t_enc_vector).sum().item()
                edge_w = s.activation * virtual_w

                if abs(edge_w) >= min_edge_weight:
                    edges.append(Edge(
                        source_id=s.node_id,
                        target_id=t.node_id,
                        weight=edge_w,
                        virtual_weight=virtual_w,
                    ))

    return edges


def compute_feature_to_logit_edges(clt, model, active_features, target_token_ids, target_positions, tokenizer, min_edge_weight=0.01):
    """
    Compute edges from ALL active features to SPECIFIC target tokens.

    Unlike per-feature top-k, this computes every active feature's effect on
    each target token. This correctly captures collective effects (e.g., "Paris"
    promoted by many features each contributing a little).
    """
    w_unembed = model.lm_head.weight
    ln_weight = model.model.norm.weight
    w_eff = (w_unembed * ln_weight).float()
    w_dec = clt.w_dec.data

    # Precompute unembedding vectors for target tokens
    target_unembed = {}
    for tid in target_token_ids:
        target_unembed[tid] = w_eff[tid]

    # Create output nodes
    output_nodes = {}
    for tid in target_token_ids:
        tok_str = tokenizer.decode([tid]).strip()
        for pos in target_positions:
            node = OutputNode(position=pos, token_id=tid, token_str=tok_str, logit_value=0.0)
            output_nodes[node.node_id] = node

    # Compute edges: every feature at target positions -> every target token
    edges = []
    for f in active_features:
        if f.position not in target_positions:
            continue

        # Sum decoder writes from f.layer to end (all reach final residual)
        dec_sum = w_dec[f.layer, f.feature_idx, f.layer:].sum(dim=0).float()

        for tid in target_token_ids:
            logit_eff = (dec_sum * target_unembed[tid]).sum().item()
            edge_w = f.activation * logit_eff

            if abs(edge_w) >= min_edge_weight:
                out_node_id = f"output_p{f.position}_t{tid}"
                edges.append(Edge(
                    source_id=f.node_id,
                    target_id=out_node_id,
                    weight=edge_w,
                    virtual_weight=logit_eff,
                ))

    return edges, list(output_nodes.values())


# ============================================================
# Full graph construction
# ============================================================

def build_attribution_graph(
    model,
    clt,
    tokenizer,
    prompt,
    target_positions=None,
    top_k_output_tokens=10,
    min_feature_activation=0.0,
    min_ff_edge_weight=10.0,
    min_fl_edge_weight=1.0,
    same_position_only=True,
    device="cuda",
):
    """
    Build a complete attribution graph for a prompt.

    Pipeline:
      1. Run model to get actual top-k predictions
      2. Run CLT encoder to get active features
      3. Compute feature->feature edges (virtual weights)
      4. Compute feature->logit edges for model's top predictions
      5. Assemble graph
    """
    from src.hooks import gather_clt_activations

    # 1. Tokenize and run model
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    tokens = inputs[0].tolist()
    str_tokens = tokenizer.convert_ids_to_tokens(tokens)

    if target_positions is None:
        target_positions = [len(tokens) - 1]

    # Get model's actual top predictions at each target position
    with torch.no_grad():
        logits = model(inputs).logits[0]

    target_token_ids = set()
    output_logit_values = {}
    for pos in target_positions:
        top_vals, top_ids = logits[pos].topk(top_k_output_tokens)
        for v, tid in zip(top_vals, top_ids):
            target_token_ids.add(tid.item())
            output_logit_values[(pos, tid.item())] = v.item()

    target_token_ids = list(target_token_ids)
    print(f"  Target tokens: {[tokenizer.decode([t]).strip() for t in target_token_ids[:10]]}")

    # 2. Get CLT inputs and extract features
    clt_inputs, clt_targets = gather_clt_activations(model, clt.num_layers, inputs)
    if next(clt.parameters()).dtype == torch.float16:
        clt_inputs = clt_inputs.half()

    active_features = extract_active_features(
        clt, clt_inputs, min_activation=min_feature_activation, skip_bos=True,
    )
    print(f"  Active features: {len(active_features)}")

    # 3. Feature->feature edges
    ff_edges = compute_feature_to_feature_edges(
        clt, active_features,
        min_edge_weight=min_ff_edge_weight,
        same_position_only=same_position_only,
    )
    print(f"  Feature->Feature edges: {len(ff_edges)}")

    # 4. Feature->logit edges
    fl_edges, output_nodes = compute_feature_to_logit_edges(
        clt, model, active_features,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        tokenizer=tokenizer,
        min_edge_weight=min_fl_edge_weight,
    )
    print(f"  Feature->Logit edges: {len(fl_edges)}")

    # Set logit values on output nodes
    for node in output_nodes:
        key = (node.position, node.token_id)
        if key in output_logit_values:
            node.logit_value = output_logit_values[key]

    # 5. Assemble
    graph = AttributionGraph(prompt=prompt, tokens=str_tokens)

    for i, (tok_id, tok_str) in enumerate(zip(tokens, str_tokens)):
        node = InputNode(position=i, token_id=tok_id, token_str=tok_str)
        graph.input_nodes[node.node_id] = node

    for f in active_features:
        graph.feature_nodes[f.node_id] = f

    for o in output_nodes:
        graph.output_nodes[o.node_id] = o

    graph.edges = ff_edges + fl_edges

    return graph


# ============================================================
# Graph pruning
# ============================================================

def prune_graph(
    graph,
    target_node_ids=None,
    top_k_edges_per_node=5,
    max_feature_nodes=50,
    min_edge_weight=1.0,
):
    """
    Prune graph by backward BFS from target output nodes.

    1. Start from target output nodes
    2. Find top-k features contributing to each output (hop 1)
    3. Find top-k features contributing to each discovered feature (hop 2+)
    4. Continue until budget or no more edges
    """
    if target_node_ids is None:
        target_node_ids = list(graph.output_nodes.keys())

    # Build incoming edge index
    incoming = {}
    for e in graph.edges:
        if abs(e.weight) < min_edge_weight:
            continue
        incoming.setdefault(e.target_id, []).append(e)

    # Start from output nodes that have incoming edges
    seed_outputs = [nid for nid in target_node_ids if nid in incoming]

    discovered_features = set()
    kept_output_ids = set()
    kept_edges = []
    frontier = seed_outputs

    while frontier and len(discovered_features) < max_feature_nodes:
        next_frontier = []
        for node_id in frontier:
            edges = incoming.get(node_id, [])
            edges.sort(key=lambda e: abs(e.weight), reverse=True)
            for e in edges[:top_k_edges_per_node]:
                kept_edges.append(e)
                src_id = e.source_id

                # Track which outputs actually get edges
                if node_id in graph.output_nodes:
                    kept_output_ids.add(node_id)

                if src_id in graph.feature_nodes and src_id not in discovered_features:
                    discovered_features.add(src_id)
                    next_frontier.append(src_id)
                    if len(discovered_features) >= max_feature_nodes:
                        break
            if len(discovered_features) >= max_feature_nodes:
                break
        frontier = next_frontier

    # Build pruned graph with only nodes that have edges
    pruned = AttributionGraph(prompt=graph.prompt, tokens=graph.tokens)

    for nid in discovered_features:
        pruned.feature_nodes[nid] = graph.feature_nodes[nid]

    for nid in kept_output_ids:
        pruned.output_nodes[nid] = graph.output_nodes[nid]

    # Only keep edges where both endpoints exist in pruned graph
    all_ids = set(pruned.feature_nodes) | set(pruned.output_nodes) | set(pruned.input_nodes)
    pruned.edges = [e for e in kept_edges if e.source_id in all_ids and e.target_id in all_ids]

    return pruned


# ============================================================
# Graph metrics
# ============================================================

def compute_graph_metrics(graph):
    metrics = {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "num_feature_nodes": len(graph.feature_nodes),
        "num_input_nodes": len(graph.input_nodes),
        "num_output_nodes": len(graph.output_nodes),
    }

    if graph.num_edges > 0:
        weights = [abs(e.weight) for e in graph.edges]
        metrics["edge_weight_mean"] = np.mean(weights)
        metrics["edge_weight_std"] = np.std(weights)
        metrics["edge_weight_min"] = np.min(weights)
        metrics["edge_weight_max"] = np.max(weights)

    # Count edge types
    ff_count = sum(1 for e in graph.edges
                   if e.source_id in graph.feature_nodes and e.target_id in graph.feature_nodes)
    fl_count = sum(1 for e in graph.edges
                   if e.source_id in graph.feature_nodes and e.target_id in graph.output_nodes)
    metrics["feature_to_feature_edges"] = ff_count
    metrics["feature_to_logit_edges"] = fl_count

    # Layer distribution
    layer_counts = {}
    for f in graph.feature_nodes.values():
        layer_counts[f.layer] = layer_counts.get(f.layer, 0) + 1
    metrics["layer_distribution"] = dict(sorted(layer_counts.items()))

    # Path length via BFS
    if graph.output_nodes and graph.feature_nodes:
        reverse_adj = {}
        for e in graph.edges:
            reverse_adj.setdefault(e.target_id, []).append(e.source_id)

        path_lengths = []
        for out_id in graph.output_nodes:
            visited = {out_id: 0}
            queue = [out_id]
            while queue:
                current = queue.pop(0)
                for src in reverse_adj.get(current, []):
                    if src not in visited:
                        visited[src] = visited[current] + 1
                        queue.append(src)
            for nid, dist in visited.items():
                if nid in graph.feature_nodes and dist > 0:
                    path_lengths.append(dist)

        if path_lengths:
            metrics["avg_path_length"] = np.mean(path_lengths)
            metrics["max_path_length"] = max(path_lengths)

    return metrics


# ============================================================
# Export
# ============================================================

def graph_to_dict(graph):
    return {
        "prompt": graph.prompt,
        "tokens": graph.tokens,
        "feature_nodes": {
            nid: {"position": n.position, "layer": n.layer, "feature_idx": n.feature_idx, "activation": n.activation}
            for nid, n in graph.feature_nodes.items()
        },
        "input_nodes": {
            nid: {"position": n.position, "token_id": n.token_id, "token_str": n.token_str}
            for nid, n in graph.input_nodes.items()
        },
        "output_nodes": {
            nid: {"position": n.position, "token_id": n.token_id, "token_str": n.token_str, "logit_value": n.logit_value}
            for nid, n in graph.output_nodes.items()
        },
        "edges": [
            {"source": e.source_id, "target": e.target_id, "weight": e.weight, "virtual_weight": e.virtual_weight}
            for e in graph.edges
        ],
    }

def save_graph(graph, path):
    with open(path, "w") as f:
        json.dump(graph_to_dict(graph), f, indent=2)
    print(f"Saved graph to {path}")
