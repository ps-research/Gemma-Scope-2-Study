"""
Step 5c: Diagnose and fix attribution graph issues.
1. Check feature→feature edge weight distribution
2. Add model's actual top predictions as output targets
3. Verify multi-hop circuits exist

Run: CUDA_VISIBLE_DEVICES=1 python diagnostics/step5c_verify_attribution.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt
from src.hooks import gather_clt_activations
from src.attribution import (
    extract_active_features,
    compute_feature_to_feature_edges,
    compute_feature_to_logit_edges,
    build_attribution_graph,
    prune_graph,
    compute_graph_metrics,
    save_graph,
    AttributionGraph,
    OutputNode,
    Edge,
)

print("=" * 70)
print("STEP 5c: DIAGNOSE ATTRIBUTION GRAPH")
print("=" * 70)

model, tokenizer = load_gemma3_1b("pt", device="cuda")
clt = load_clt(width="262k", l0="big", affine=True, device="cuda", half_precision=True)

prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")

# ============================================================
# DIAGNOSIS 1: Feature→Feature edge weight distribution
# ============================================================
print(f"\n{'='*70}")
print("DIAGNOSIS 1: Feature→Feature edge weight distribution")
print(f"{'='*70}")

clt_inputs, clt_targets = gather_clt_activations(model, 26, inputs)
clt_inputs_h = clt_inputs.half()

active = extract_active_features(clt, clt_inputs_h, min_activation=0.0, skip_bos=True)
print(f"Active features: {len(active)}")

# Get ALL feature→feature edges with very low threshold
t0 = time.time()
ff_edges = compute_feature_to_feature_edges(clt, active, min_edge_weight=0.0, same_position_only=True)
print(f"Feature→Feature edges (no threshold): {len(ff_edges)} in {time.time()-t0:.1f}s")

weights = [abs(e.weight) for e in ff_edges]
print(f"\n  Weight distribution:")
print(f"    min:    {np.min(weights):.4f}")
print(f"    25th:   {np.percentile(weights, 25):.4f}")
print(f"    50th:   {np.percentile(weights, 50):.4f}")
print(f"    75th:   {np.percentile(weights, 75):.4f}")
print(f"    90th:   {np.percentile(weights, 90):.4f}")
print(f"    95th:   {np.percentile(weights, 95):.4f}")
print(f"    99th:   {np.percentile(weights, 99):.4f}")
print(f"    max:    {np.max(weights):.4f}")

# Count how many above various thresholds
for thresh in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
    count = sum(1 for w in weights if w >= thresh)
    print(f"    >= {thresh:>6.1f}: {count:>6d} edges ({100*count/len(weights):.1f}%)")

# Show top 10 feature→feature edges
print(f"\n  Top 10 Feature→Feature edges:")
ff_edges.sort(key=lambda e: abs(e.weight), reverse=True)
for i, e in enumerate(ff_edges[:10]):
    src = next(f for f in active if f.node_id == e.source_id)
    tgt = next(f for f in active if f.node_id == e.target_id)
    print(f"    {i+1}. L{src.layer:02d}/f{src.feature_idx} (act={src.activation:.0f})"
          f" → L{tgt.layer:02d}/f{tgt.feature_idx} (act={tgt.activation:.0f})"
          f"  w={e.weight:.1f} vw={e.virtual_weight:.4f}")

# ============================================================
# DIAGNOSIS 2: Why is "Paris" missing?
# ============================================================
print(f"\n{'='*70}")
print("DIAGNOSIS 2: Aggregate logit effects on 'Paris'")
print(f"{'='*70}")

# Get the Paris token ID
paris_tokens = tokenizer.encode("Paris", add_special_tokens=False)
print(f"'Paris' token IDs: {paris_tokens}")
paris_id = paris_tokens[0]

# For each active feature at position 5 (last token), compute its logit effect on "Paris"
w_unembed = model.lm_head.weight  # (vocab, d_model)
ln_weight = model.model.norm.weight
w_eff = (w_unembed * ln_weight).float()  # (vocab, d_model)
paris_unembed = w_eff[paris_id]  # (d_model,)

w_dec = clt.w_dec.data

last_pos_features = [f for f in active if f.position == inputs.shape[1] - 1]
print(f"Active features at last position: {len(last_pos_features)}")

paris_contributions = []
for f in last_pos_features:
    dec_sum = w_dec[f.layer, f.feature_idx, f.layer:].sum(dim=0).float()
    logit_effect = (dec_sum * paris_unembed).sum().item()
    attributed_effect = f.activation * logit_effect
    paris_contributions.append((f, logit_effect, attributed_effect))

paris_contributions.sort(key=lambda x: abs(x[2]), reverse=True)

print(f"\nTop 15 features contributing to 'Paris' prediction:")
total_attributed = sum(x[2] for x in paris_contributions)
print(f"  Total attributed logit effect on 'Paris': {total_attributed:.2f}")
print()

for i, (f, logit_eff, attr_eff) in enumerate(paris_contributions[:15]):
    print(f"  {i+1:2d}. L{f.layer:02d}/f{f.feature_idx:5d} act={f.activation:7.1f}"
          f"  logit_eff={logit_eff:+.4f}  attributed={attr_eff:+.1f}")

# Also show for "a" (the model's top prediction) for comparison
a_id = tokenizer.encode("a", add_special_tokens=False)[0]
a_unembed = w_eff[a_id]

a_contributions = []
for f in last_pos_features:
    dec_sum = w_dec[f.layer, f.feature_idx, f.layer:].sum(dim=0).float()
    logit_effect = (dec_sum * a_unembed).sum().item()
    attributed_effect = f.activation * logit_effect
    a_contributions.append((f, logit_effect, attributed_effect))

a_contributions.sort(key=lambda x: abs(x[2]), reverse=True)
total_a = sum(x[2] for x in a_contributions)
print(f"\nFor comparison, total attributed to 'a': {total_a:.2f}")
print(f"Top 5 features for 'a':")
for i, (f, logit_eff, attr_eff) in enumerate(a_contributions[:5]):
    print(f"  {i+1}. L{f.layer:02d}/f{f.feature_idx:5d} act={f.activation:7.1f}"
          f"  logit_eff={logit_eff:+.4f}  attributed={attr_eff:+.1f}")

# ============================================================
# FIX: Build graph with correct thresholds
# ============================================================
print(f"\n{'='*70}")
print("FIXED GRAPH: lower thresholds + model's actual top predictions")
print(f"{'='*70}")

# Build graph with lower edge threshold to capture FF edges
graph = build_attribution_graph(
    model=model, clt=clt, tokenizer=tokenizer,
    prompt=prompt,
    target_positions=[inputs.shape[1] - 1],
    min_feature_activation=0.0,
    min_edge_weight=0.01,  # much lower to capture FF edges
    top_k_logit_tokens=10,
    same_position_only=True,
)

# Manually add output nodes for model's actual top predictions
logits = model(inputs).logits[0, -1]
top_vals, top_ids = logits.topk(10)
last_pos = inputs.shape[1] - 1

for v, tid in zip(top_vals, top_ids):
    tok_str = tokenizer.decode([tid.item()]).strip()
    out_node = OutputNode(
        position=last_pos,
        token_id=tid.item(),
        token_str=tok_str,
        logit_value=v.item(),
    )
    if out_node.node_id not in graph.output_nodes:
        graph.output_nodes[out_node.node_id] = out_node

# Add edges from features to these model-top predictions
for f in last_pos_features:
    dec_sum = w_dec[f.layer, f.feature_idx, f.layer:].sum(dim=0).float()
    for tid in top_ids:
        tok_unembed = w_eff[tid.item()]
        logit_eff = (dec_sum * tok_unembed).sum().item()
        attr_eff = f.activation * logit_eff
        if abs(attr_eff) >= 0.5:
            tok_str = tokenizer.decode([tid.item()]).strip()
            out_node_id = f"output_p{last_pos}_t{tid.item()}"
            graph.edges.append(Edge(
                source_id=f.node_id,
                target_id=out_node_id,
                weight=attr_eff,
                virtual_weight=logit_eff,
            ))

print(f"Full graph: {graph.summary()}")

# Prune with calibrated thresholds
pruned = prune_graph(
    graph,
    top_k_edges_per_node=5,
    max_feature_nodes=40,
    min_edge_weight=0.5,  # lower to allow FF edges through
)

print(f"\nPruned: {pruned.summary()}")
metrics = compute_graph_metrics(pruned)
print(f"Path length: avg={metrics.get('avg_path_length', 'N/A')}, max={metrics.get('max_path_length', 'N/A')}")

# Show feature→feature edges
ff_in_pruned = [e for e in pruned.edges
                if e.source_id in pruned.feature_nodes and e.target_id in pruned.feature_nodes]
print(f"\nFeature→Feature edges in pruned graph: {len(ff_in_pruned)}")
for i, e in enumerate(sorted(ff_in_pruned, key=lambda x: abs(x.weight), reverse=True)[:10]):
    src = pruned.feature_nodes[e.source_id]
    tgt = pruned.feature_nodes[e.target_id]
    print(f"  {i+1}. L{src.layer:02d}/f{src.feature_idx} (act={src.activation:.0f})"
          f" → L{tgt.layer:02d}/f{tgt.feature_idx} (act={tgt.activation:.0f})"
          f"  w={e.weight:.1f}")

# Show Paris-related edges
print(f"\nEdges pointing to 'Paris':")
paris_node_id = f"output_p{last_pos}_t{paris_id}"
paris_edges = [e for e in pruned.edges if e.target_id == paris_node_id]
for e in sorted(paris_edges, key=lambda x: abs(x.weight), reverse=True)[:10]:
    src = pruned.all_nodes.get(e.source_id)
    if hasattr(src, 'layer'):
        print(f"  L{src.layer:02d}/f{src.feature_idx} (act={src.activation:.0f}) → Paris  w={e.weight:.1f}")

save_graph(pruned, "/workspace/Gemma-Scope-2-Study/outputs/attribution_graph_france_v3.json")

print(f"\n{'='*70}")
print(f"GPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
print("STEP 5c COMPLETE")
print(f"{'='*70}")
