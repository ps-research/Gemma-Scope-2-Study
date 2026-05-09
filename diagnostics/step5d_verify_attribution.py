"""
Step 5d: Verify the corrected attribution graph.
Key checks: multi-hop circuits, Paris in outputs, FF edges in pruned graph.

Run: CUDA_VISIBLE_DEVICES=1 python diagnostics/step5d_verify_attribution.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt
from src.attribution import (
    build_attribution_graph, prune_graph, compute_graph_metrics, save_graph,
)

print("=" * 70)
print("STEP 5d: VERIFY CORRECTED ATTRIBUTION GRAPH")
print("=" * 70)

model, tokenizer = load_gemma3_1b("pt", device="cuda")
clt = load_clt(width="262k", l0="big", affine=True, device="cuda", half_precision=True)

prompt = "The capital of France is"

t0 = time.time()
graph = build_attribution_graph(
    model=model, clt=clt, tokenizer=tokenizer,
    prompt=prompt,
    top_k_output_tokens=10,
    min_ff_edge_weight=10.0,
    min_fl_edge_weight=1.0,
)
print(f"\nFull graph ({time.time()-t0:.1f}s):")
print(graph.summary())

# Count edge types
ff = sum(1 for e in graph.edges if e.source_id in graph.feature_nodes and e.target_id in graph.feature_nodes)
fl = sum(1 for e in graph.edges if e.source_id in graph.feature_nodes and e.target_id in graph.output_nodes)
print(f"  FF edges: {ff}, FL edges: {fl}")

# ============================================================
# PRUNE
# ============================================================
print(f"\n{'='*70}")
print("PRUNING")
print(f"{'='*70}")

pruned = prune_graph(
    graph,
    top_k_edges_per_node=5,
    max_feature_nodes=50,
    min_edge_weight=5.0,
)
print(f"\nPruned graph:")
print(pruned.summary())

metrics = compute_graph_metrics(pruned)
for k, v in metrics.items():
    if k == "layer_distribution":
        print(f"  {k}: {v}")
    elif isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# ============================================================
# OUTPUT NODES
# ============================================================
print(f"\n{'='*70}")
print("OUTPUT TOKENS IN PRUNED GRAPH")
print(f"{'='*70}")

for nid, node in sorted(pruned.output_nodes.items(), key=lambda x: x[1].logit_value, reverse=True):
    incoming = pruned.get_incoming_edges(nid)
    total_w = sum(e.weight for e in incoming)
    print(f"  '{node.token_str:15s}' logit={node.logit_value:7.2f}  "
          f"{len(incoming):3d} contributing features  total_attr_weight={total_w:+.1f}")

# ============================================================
# FEATURE→FEATURE EDGES
# ============================================================
print(f"\n{'='*70}")
print("FEATURE→FEATURE EDGES (multi-hop circuits)")
print(f"{'='*70}")

ff_in_pruned = [e for e in pruned.edges
                if e.source_id in pruned.feature_nodes and e.target_id in pruned.feature_nodes]
print(f"Total FF edges in pruned graph: {len(ff_in_pruned)}")
for i, e in enumerate(sorted(ff_in_pruned, key=lambda x: abs(x.weight), reverse=True)[:15]):
    src = pruned.feature_nodes[e.source_id]
    tgt = pruned.feature_nodes[e.target_id]
    print(f"  {i+1:2d}. L{src.layer:02d}/f{src.feature_idx:5d} (act={src.activation:7.0f})"
          f" → L{tgt.layer:02d}/f{tgt.feature_idx:5d} (act={tgt.activation:7.0f})"
          f"  w={e.weight:+.0f}  vw={e.virtual_weight:+.4f}")

# ============================================================
# TOP FEATURES → PARIS
# ============================================================
print(f"\n{'='*70}")
print("FEATURES → 'Paris' (tracing the prediction)")
print(f"{'='*70}")

paris_id = tokenizer.encode("Paris", add_special_tokens=False)[0]
paris_node_id = f"output_p{len(tokenizer.encode(prompt, add_special_tokens=True))-1}_t{paris_id}"

paris_edges = pruned.get_incoming_edges(paris_node_id)
if paris_edges:
    print(f"Direct contributors to 'Paris' ({len(paris_edges)} edges):")
    for e in sorted(paris_edges, key=lambda x: abs(x.weight), reverse=True)[:10]:
        src = pruned.feature_nodes.get(e.source_id)
        if src:
            # Check if this feature has incoming FF edges (= multi-hop)
            ff_into_src = [ee for ee in pruned.edges
                          if ee.target_id == src.node_id and ee.source_id in pruned.feature_nodes]
            hop_info = f"  ({len(ff_into_src)} upstream features)" if ff_into_src else ""
            print(f"  L{src.layer:02d}/f{src.feature_idx:5d} act={src.activation:7.0f}"
                  f"  → Paris  w={e.weight:+.0f}{hop_info}")
else:
    print("  Paris not in pruned output nodes (may need higher top_k_output_tokens)")

# ============================================================
# MULTI-HOP PATH EXAMPLE
# ============================================================
print(f"\n{'='*70}")
print("MULTI-HOP PATHS (example circuits)")
print(f"{'='*70}")

# Find features that appear as both source in FF and source in FL
ff_sources = {e.source_id for e in ff_in_pruned}
fl_sources = {e.source_id for e in pruned.edges
              if e.source_id in pruned.feature_nodes and e.target_id in pruned.output_nodes}
ff_targets = {e.target_id for e in ff_in_pruned}

# Features that are FF targets AND FL sources = middle of a 2-hop path
middle_features = ff_targets & fl_sources
print(f"Features in middle of 2-hop paths: {len(middle_features)}")

for mid_id in list(middle_features)[:5]:
    mid = pruned.feature_nodes[mid_id]
    # Upstream features
    upstream = [(pruned.feature_nodes[e.source_id], e) for e in ff_in_pruned if e.target_id == mid_id]
    # Downstream outputs
    downstream = [(pruned.output_nodes[e.target_id], e) for e in pruned.edges
                  if e.source_id == mid_id and e.target_id in pruned.output_nodes]

    if upstream and downstream:
        up_f, up_e = max(upstream, key=lambda x: abs(x[1].weight))
        dn_o, dn_e = max(downstream, key=lambda x: abs(x[1].weight))
        print(f"\n  L{up_f.layer:02d}/f{up_f.feature_idx} (act={up_f.activation:.0f})"
              f" →[w={up_e.weight:+.0f}]→"
              f" L{mid.layer:02d}/f{mid.feature_idx} (act={mid.activation:.0f})"
              f" →[w={dn_e.weight:+.0f}]→"
              f" '{dn_o.token_str}'")

save_graph(pruned, "/workspace/Gemma-Scope-2-Study/outputs/attribution_graph_france_v4.json")

print(f"\n{'='*70}")
print(f"GPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
print("STEP 5d COMPLETE")
print(f"{'='*70}")
