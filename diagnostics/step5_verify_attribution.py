"""
Step 5: Verify attribution graph computation.
Builds a full attribution graph on a simple prompt, prunes it,
and computes metrics.

Run: CUDA_VISIBLE_DEVICES=1 python diagnostics/step5_verify_attribution.py
"""
import sys, os, gc, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

from src.loader import load_gemma3_1b, load_clt
from src.attribution import (
    build_attribution_graph,
    prune_graph,
    compute_graph_metrics,
    save_graph,
)

print("=" * 70)
print("STEP 5: VERIFY ATTRIBUTION GRAPH COMPUTATION")
print("=" * 70)

# 1. Load model + CLT
model, tokenizer = load_gemma3_1b("pt", device="cuda")

clt = load_clt(
    width="262k", l0="big", affine=True,
    device="cuda", half_precision=True,
)

# ============================================================
# 2. Build attribution graph on a simple factual prompt
# ============================================================
prompt = "The capital of France is"
print(f"\n{'='*70}")
print(f"PROMPT: '{prompt}'")
print(f"{'='*70}")

t0 = time.time()
graph = build_attribution_graph(
    model=model,
    clt=clt,
    tokenizer=tokenizer,
    prompt=prompt,
    target_positions=None,  # last token
    min_feature_activation=0.0,
    min_edge_weight=0.1,
    top_k_logit_tokens=5,
    same_position_only=True,
)
t_build = time.time() - t0

print(f"\nFull graph built in {t_build:.1f}s:")
print(graph.summary())

# ============================================================
# 3. Compute metrics on full graph
# ============================================================
metrics = compute_graph_metrics(graph)
print(f"\nFull graph metrics:")
for k, v in metrics.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"    Layer {kk}: {vv} features")
    elif isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# ============================================================
# 4. Prune graph
# ============================================================
print(f"\n{'='*70}")
print("PRUNING")
print(f"{'='*70}")

pruned = prune_graph(
    graph,
    top_k_edges_per_node=3,
    max_total_nodes=30,
    min_edge_weight=1.0,
)
print(f"\nPruned graph:")
print(pruned.summary())

pruned_metrics = compute_graph_metrics(pruned)
print(f"\nPruned graph metrics:")
for k, v in pruned_metrics.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for kk, vv in v.items():
            print(f"    Layer {kk}: {vv} features")
    elif isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# ============================================================
# 5. Inspect top edges
# ============================================================
print(f"\n{'='*70}")
print("TOP 15 EDGES IN PRUNED GRAPH (by |weight|)")
print(f"{'='*70}")

sorted_edges = sorted(pruned.edges, key=lambda e: abs(e.weight), reverse=True)
for i, e in enumerate(sorted_edges[:15]):
    src = pruned.all_nodes.get(e.source_id)
    tgt = pruned.all_nodes.get(e.target_id)

    if hasattr(src, 'layer'):
        src_str = f"L{src.layer:02d}/f{src.feature_idx} @pos{src.position} (act={src.activation:.1f})"
    elif hasattr(src, 'token_str'):
        src_str = f"input '{src.token_str}' @pos{src.position}"
    else:
        src_str = e.source_id

    if hasattr(tgt, 'layer'):
        tgt_str = f"L{tgt.layer:02d}/f{tgt.feature_idx} @pos{tgt.position}"
    elif hasattr(tgt, 'token_str') and hasattr(tgt, 'logit_value'):
        tgt_str = f"output '{tgt.token_str}' @pos{tgt.position}"
    else:
        tgt_str = e.target_id

    print(f"  {i+1:2d}. {src_str}")
    print(f"      → {tgt_str}")
    print(f"      weight={e.weight:.4f}  virtual_weight={e.virtual_weight:.6f}")
    print()

# ============================================================
# 6. Inspect output nodes: what tokens are predicted?
# ============================================================
print(f"{'='*70}")
print("OUTPUT NODES (predicted tokens)")
print(f"{'='*70}")

# Get all output nodes sorted by total incoming edge weight
for node_id, node in sorted(
    pruned.output_nodes.items(),
    key=lambda x: abs(x[1].logit_value),
    reverse=True,
):
    incoming = pruned.get_incoming_edges(node_id)
    total_weight = sum(e.weight for e in incoming)
    print(f"  '{node.token_str}' (logit={node.logit_value:.3f}): "
          f"{len(incoming)} incoming edges, total weight={total_weight:.3f}")

# ============================================================
# 7. Save graph
# ============================================================
os.makedirs("/mnt/storage/sandeep/priyansh/Gemma-Scope-2-Study/outputs", exist_ok=True)
save_graph(pruned, "/mnt/storage/sandeep/priyansh/Gemma-Scope-2-Study/outputs/attribution_graph_france.json")

# ============================================================
# 8. Second prompt: test on a different domain
# ============================================================
print(f"\n{'='*70}")
prompt2 = "The CEO of Apple is"
print(f"PROMPT 2: '{prompt2}'")
print(f"{'='*70}")

t0 = time.time()
graph2 = build_attribution_graph(
    model=model, clt=clt, tokenizer=tokenizer,
    prompt=prompt2,
    min_edge_weight=0.1,
    same_position_only=True,
)
t_build = time.time() - t0
print(f"\nBuilt in {t_build:.1f}s:")
print(graph2.summary())

pruned2 = prune_graph(graph2, top_k_edges_per_node=3, max_total_nodes=30, min_edge_weight=1.0)
print(f"\nPruned:")
print(pruned2.summary())

# Show top edges
sorted_edges2 = sorted(pruned2.edges, key=lambda e: abs(e.weight), reverse=True)
print(f"\nTop 5 edges:")
for e in sorted_edges2[:5]:
    src = pruned2.all_nodes.get(e.source_id)
    tgt = pruned2.all_nodes.get(e.target_id)
    src_info = f"L{src.layer}/f{src.feature_idx}@p{src.position}" if hasattr(src, 'layer') else e.source_id
    tgt_info = f"'{tgt.token_str}'" if hasattr(tgt, 'logit_value') else f"L{tgt.layer}/f{tgt.feature_idx}" if hasattr(tgt, 'layer') else e.target_id
    print(f"  {src_info} → {tgt_info}  w={e.weight:.3f}")

save_graph(pruned2, "/mnt/storage/sandeep/priyansh/Gemma-Scope-2-Study/outputs/attribution_graph_apple.json")

# Memory report
print(f"\n{'='*70}")
print("GPU MEMORY")
print(f"{'='*70}")
print(f"  Allocated: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")

print(f"\n{'='*70}")
print("STEP 5 COMPLETE — Attribution graphs working")
print(f"{'='*70}")
