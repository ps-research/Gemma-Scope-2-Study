"""
Step 5b: Verify attribution graph with bug fixes.
Also shows model's actual top predictions to sanity-check.

Run: CUDA_VISIBLE_DEVICES=1 python diagnostics/step5b_verify_attribution.py
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
print("STEP 5b: VERIFY ATTRIBUTION GRAPH (with fixes)")
print("=" * 70)

model, tokenizer = load_gemma3_1b("pt", device="cuda")
clt = load_clt(width="262k", l0="big", affine=True, device="cuda", half_precision=True)

# ============================================================
# 1. First, verify the model actually predicts "Paris"
# ============================================================
prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")

with torch.no_grad():
    logits = model(inputs).logits[0, -1]  # logits at last position

top_vals, top_ids = logits.topk(10)
print(f"\nModel's top 10 predictions for '{prompt}':")
for v, tid in zip(top_vals, top_ids):
    tok = tokenizer.decode([tid.item()]).strip()
    print(f"  {tok:20s}  logit={v.item():.3f}")

# ============================================================
# 2. Build attribution graph
# ============================================================
print(f"\n{'='*70}")
print(f"ATTRIBUTION GRAPH: '{prompt}'")
print(f"{'='*70}")

t0 = time.time()
graph = build_attribution_graph(
    model=model, clt=clt, tokenizer=tokenizer,
    prompt=prompt,
    target_positions=None,
    min_feature_activation=0.0,
    min_edge_weight=0.5,
    top_k_logit_tokens=10,
    same_position_only=True,
)
t_build = time.time() - t0
print(f"\nFull graph ({t_build:.1f}s):")
print(graph.summary())

# ============================================================
# 3. Prune — THIS SHOULD NOW WORK
# ============================================================
print(f"\n{'='*70}")
print("PRUNING")
print(f"{'='*70}")

pruned = prune_graph(
    graph,
    top_k_edges_per_node=5,
    max_feature_nodes=40,
    min_edge_weight=5.0,
)
print(f"\nPruned graph:")
print(pruned.summary())

metrics = compute_graph_metrics(pruned)
print(f"\nPruned metrics:")
for k, v in metrics.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for kk, vv in sorted(v.items()):
            print(f"    Layer {kk}: {vv} features")
    elif isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# ============================================================
# 4. Top edges in pruned graph
# ============================================================
print(f"\n{'='*70}")
print("TOP 15 EDGES IN PRUNED GRAPH")
print(f"{'='*70}")

sorted_edges = sorted(pruned.edges, key=lambda e: abs(e.weight), reverse=True)
for i, e in enumerate(sorted_edges[:15]):
    src = pruned.all_nodes.get(e.source_id)
    tgt = pruned.all_nodes.get(e.target_id)

    if hasattr(src, 'layer'):
        src_str = f"L{src.layer:02d}/f{src.feature_idx:5d} @p{src.position} (act={src.activation:.1f})"
    else:
        src_str = e.source_id

    if hasattr(tgt, 'logit_value'):
        tgt_str = f"→ '{tgt.token_str}' (logit={tgt.logit_value:.2f})"
    elif hasattr(tgt, 'layer'):
        tgt_str = f"→ L{tgt.layer:02d}/f{tgt.feature_idx:5d} @p{tgt.position}"
    else:
        tgt_str = f"→ {e.target_id}"

    print(f"  {i+1:2d}. {src_str}  {tgt_str}  [w={e.weight:.1f}]")

# ============================================================
# 5. Output nodes with incoming edges
# ============================================================
print(f"\n{'='*70}")
print("OUTPUT NODES WITH INCOMING EDGES")
print(f"{'='*70}")

for nid, node in pruned.output_nodes.items():
    incoming = pruned.get_incoming_edges(nid)
    if incoming:
        total_w = sum(e.weight for e in incoming)
        print(f"  '{node.token_str}' (logit={node.logit_value:.3f}): "
              f"{len(incoming)} edges, total_weight={total_w:.1f}")
        for e in sorted(incoming, key=lambda x: abs(x.weight), reverse=True)[:3]:
            src = pruned.feature_nodes.get(e.source_id)
            if src:
                print(f"    ← L{src.layer:02d}/f{src.feature_idx} @p{src.position} "
                      f"(act={src.activation:.1f}, edge_w={e.weight:.1f})")

# ============================================================
# 6. Feature→Feature paths: show multi-hop circuits
# ============================================================
print(f"\n{'='*70}")
print("FEATURE→FEATURE EDGES (top 10)")
print(f"{'='*70}")

ff_edges = [e for e in pruned.edges
            if e.source_id in pruned.feature_nodes and e.target_id in pruned.feature_nodes]
ff_edges.sort(key=lambda e: abs(e.weight), reverse=True)

for i, e in enumerate(ff_edges[:10]):
    src = pruned.feature_nodes[e.source_id]
    tgt = pruned.feature_nodes[e.target_id]
    print(f"  {i+1:2d}. L{src.layer:02d}/f{src.feature_idx:5d} @p{src.position} (act={src.activation:.1f})"
          f" → L{tgt.layer:02d}/f{tgt.feature_idx:5d} @p{tgt.position} (act={tgt.activation:.1f})"
          f" [w={e.weight:.1f}]")

# ============================================================
# 7. Save
# ============================================================
save_graph(pruned, "/workspace/Gemma-Scope-2-Study/outputs/attribution_graph_france_v2.json")

print(f"\n{'='*70}")
print(f"GPU: {torch.cuda.memory_allocated()/(1024**3):.2f} GB allocated")
print("STEP 5b COMPLETE")
print(f"{'='*70}")
