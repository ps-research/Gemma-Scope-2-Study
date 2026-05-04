"""
Step 2: Download ONE CLT layer file + config, inspect actual tensor shapes.
This tells us the true decoder structure.
Run: CUDA_VISIBLE_DEVICES=1 python step2_inspect_shapes.py
"""
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json
import os

WORK_DIR = "/mnt/storage/sandeep/priyansh/Gemma-Scope-2-Study"
os.makedirs(f"{WORK_DIR}/cache", exist_ok=True)

# ============================================================
# 1. INSPECT CLT CONFIG
# ============================================================
print("=" * 70)
print("1. CLT CONFIG FILES")
print("=" * 70)

for variant in [
    "width_262k_l0_medium",
    "width_262k_l0_medium_affine",
    "width_262k_l0_big",
    "width_262k_l0_big_affine",
]:
    print(f"\n--- clt/{variant}/config.json ---")
    try:
        path = hf_hub_download(
            repo_id="google/gemma-scope-2-1b-pt",
            filename=f"clt/{variant}/config.json",
            cache_dir=f"{WORK_DIR}/cache",
        )
        with open(path) as f:
            config = json.load(f)
        for k, v in sorted(config.items()):
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  FAILED: {e}")

# ============================================================
# 2. INSPECT CLT LAYER 0 TENSOR SHAPES
# ============================================================
print("\n" + "=" * 70)
print("2. CLT TENSOR SHAPES (layer 0, width_262k_l0_medium)")
print("=" * 70)

path = hf_hub_download(
    repo_id="google/gemma-scope-2-1b-pt",
    filename="clt/width_262k_l0_medium/params_layer_0.safetensors",
    cache_dir=f"{WORK_DIR}/cache",
)
params = load_file(path, device="cpu")

print("\nTensor names, shapes, dtypes, and sizes:")
total_params = 0
for name in sorted(params.keys()):
    tensor = params[name]
    n_params = tensor.numel()
    total_params += n_params
    size_mb = n_params * tensor.element_size() / (1024**2)
    print(f"  {name:30s}  shape={str(list(tensor.shape)):30s}  dtype={str(tensor.dtype):15s}  {size_mb:>8.1f} MB  ({n_params:>12,} params)")

print(f"\n  Total params: {total_params:,}")
print(f"  Total size (as stored): {sum(t.numel() * t.element_size() for t in params.values()) / (1024**2):.1f} MB")

# ============================================================
# 3. COMPARE WITH AFFINE VARIANT
# ============================================================
print("\n" + "=" * 70)
print("3. CLT TENSOR SHAPES (layer 0, width_262k_l0_medium_affine)")
print("=" * 70)

path = hf_hub_download(
    repo_id="google/gemma-scope-2-1b-pt",
    filename="clt/width_262k_l0_medium_affine/params_layer_0.safetensors",
    cache_dir=f"{WORK_DIR}/cache",
)
params_affine = load_file(path, device="cpu")

for name in sorted(params_affine.keys()):
    tensor = params_affine[name]
    n_params = tensor.numel()
    size_mb = n_params * tensor.element_size() / (1024**2)
    print(f"  {name:30s}  shape={str(list(tensor.shape)):30s}  dtype={str(tensor.dtype):15s}  {size_mb:>8.1f} MB")

# ============================================================
# 4. CHECK A MIDDLE LAYER (is decoder shape same or different?)
# ============================================================
print("\n" + "=" * 70)
print("4. CLT TENSOR SHAPES (layer 13 vs layer 0 — checking if decoder grows)")
print("=" * 70)

path_13 = hf_hub_download(
    repo_id="google/gemma-scope-2-1b-pt",
    filename="clt/width_262k_l0_medium/params_layer_13.safetensors",
    cache_dir=f"{WORK_DIR}/cache",
)
params_13 = load_file(path_13, device="cpu")

for name in sorted(params_13.keys()):
    t0 = params[name]
    t13 = params_13[name]
    match = "SAME" if list(t0.shape) == list(t13.shape) else "DIFFERENT!"
    print(f"  {name:30s}  layer_0={str(list(t0.shape)):25s}  layer_13={str(list(t13.shape)):25s}  {match}")

# ============================================================
# 5. ALSO INSPECT A SINGLE-LAYER TRANSCODER FOR COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("5. SINGLE-LAYER TRANSCODER SHAPES (layer 13, 65k, medium, affine)")
print("=" * 70)

path_tc = hf_hub_download(
    repo_id="google/gemma-scope-2-1b-pt",
    filename="transcoder/layer_13_width_65k_l0_medium_affine/params.safetensors",
    cache_dir=f"{WORK_DIR}/cache",
)
params_tc = load_file(path_tc, device="cpu")

for name in sorted(params_tc.keys()):
    tensor = params_tc[name]
    n_params = tensor.numel()
    size_mb = n_params * tensor.element_size() / (1024**2)
    print(f"  {name:30s}  shape={str(list(tensor.shape)):30s}  dtype={str(tensor.dtype):15s}  {size_mb:>8.1f} MB")

# ============================================================
# 6. INSPECT A RESIDUAL SAE AND CROSSCODER TOO
# ============================================================
print("\n" + "=" * 70)
print("6. RESIDUAL SAE SHAPES (layer 22, 65k, medium)")
print("=" * 70)

path_sae = hf_hub_download(
    repo_id="google/gemma-scope-2-1b-pt",
    filename="resid_post/layer_22_width_65k_l0_medium/params.safetensors",
    cache_dir=f"{WORK_DIR}/cache",
)
params_sae = load_file(path_sae, device="cpu")

for name in sorted(params_sae.keys()):
    tensor = params_sae[name]
    size_mb = tensor.numel() * tensor.element_size() / (1024**2)
    print(f"  {name:30s}  shape={str(list(tensor.shape)):30s}  dtype={str(tensor.dtype):15s}  {size_mb:>8.1f} MB")

print("\n" + "=" * 70)
print("7. CROSSCODER SHAPES (layer 0 of 4-layer crosscoder, 262k, medium)")
print("=" * 70)

path_cc = hf_hub_download(
    repo_id="google/gemma-scope-2-1b-pt",
    filename="crosscoder/layer_7_13_17_22_width_262k_l0_medium/params_layer_0.safetensors",
    cache_dir=f"{WORK_DIR}/cache",
)
params_cc = load_file(path_cc, device="cpu")

for name in sorted(params_cc.keys()):
    tensor = params_cc[name]
    size_mb = tensor.numel() * tensor.element_size() / (1024**2)
    print(f"  {name:30s}  shape={str(list(tensor.shape)):30s}  dtype={str(tensor.dtype):15s}  {size_mb:>8.1f} MB")

# ============================================================
# 8. MEMORY PLAN
# ============================================================
print("\n" + "=" * 70)
print("8. REVISED MEMORY PLAN (based on actual sizes)")
print("=" * 70)

# Actual CLT size from repo
clt_262k_total_gb = 30.37
clt_524k_total_gb = 60.74
model_gb = 1.9

print(f"Gemma 3 1B (bf16):                    {model_gb:.1f} GB")
print(f"CLT 262k (full, from repo):           {clt_262k_total_gb:.1f} GB")
print(f"CLT 524k (full, from repo):           {clt_524k_total_gb:.1f} GB")
print(f"Available GPU memory:                 79.3 GB")
print()
print(f"Model + CLT 262k =                   {model_gb + clt_262k_total_gb:.1f} GB  {'FITS!' if model_gb + clt_262k_total_gb < 79 else 'TOO BIG'}")
print(f"Model + CLT 524k =                   {model_gb + clt_524k_total_gb:.1f} GB  {'FITS!' if model_gb + clt_524k_total_gb < 79 else 'TOO BIG'}")
print(f"Model + CLT 262k + overhead (~5GB) =  {model_gb + clt_262k_total_gb + 5:.1f} GB  {'FITS!' if model_gb + clt_262k_total_gb + 5 < 79 else 'TIGHT'}")

print("\n" + "=" * 70)
print("DONE — paste this full output back")
print("=" * 70)
