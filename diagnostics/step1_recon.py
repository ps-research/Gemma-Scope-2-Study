"""
Step 1: Reconnaissance - GPU info, Gemma 3 1B architecture, CLT file sizes.
Run: python step1_recon.py
"""
import subprocess
import sys

# ============================================================
# 1. GPU INFO
# ============================================================
print("=" * 70)
print("1. GPU INFORMATION")
print("=" * 70)
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"nvidia-smi failed: {e}")

# Check which GPU we'll use
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f"         Total memory: {mem:.1f} GB")

# ============================================================
# 2. GEMMA 3 1B ARCHITECTURE
# ============================================================
print("\n" + "=" * 70)
print("2. GEMMA 3 1B ARCHITECTURE")
print("=" * 70)
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("google/gemma-3-1b-pt")
    print(f"hidden_size (d_model):    {cfg.hidden_size}")
    print(f"num_hidden_layers:        {cfg.num_hidden_layers}")
    print(f"num_attention_heads:      {cfg.num_attention_heads}")
    print(f"num_key_value_heads:      {getattr(cfg, 'num_key_value_heads', 'N/A')}")
    print(f"intermediate_size (MLP):  {cfg.intermediate_size}")
    print(f"head_dim:                 {getattr(cfg, 'head_dim', 'N/A')}")
    print(f"vocab_size:               {cfg.vocab_size}")
    print(f"max_position_embeddings:  {getattr(cfg, 'max_position_embeddings', 'N/A')}")
    print(f"model_type:               {cfg.model_type}")
    
    # Print full config for reference
    print("\n--- FULL CONFIG ---")
    for k, v in sorted(cfg.to_dict().items()):
        print(f"  {k}: {v}")
except Exception as e:
    print(f"Config loading failed: {e}")
    print("You may need: pip install transformers")

# ============================================================
# 3. CLT FILE SIZES ON HUGGINGFACE
# ============================================================
print("\n" + "=" * 70)
print("3. CLT FILE SIZES (from HuggingFace)")
print("=" * 70)
try:
    from huggingface_hub import HfApi
    api = HfApi()
    
    items = list(api.list_repo_tree(
        "google/gemma-scope-2-1b-pt",
        recursive=True,
    ))
    
    clt_files = []
    all_dirs = set()
    
    for item in items:
        if hasattr(item, "size"):
            if "clt/" in item.rfilename:
                clt_files.append((item.rfilename, item.size))
        else:
            name = item.rfilename if hasattr(item, "rfilename") else str(item)
            if "clt" in name.lower():
                all_dirs.add(name)
    
    print("\n--- CLT DIRECTORIES ---")
    for d in sorted(all_dirs):
        print(f"  {d}")
    
    print(f"\n--- CLT FILES ({len(clt_files)} total) ---")
    total_clt = 0
    # Group by parent folder
    from collections import defaultdict
    folders = defaultdict(list)
    for fname, size in clt_files:
        folder = "/".join(fname.split("/")[:-1])
        folders[folder].append((fname, size))
    
    for folder in sorted(folders.keys()):
        files = folders[folder]
        folder_total = sum(s for _, s in files)
        total_clt += folder_total
        gb = folder_total / (1024**3)
        print(f"\n  [{folder}] — {gb:.2f} GB total, {len(files)} files")
        for fname, size in sorted(files)[:3]:  # show first 3
            mb = size / (1024**2)
            print(f"    {mb:>10.1f} MB  {fname.split('/')[-1]}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more files")
    
    print(f"\n  TOTAL CLT SIZE: {total_clt / (1024**3):.2f} GB")

except Exception as e:
    print(f"HuggingFace listing failed: {e}")
    print("You may need: pip install huggingface_hub")

# ============================================================
# 4. ALL TOP-LEVEL FOLDERS IN THE REPO
# ============================================================
print("\n" + "=" * 70)
print("4. ALL TOP-LEVEL FOLDERS IN gemma-scope-2-1b-pt")
print("=" * 70)
try:
    top_folders = defaultdict(lambda: {"count": 0, "size": 0})
    for item in items:
        if hasattr(item, "size"):
            top = item.rfilename.split("/")[0]
            top_folders[top]["count"] += 1
            top_folders[top]["size"] += item.size
    
    for folder in sorted(top_folders.keys()):
        info = top_folders[folder]
        gb = info["size"] / (1024**3)
        print(f"  {folder:30s}  {info['count']:>5d} files  {gb:>8.2f} GB")
    
    grand_total = sum(v["size"] for v in top_folders.values())
    print(f"\n  GRAND TOTAL: {grand_total / (1024**3):.2f} GB")
    
except Exception as e:
    print(f"Folder listing failed: {e}")

# ============================================================
# 5. MEMORY ESTIMATES
# ============================================================
print("\n" + "=" * 70)
print("5. MEMORY ESTIMATES FOR YOUR GPU")
print("=" * 70)
try:
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    
    print(f"Using d_model={d_model}, n_layers={n_layers}")
    print()
    
    # Gemma 3 1B model size
    model_params = 1e9  # approximate
    model_gb = model_params * 2 / (1024**3)  # bf16
    print(f"Gemma 3 1B (bf16):           ~{model_gb:.1f} GB")
    
    # Single-layer SAE sizes
    for width_label, width in [("16k", 16384), ("65k", 65536), ("262k", 262144)]:
        # w_enc + w_dec + threshold + b_enc + b_dec
        params = 2 * d_model * width + width + width + d_model
        gb = params * 4 / (1024**3)  # fp32
        gb_half = params * 2 / (1024**3)  # fp16
        print(f"Single SAE {width_label:>4s} (fp32/fp16): {gb:.2f} / {gb_half:.2f} GB")
    
    # CLT sizes (the big question)
    for width_label, width in [("262k", 262144)]:
        # Per layer: w_enc(d_model, d_sae) + w_dec(d_sae, n_layers, d_model) + threshold + b_enc
        per_layer_enc = d_model * width  # encoder
        per_layer_dec = width * n_layers * d_model  # decoder writes to ALL layers
        per_layer_other = width + width  # threshold + b_enc
        per_layer_total = per_layer_enc + per_layer_dec + per_layer_other
        per_layer_gb_half = per_layer_total * 2 / (1024**3)
        
        total_params = n_layers * per_layer_total + n_layers * d_model  # + b_dec
        total_gb_half = total_params * 2 / (1024**3)
        total_gb_full = total_params * 4 / (1024**3)
        
        print(f"\nCLT {width_label} estimates:")
        print(f"  Per-layer file (fp16):     ~{per_layer_gb_half:.2f} GB")
        print(f"  Full CLT stacked (fp16):   ~{total_gb_half:.1f} GB")
        print(f"  Full CLT stacked (fp32):   ~{total_gb_full:.1f} GB")

except Exception as e:
    print(f"Estimation failed: {e}")

print("\n" + "=" * 70)
print("DONE — paste this full output back")
print("=" * 70)
