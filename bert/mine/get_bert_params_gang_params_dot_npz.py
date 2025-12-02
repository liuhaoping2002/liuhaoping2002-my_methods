# export_bert_params_for_client_with_embeddings_fix.py
# 可靠导出 BERT 参数到 bert_params/params.npz，确保 wte 和 wpe 正确写入并打印 key/shape。
import os
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

OUT_DIR = "bert_params"
OUT_NPZ = os.path.join(OUT_DIR, "params.npz")
TOKENIZER_DIR = os.path.join(OUT_DIR, "tokenizer")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

print("Loading bert-base-uncased ...")
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
state = model.state_dict()
cfg = model.config

def to_np(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    else:
        return None

# detect n_layer robustly
n_layers = 0
for k in state.keys():
    if "encoder.layer." in k:
        parts = k.split('.')
        # try to get layer index (works for both 'bert.encoder.layer.{i}...' and 'encoder.layer.{i}...')
        for pidx, p in enumerate(parts):
            if p == 'layer' and pidx+1 < len(parts):
                try:
                    idx = int(parts[pidx+1])
                    n_layers = max(n_layers, idx + 1)
                except:
                    pass

if n_layers == 0:
    raise RuntimeError("无法检测 encoder 层数，请检查 state_dict 的 key 命名。")

print("Detected n_layers =", n_layers)

ln1_gamma_list = []
ln1_beta_list = []
ln2_gamma_list = []
ln2_beta_list = []

# determine prefix style
sample_key = next(iter(state.keys()))
key_has_bert_prefix = sample_key.startswith("bert.")

# helper to try multiple candidate keys for a single param
def find_key(candidates):
    for k in candidates:
        if k in state:
            return k
    return None

for i in range(n_layers):
    # build candidate bases robustly: check both with and without 'bert.' prefix
    possible_bases = []
    if key_has_bert_prefix:
        possible_bases.append(f"bert.encoder.layer.{i}.")
        possible_bases.append(f"bert.encoder.layer.{i}.")  # redundant but explicit
    possible_bases.append(f"encoder.layer.{i}.")
    # try typical endings
    ln1_gamma_k = None
    ln1_beta_k  = None
    ln2_gamma_k = None
    ln2_beta_k  = None
    for base in possible_bases:
        if ln1_gamma_k is None and (base + "attention.output.LayerNorm.weight") in state:
            ln1_gamma_k = base + "attention.output.LayerNorm.weight"
            ln1_beta_k  = base + "attention.output.LayerNorm.bias"
        if ln2_gamma_k is None and (base + "output.LayerNorm.weight") in state:
            ln2_gamma_k = base + "output.LayerNorm.weight"
            ln2_beta_k  = base + "output.LayerNorm.bias"
    # final fallback: attempt common alternative names
    if ln1_gamma_k is None:
        ln1_gamma_k = find_key([f"layer.{i}.attention.output.LayerNorm.weight", f"layer.{i}.attention.output.LayerNorm.weight"])
    if ln2_gamma_k is None:
        ln2_gamma_k = find_key([f"layer.{i}.output.LayerNorm.weight"])

    # fetch arrays robustly
    ln1_gamma = to_np(state.get(ln1_gamma_k)) if ln1_gamma_k else None
    ln1_beta  = to_np(state.get(ln1_beta_k))  if ln1_beta_k else None
    ln2_gamma = to_np(state.get(ln2_gamma_k)) if ln2_gamma_k else None
    ln2_beta  = to_np(state.get(ln2_beta_k))  if ln2_beta_k else None

    if ln1_gamma is None or ln1_beta is None:
        # fallback to zeros but warn
        print(f"Warning: missing ln1 for layer {i}, keys tried: ln1_gamma_k={ln1_gamma_k}, using zeros")
        ln1_gamma = np.zeros((cfg.hidden_size,), dtype=np.float32)
        ln1_beta  = np.zeros((cfg.hidden_size,), dtype=np.float32)
    if ln2_gamma is None or ln2_beta is None:
        print(f"Warning: missing ln2 for layer {i}, keys tried: ln2_gamma_k={ln2_gamma_k}, using zeros")
        ln2_gamma = np.zeros((cfg.hidden_size,), dtype=np.float32)
        ln2_beta  = np.zeros((cfg.hidden_size,), dtype=np.float32)

    ln1_gamma_list.append(ln1_gamma)
    ln1_beta_list.append(ln1_beta)
    ln2_gamma_list.append(ln2_gamma)
    ln2_beta_list.append(ln2_beta)

ln1_gamma_arr = np.stack(ln1_gamma_list, axis=0).astype(np.float32)
ln1_beta_arr  = np.stack(ln1_beta_list, axis=0).astype(np.float32)
ln2_gamma_arr = np.stack(ln2_gamma_list, axis=0).astype(np.float32)
ln2_beta_arr  = np.stack(ln2_beta_list, axis=0).astype(np.float32)
final_gamma = ln2_gamma_arr[-1].astype(np.float32)
final_beta  = ln2_beta_arr[-1].astype(np.float32)

# === Robust embeddings lookup ===
# Search for the best key candidates for word and position embeddings
word_candidates = [
    "bert.embeddings.word_embeddings.weight",
    "embeddings.word_embeddings.weight",
    "word_embeddings.weight",
    "bert.embeddings.word_embeddings.weight",
    "bert.embeddings.token_embeddings.weight"
]
pos_candidates = [
    "bert.embeddings.position_embeddings.weight",
    "embeddings.position_embeddings.weight",
    "position_embeddings.weight",
    "bert.embeddings.position_embeddings.weight"
]

wte_key = find_key(word_candidates)
wpe_key = find_key(pos_candidates)

if wte_key is None:
    # try broader search: any key containing 'word_embeddings' or 'word_embed'
    for k in state.keys():
        if 'word_embeddings' in k or 'word_embed' in k:
            wte_key = k
            break

if wpe_key is None:
    for k in state.keys():
        if 'position_embeddings' in k or 'position_embed' in k:
            wpe_key = k
            break

if wte_key is None or wpe_key is None:
    print("ERROR: cannot find embedding keys automatically.")
    print("Available keys sample (first 80):")
    print(list(state.keys())[:80])
    raise RuntimeError("Failed to locate wte or wpe keys in state_dict. Inspect keys above.")

wte = to_np(state[wte_key])
wpe = to_np(state[wpe_key])

print(f"Found embeddings keys: wte_key={wte_key}, wpe_key={wpe_key}")
print("wte shape:", None if wte is None else wte.shape)
print("wpe shape:", None if wpe is None else wpe.shape)

# Save tokenizer
print("Saving tokenizer to", TOKENIZER_DIR)
tokenizer.save_pretrained(TOKENIZER_DIR)

# Save params.npz
print("Saving params to", OUT_NPZ)
np.savez(OUT_NPZ,
         n_layer = np.array([n_layers], dtype=np.int32),
         ln1_gamma = ln1_gamma_arr,
         ln1_beta  = ln1_beta_arr,
         ln2_gamma = ln2_gamma_arr,
         ln2_beta  = ln2_beta_arr,
         final_gamma = final_gamma,
         final_beta  = final_beta,
         wte = wte.astype(np.float32),
         wpe = wpe.astype(np.float32)
         )

print("Saved. Contents and shapes:")
with np.load(OUT_NPZ) as d:
    for k in d.files:
        print(" ", k, d[k].shape)
print("Tokenizer vocab_size:", tokenizer.vocab_size)
