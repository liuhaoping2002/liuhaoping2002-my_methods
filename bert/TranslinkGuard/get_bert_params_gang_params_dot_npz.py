# export_bert_params_for_client.py
# 用法: pip install transformers torch numpy
#       python export_bert_params_for_client.py
import os
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

OUT_DIR = "bert_params"
OUT_NPZ = os.path.join(OUT_DIR, "params.npz")
TOKENIZER_DIR = os.path.join(OUT_DIR, "tokenizer")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

print("Loading bert-base-uncased (will download if needed)...")
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
state = model.state_dict()

def to_np(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    else:
        return np.asarray(x, dtype=np.float32)

# detect number of layers
n_layers = 0
for k in state.keys():
    if k.startswith("encoder.layer."):  # note: BertModel state keys are like 'encoder.layer.{i}...'
        parts = k.split('.')
        try:
            idx = int(parts[2])  # parts: ['encoder','layer','{i}',... ] or if prefixed with 'bert.' then adjust
            n_layers = max(n_layers, idx + 1)
        except:
            pass

# Some HF versions use 'bert.encoder.layer.{i}...', handle both
if n_layers == 0:
    for k in state.keys():
        if k.startswith("bert.encoder.layer."):
            parts = k.split('.')
            try:
                idx = int(parts[3])
                n_layers = max(n_layers, idx + 1)
            except:
                pass

if n_layers == 0:
    raise RuntimeError("无法检测到 encoder 层数，请检查 transformers 版本与 state_dict 的 key 格式.")

print("Detected n_layers =", n_layers)

ln1_gamma_list = []
ln1_beta_list = []
ln2_gamma_list = []
ln2_beta_list = []

# key prefix patterns to try (support both "bert.encoder.layer.{i}" and "encoder.layer.{i}")
prefix_patterns = [f"bert.encoder.layer.{i}." for i in range(n_layers)]
# We'll find correct keybase by checking an existing key
sample_key = next(iter(state.keys()))
key_has_bert_prefix = sample_key.startswith("bert.")

for i in range(n_layers):
    if key_has_bert_prefix:
        base = f"bert.encoder.layer.{i}."
    else:
        base = f"encoder.layer.{i}."

    # attention.output.LayerNorm -> ln1
    ln1_gamma_k = base + "attention.output.LayerNorm.weight"
    ln1_beta_k  = base + "attention.output.LayerNorm.bias"
    # output.LayerNorm -> ln2
    ln2_gamma_k = base + "output.LayerNorm.weight"
    ln2_beta_k  = base + "output.LayerNorm.bias"

    # safe get with fallback zeros if missing (but usually present)
    ln1_gamma = to_np(state.get(ln1_gamma_k, np.zeros((model.config.hidden_size,), dtype=np.float32)))
    ln1_beta  = to_np(state.get(ln1_beta_k, np.zeros((model.config.hidden_size,), dtype=np.float32)))
    ln2_gamma = to_np(state.get(ln2_gamma_k, np.zeros((model.config.hidden_size,), dtype=np.float32)))
    ln2_beta  = to_np(state.get(ln2_beta_k, np.zeros((model.config.hidden_size,), dtype=np.float32)))

    ln1_gamma_list.append(ln1_gamma)
    ln1_beta_list.append(ln1_beta)
    ln2_gamma_list.append(ln2_gamma)
    ln2_beta_list.append(ln2_beta)

# final gamma/beta: use last layer's output.LayerNorm
final_gamma = ln2_gamma_list[-1].astype(np.float32)
final_beta  = ln2_beta_list[-1].astype(np.float32)

# stack into ndarrays shape (n_layer, hidden)
ln1_gamma_arr = np.stack(ln1_gamma_list, axis=0).astype(np.float32)
ln1_beta_arr  = np.stack(ln1_beta_list, axis=0).astype(np.float32)
ln2_gamma_arr = np.stack(ln2_gamma_list, axis=0).astype(np.float32)
ln2_beta_arr  = np.stack(ln2_beta_list, axis=0).astype(np.float32)

# save tokenizer
print("Saving tokenizer to", TOKENIZER_DIR)
tokenizer.save_pretrained(TOKENIZER_DIR)

# save the params npz
print("Saving params to", OUT_NPZ)
np.savez(OUT_NPZ,
         n_layer = np.array([n_layers], dtype=np.int32),
         ln1_gamma = ln1_gamma_arr,
         ln1_beta  = ln1_beta_arr,
         ln2_gamma = ln2_gamma_arr,
         ln2_beta  = ln2_beta_arr,
         final_gamma = final_gamma,
         final_beta  = final_beta
         )

print("Done. Saved keys and shapes:")
with np.load(OUT_NPZ) as d:
    for k in d.files:
        print(" ", k, d[k].shape)
