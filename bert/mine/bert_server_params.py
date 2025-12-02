# convert_bert_to_server_npz_nopickle.py
# 作用：把 HuggingFace bert-base-uncased 参数导出为 bert_server_params.npz
# 要求：
#  - 所有 2D linear 权重按 (in_features, out_features) 保存（即对 PyTorch 的 (out,in) 做 .T）
#  - 每层权重堆成真实 ndarray（非 object array），例如 c_attn_w -> shape (n_layer, in, 3*hidden)
#  - embeddings 保持 (vocab, hidden)
#  - lm_head 保存为 (hidden, vocab)
#
# 使用：
#   pip install transformers torch numpy
#   python convert_bert_to_server_npz_nopickle.py

import os
import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

OUT_NPZ = "bert_server_params.npz"
TOKENIZER_DIR = "bert_params/tokenizer"
os.makedirs(TOKENIZER_DIR, exist_ok=True)

print("Loading bert-base-uncased ...")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
state = model.state_dict()

# save tokenizer
print(f"Saving tokenizer to {TOKENIZER_DIR} ...")
tokenizer.save_pretrained(TOKENIZER_DIR)

def to_np(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    else:
        return np.asarray(x, dtype=np.float32)

# detect n_layer
n_layers = 0
for k in state.keys():
    if k.startswith("bert.encoder.layer."):
        parts = k.split('.')
        try:
            idx = int(parts[3])
            n_layers = max(n_layers, idx + 1)
        except:
            pass

print("Detected layers:", n_layers)
if n_layers == 0:
    raise RuntimeError("Cannot detect encoder layers in model state dict.")

# collect per-layer arrays into python lists first
c_attn_w_list = []
c_attn_b_list = []
c_proj_w_list = []
c_proj_b_list = []
mlp_c_fc_w_list = []
mlp_c_fc_b_list = []
mlp_c_proj_w_list = []
mlp_c_proj_b_list = []

for i in range(n_layers):
    prefix = f"bert.encoder.layer.{i}."

    # Q K V (PyTorch: (out, in)) -> transpose to (in, out)
    q_w = to_np(state[f"{prefix}attention.self.query.weight"])   # (out, in)
    k_w = to_np(state[f"{prefix}attention.self.key.weight"])
    v_w = to_np(state[f"{prefix}attention.self.value.weight"])
    q_b = to_np(state.get(f"{prefix}attention.self.query.bias", np.zeros((q_w.shape[0],), dtype=np.float32)))
    k_b = to_np(state.get(f"{prefix}attention.self.key.bias", np.zeros((k_w.shape[0],), dtype=np.float32)))
    v_b = to_np(state.get(f"{prefix}attention.self.value.bias", np.zeros((v_w.shape[0],), dtype=np.float32)))

    # transpose each to (in, out)
    q_w_t = q_w.T
    k_w_t = k_w.T
    v_w_t = v_w.T

    # concat on out dimension -> (in, 3*hidden)
    c_attn_w = np.concatenate([q_w_t, k_w_t, v_w_t], axis=1)
    c_attn_b = np.concatenate([q_b, k_b, v_b], axis=0)

    c_attn_w_list.append(c_attn_w)
    c_attn_b_list.append(c_attn_b)

    # attention output proj (out,in) -> transpose
    proj_w = to_np(state[f"{prefix}attention.output.dense.weight"])
    proj_b = to_np(state.get(f"{prefix}attention.output.dense.bias", np.zeros((proj_w.shape[0],), dtype=np.float32)))
    c_proj_w_list.append(proj_w.T)
    c_proj_b_list.append(proj_b)

    # MLP
    ff1_w = to_np(state[f"{prefix}intermediate.dense.weight"])  # (out, in)
    ff1_b = to_np(state.get(f"{prefix}intermediate.dense.bias", np.zeros((ff1_w.shape[0],), dtype=np.float32)))
    ff2_w = to_np(state[f"{prefix}output.dense.weight"])
    ff2_b = to_np(state.get(f"{prefix}output.dense.bias", np.zeros((ff2_w.shape[0],), dtype=np.float32)))

    mlp_c_fc_w_list.append(ff1_w.T)   # (in=hidden, out=intermediate)
    mlp_c_fc_b_list.append(ff1_b)
    mlp_c_proj_w_list.append(ff2_w.T) # (in=intermediate, out=hidden)
    mlp_c_proj_b_list.append(ff2_b)

# Stack lists into real ndarrays (no object arrays)
# Verify consistent shapes across layers
def stack_check(lst, name):
    shapes = [arr.shape for arr in lst]
    first = shapes[0]
    for s in shapes:
        if s != first:
            raise RuntimeError(f"Layer shapes for {name} not consistent: {shapes}")
    return np.stack(lst, axis=0)  # shape (n_layer, ...) 

c_attn_w_arr = stack_check(c_attn_w_list, 'c_attn_w')   # (n_layer, in, 3*hidden)
c_attn_b_arr = stack_check(c_attn_b_list, 'c_attn_b')   # (n_layer, 3*hidden)
c_proj_w_arr = stack_check(c_proj_w_list, 'c_proj_w')   # (n_layer, in, out)
c_proj_b_arr = stack_check(c_proj_b_list, 'c_proj_b')   # (n_layer, out)
mlp_c_fc_w_arr = stack_check(mlp_c_fc_w_list, 'mlp_c_fc_w') # (n_layer, in, intermediate)
mlp_c_fc_b_arr = stack_check(mlp_c_fc_b_list, 'mlp_c_fc_b') # (n_layer, intermediate)
mlp_c_proj_w_arr = stack_check(mlp_c_proj_w_list, 'mlp_c_proj_w') # (n_layer, in=intermediate, out=hidden)
mlp_c_proj_b_arr = stack_check(mlp_c_proj_b_list, 'mlp_c_proj_b') # (n_layer, hidden)

# Embeddings: keep as (vocab, hidden)
wte = to_np(state["bert.embeddings.word_embeddings.weight"])  # (vocab, hidden)
wpe = to_np(state["bert.embeddings.position_embeddings.weight"])  # (max_pos, hidden)

# lm_head: prefer decoder.weight (vocab, hidden) -> convert to (hidden, vocab)
if "cls.predictions.decoder.weight" in state:
    lm_head = to_np(state["cls.predictions.decoder.weight"]).T  # (hidden, vocab)
else:
    # fallback to embedding tie
    lm_head = wte.T  # (hidden, vocab)

# n_layer as 1-element array
n_layer_arr = np.array([n_layers], dtype=np.int32)

# Save everything into npz WITHOUT object arrays
print("Saving to:", OUT_NPZ)
np.savez(OUT_NPZ,
         n_layer=n_layer_arr,
         c_attn_w=c_attn_w_arr,
         c_attn_b=c_attn_b_arr,
         c_proj_w=c_proj_w_arr,
         c_proj_b=c_proj_b_arr,
         mlp_c_fc_w=mlp_c_fc_w_arr,
         mlp_c_fc_b=mlp_c_fc_b_arr,
         mlp_c_proj_w=mlp_c_proj_w_arr,
         mlp_c_proj_b=mlp_c_proj_b_arr,
         lm_head_w=lm_head,
         wte=wte,
         wpe=wpe
         )

print("Saved. Summary shapes:")
print(" c_attn_w:", c_attn_w_arr.shape)
print(" c_attn_b:", c_attn_b_arr.shape)
print(" c_proj_w:", c_proj_w_arr.shape)
print(" mlp_c_fc_w:", mlp_c_fc_w_arr.shape)
print(" lm_head_w:", lm_head.shape)
print(" wte (vocab, hidden):", wte.shape)
print(" wpe (pos, hidden):", wpe.shape)
print(" tokenizer saved at:", TOKENIZER_DIR)
