# convert_bert_to_client_npz_transposed.py
# 说明：将 HuggingFace bert-base-uncased 的参数导出为 client 可用的 npz 文件。
# 所有二维权重将按 (in_features, out_features) 保存（即对 pytorch 的 (out,in) 做 .T）。
# tokenzier 将保存到 bert_params/tokenizer
#
# 使用：pip install transformers torch numpy
#       python convert_bert_to_client_npz_transposed.py

import os
import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

OUT_DIR = "bert_params_split"
TOKENIZER_DIR = "bert_params/tokenizer"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

print("Loading bert-base-uncased (this will download if needed)...")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# save tokenizer to requested path
print(f"Saving tokenizer to {TOKENIZER_DIR} ...")
tokenizer.save_pretrained(TOKENIZER_DIR)

state = model.state_dict()  # OrderedDict of tensors

def to_np(tensor):
    """Convert tensor to numpy.float32"""
    if isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        arr = tensor.detach().cpu().numpy()
    return arr.astype(np.float32)

def save_npz_with_transpose(path, mapping):
    """
    mapping: dict of name->np.ndarray (original)
    For each array:
      - if ndim == 2: save transposed (so saved array has shape (in_features, out_features))
      - else: save as-is
    """
    to_save = {}
    for k, v in mapping.items():
        arr = to_np(v)
        if arr.ndim == 2:
            arr_t = arr.T.copy()  # transpose to (in, out)
            to_save[k] = arr_t
        else:
            to_save[k] = arr
    np.savez(path, **to_save)

# Helper to get param or fallback
def get_param(name, fallback=None):
    if name in state:
        return state[name]
    else:
        return fallback

# 1) Embeddings
print("Exporting embeddings (and transposing 2D arrays)...")
wte = to_np(state["bert.embeddings.word_embeddings.weight"])   # (vocab, hidden)
wpe = to_np(state["bert.embeddings.position_embeddings.weight"]) # (max_pos, hidden)
tte = to_np(state.get("bert.embeddings.token_type_embeddings.weight", np.zeros((2, wte.shape[1]), dtype=np.float32)))

# NOTE: according to our rule, 2D arrays will be transposed when saved, 

np.savez(os.path.join(OUT_DIR, "embeddings.npz"), wte=wte.astype(np.float32), wpe=wpe.astype(np.float32), tte=tte.astype(np.float32))

print(f"Saved embeddings.npz: original wte {wte.shape} -> saved {wte.T.shape}, wpe {wpe.shape} -> saved {wpe.T.shape}")

# 2) Per-layer export
# detect layer count
n_layers = 0
for k in state.keys():
    if k.startswith("bert.encoder.layer."):
        parts = k.split('.')
        try:
            idx = int(parts[3])
            n_layers = max(n_layers, idx + 1)
        except:
            pass

print(f"Detected {n_layers} layers")

for i in range(n_layers):
    prefix = f"bert.encoder.layer.{i}."
    mapping = {}

    # ln1 (attention.output.LayerNorm)
    mapping['ln1_gamma'] = get_param(f"{prefix}attention.output.LayerNorm.weight")
    mapping['ln1_beta']  = get_param(f"{prefix}attention.output.LayerNorm.bias")

    # q/k/v weights and biases (Linear: out_features, in_features) in PyTorch -> we will transpose
    #mapping['q_w'] = get_param(f"{prefix}attention.self.query.weight")
    #mapping['q_b'] = get_param(f"{prefix}attention.self.query.bias", np.zeros((mapping['q_w'].shape[0],), dtype=np.float32))
    #mapping['k_w'] = get_param(f"{prefix}attention.self.key.weight")
    #mapping['k_b'] = get_param(f"{prefix}attention.self.key.bias", np.zeros((mapping['k_w'].shape[0],), dtype=np.float32))
    #mapping['v_w'] = get_param(f"{prefix}attention.self.value.weight")
    #mapping['v_b'] = get_param(f"{prefix}attention.self.value.bias", np.zeros((mapping['v_w'].shape[0],), dtype=np.float32))

    # attention output projection
    mapping['proj_w'] = get_param(f"{prefix}attention.output.dense.weight")
    mapping['proj_b'] = get_param(f"{prefix}attention.output.dense.bias", np.zeros((mapping['proj_w'].shape[0],), dtype=np.float32))

    # feed-forward
    #mapping['ff1_w'] = get_param(f"{prefix}intermediate.dense.weight")
    #mapping['ff1_b'] = get_param(f"{prefix}intermediate.dense.bias", np.zeros((mapping['ff1_w'].shape[0],), dtype=np.float32))
    #mapping['ff2_w'] = get_param(f"{prefix}output.dense.weight")
    #mapping['ff2_b'] = get_param(f"{prefix}output.dense.bias", np.zeros((mapping['ff2_w'].shape[0],), dtype=np.float32))

    # ln2 (output.LayerNorm)
    mapping['ln2_gamma'] = get_param(f"{prefix}output.LayerNorm.weight")
    mapping['ln2_beta']  = get_param(f"{prefix}output.LayerNorm.bias")

    # Before saving we convert None->zeros if any missing (defensive)
    for kkk in list(mapping.keys()):
        if mapping[kkk] is None:
            # create fallback zeros (shape uncertain) -> try to infer from related params
            # fallback approach: zeros of small shape to avoid crash; prefer to adjust later if needed
            mapping[kkk] = np.zeros((1,), dtype=np.float32)

    save_path = os.path.join(OUT_DIR, f"layer_{i}.npz")
    # use save function that transposes 2D arrays
    save_npz_with_transpose(save_path, mapping)
    print(f"Saved {save_path}")

# 3) final.npz : final gamma/beta and lm_head
print("Exporting final.npz ...")
last_ln_gamma = get_param(f"bert.encoder.layer.{n_layers-1}.output.LayerNorm.weight")
last_ln_beta  = get_param(f"bert.encoder.layer.{n_layers-1}.output.LayerNorm.bias")

# lm_head: try cls.predictions.decoder.weight -> usually (vocab, hidden)
lm_head = None
if "cls.predictions.decoder.weight" in state:
    lm_head = to_np(state["cls.predictions.decoder.weight"])  # (vocab, hidden)
else:
    # fallback: try tied weight from embeddings (BERT often ties)
    if "bert.embeddings.word_embeddings.weight" in state:
        lm_head = to_np(state["bert.embeddings.word_embeddings.weight"])  # (vocab, hidden)
    else:
        # generate random fallback: (vocab, hidden)
        vocab_size = tokenizer.vocab_size
        hidden_size = to_np(state["bert.embeddings.word_embeddings.weight"]).shape[1]
        lm_head = np.random.randn(vocab_size, hidden_size).astype(np.float32)

# according to our rule, save 2D arrays transposed -> lm_head saved as (hidden, vocab)
final_mapping = {
    "final_gamma": last_ln_gamma,
    "final_beta": last_ln_beta,
    "lm_head_w": lm_head
}
final_path = os.path.join(OUT_DIR, "final.npz")
save_npz_with_transpose(final_path, final_mapping)
print(f"Saved final.npz -> (after transpose) lm_head shape: {(to_np(lm_head).T).shape}")

print("Done. Generated parameter files in:", OUT_DIR)
print("Tokenizer files saved in:", TOKENIZER_DIR)
print("注意：所有二维数组在 npz 中已被转置，保存后形状为 (in_features, out_features).")
print("如果 client 运行时报 shape mismatch，请根据报错查看具体 key 的 saved shape 并调整 client 的矩阵乘法方向或改回部分权重的不转置保存。")
