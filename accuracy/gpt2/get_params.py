"""
save_gpt2_params.py

用法:
    python save_gpt2_params.py --size 124m --out params_124m.npz

支持 size: 124m, 355m, 774m, 1.5b
会下载 HuggingFace 上对应的 gpt2 权重（如果本地无缓存会从网络下载）。

保存的 npz 包含（示例命名）:
 - wte                : token embedding, shape (vocab_size, n_embd)
 - wpe                : pos embedding,   shape (n_positions, n_embd)
 - ln_f_weight, ln_f_bias : final layernorm params, shape (n_embd,)
 - For each layer i:
    - ln1_weight_i, ln1_bias_i
    - ln2_weight_i, ln2_bias_i
    - c_attn_W_i, c_attn_b_i    (注意：这是 concat [Wq;Wk;Wv] 形式)
    - c_proj_W_i, c_proj_b_i
    - mlp_fc_W_i, mlp_fc_b_i    (c_fc)
    - mlp_proj_W_i, mlp_proj_b_i (c_proj)
"""

import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_MAP = {
    "124m": "gpt2",
    "355m": "gpt2-medium",
    "774m": "gpt2-large",
    "1.5b": "gpt2-xl",
}

def extract_and_save(model_name, out_path):
    print("loading model:", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    state = model.state_dict()

    params = {}
    # embeddings
    params['wte'] = state['transformer.wte.weight'].cpu().numpy()
    params['wpe'] = state['transformer.wpe.weight'].cpu().numpy()
    # final ln
    params['ln_f_weight'] = state['transformer.ln_f.weight'].cpu().numpy()
    params['ln_f_bias'] = state['transformer.ln_f.bias'].cpu().numpy()

    n_layer = 0
    # detect number of layers
    for k in state.keys():
        if k.startswith('transformer.h.'):
            parts = k.split('.')
            # transformer.h.{i}.ln_1.weight
            if parts[2].isdigit():
                n_layer = max(n_layer, int(parts[2]) + 1)
    print("detected n_layer:", n_layer)
    params['n_layer'] = np.array(n_layer)

    # per-layer params
    for i in range(n_layer):
        prefix = f'transformer.h.{i}.'
        # ln1, ln2
        params[f'ln1_weight_{i}'] = state[prefix + 'ln_1.weight'].cpu().numpy()
        params[f'ln1_bias_{i}']   = state[prefix + 'ln_1.bias'].cpu().numpy()
        params[f'ln2_weight_{i}'] = state[prefix + 'ln_2.weight'].cpu().numpy()
        params[f'ln2_bias_{i}']   = state[prefix + 'ln_2.bias'].cpu().numpy()
        # attention: c_attn (combined q,k,v) and c_proj
        params[f'c_attn_W_{i}'] = state[prefix + 'attn.c_attn.weight'].cpu().numpy()
        params[f'c_attn_b_{i}'] = state[prefix + 'attn.c_attn.bias'].cpu().numpy()
        params[f'c_proj_W_{i}'] = state[prefix + 'attn.c_proj.weight'].cpu().numpy()
        params[f'c_proj_b_{i}'] = state[prefix + 'attn.c_proj.bias'].cpu().numpy()
        # mlp
        params[f'mlp_fc_W_{i}']   = state[prefix + 'mlp.c_fc.weight'].cpu().numpy()
        params[f'mlp_fc_b_{i}']   = state[prefix + 'mlp.c_fc.bias'].cpu().numpy()
        params[f'mlp_proj_W_{i}'] = state[prefix + 'mlp.c_proj.weight'].cpu().numpy()
        params[f'mlp_proj_b_{i}'] = state[prefix + 'mlp.c_proj.bias'].cpu().numpy()
        print(f"shape of c_attn_w {i}:", params[f'c_attn_W_{i}'].shape)
    # lm_head: usually tied to embeddings (transformer.wte). We save lm_head if exists separately.
    lm_head_key = 'lm_head.weight'
    if lm_head_key in state:
        params['lm_head'] = state[lm_head_key].cpu().numpy()

    # tokenizer info
    params['vocab_size'] = np.array(tokenizer.vocab_size)
    params['n_positions'] = np.array(model.config.n_positions)
    params['n_ctx'] = np.array(model.config.n_ctx)
    params['n_embd'] = np.array(model.config.n_embd)
    params['n_head'] = np.array(model.config.n_head)
    params['n_inner'] = np.array(getattr(model.config, 'n_inner', -1))

    print("saving to", out_path)
    np.savez(out_path, **params)
    print("done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODEL_MAP.keys(), default="124m")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    model_name = MODEL_MAP[args.size]
    out_path = args.out or f"gpt2_{args.size}_params.npz"
    extract_and_save(model_name, out_path)
