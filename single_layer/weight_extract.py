import torch
from transformers import GPT2Model, GPT2Config
import numpy as np

# 加载 GPT-2 small
config = GPT2Config.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# 提取第一个 Transformer layer (h[0])
layer = model.h[0]

# Self-Attention 权重
c_attn_weight = layer.attn.c_attn.weight.detach().numpy()  # (768, 2304)
w_q = c_attn_weight[:, :768]  # (768, 768)
w_k = c_attn_weight[:, 768:1536]  # (768, 768)
w_v = c_attn_weight[:, 1536:]  # (768, 768)
w_o = layer.attn.c_proj.weight.detach().numpy()  # (768, 768)

# FFN 权重
ff_linear1 = layer.mlp.c_fc.weight.detach().numpy()  # (768, 3072)
ff_linear2 = layer.mlp.c_proj.weight.detach().numpy()  # (3072, 768)

# LayerNorm 参数
ln1_weight = layer.ln_1.weight.detach().numpy()  # (768,)
ln1_bias = layer.ln_1.bias.detach().numpy()  # (768,)
ln2_weight = layer.ln_2.weight.detach().numpy()  # (768,)
ln2_bias = layer.ln_2.bias.detach().numpy()  # (768,)

# 保存为 .npy
np.save('w_q.npy', w_q)
np.save('w_k.npy', w_k)
np.save('w_v.npy', w_v)
np.save('w_o.npy', w_o)
np.save('ff_linear1.npy', ff_linear1)
np.save('ff_linear2.npy', ff_linear2)
np.save('ln1_weight.npy', ln1_weight)
np.save('ln1_bias.npy', ln1_bias)
np.save('ln2_weight.npy', ln2_weight)
np.save('ln2_bias.npy', ln2_bias)

print("All weights saved.")