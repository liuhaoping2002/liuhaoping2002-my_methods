# export_vit_for_gpt2_server.py （终极修正版，转置权重）
import numpy as np
import torch
from transformers import ViTModel

model = ViTModel.from_pretrained("google/vit-base-patch16-224")
print("ViT-base loaded: 12 layers, 768 dim, 12 heads")

n_layer = 12
d_model = 768

# 提取 QKV 并拼接成 [3d, d] 但转置成 [768, 2304] 以兼容服务器 np.dot(inp, w)
c_attn_w_list = []
c_attn_b_list = []
for layer in model.encoder.layer:
    q_w = layer.attention.attention.query.weight.T  # [768, 768] → [768, 768]
    k_w = layer.attention.attention.key.weight.T
    v_w = layer.attention.attention.value.weight.T

    # Q V K 顺序，并转置后 cat on dim=1 (columns) → [768, 2304]
    cat_w = torch.cat([q_w, v_w, k_w], dim=1)  # 注意 dim=1
    cat_b = torch.cat([
        layer.attention.attention.query.bias,
        layer.attention.attention.value.bias,
        layer.attention.attention.key.bias
    ], dim=0)

    c_attn_w_list.append(cat_w.detach().cpu().numpy())
    c_attn_b_list.append(cat_b.detach().cpu().numpy())

c_attn_w = np.stack(c_attn_w_list)  # [12, 768, 2304]
c_attn_b = np.stack(c_attn_b_list)  # [12, 2304]

# c_proj_w：转置成 [768, 768]
c_proj_w_list = []
c_proj_b_list = []
for layer in model.encoder.layer:
    proj_w = layer.attention.output.dense.weight.T  # [768, 768] → [768, 768]
    proj_b = layer.attention.output.dense.bias

    c_proj_w_list.append(proj_w.detach().cpu().numpy())
    c_proj_b_list.append(proj_b.detach().cpu().numpy())

c_proj_w = np.stack(c_proj_w_list)  # [12, 768, 768]
c_proj_b = np.stack(c_proj_b_list)  # [12, 768]

# MLP：转置 up/down
mlp_c_fc_w_list = []
mlp_c_fc_b_list = []
mlp_c_proj_w_list = []
mlp_c_proj_b_list = []
for layer in model.encoder.layer:
    fc_w = layer.intermediate.dense.weight.T  # [768, 3072] → [768, 3072]
    fc_b = layer.intermediate.dense.bias

    proj_w = layer.output.dense.weight.T  # [3072, 768] → [3072, 768]
    proj_b = layer.output.dense.bias

    mlp_c_fc_w_list.append(fc_w.detach().cpu().numpy())
    mlp_c_fc_b_list.append(fc_b.detach().cpu().numpy())
    mlp_c_proj_w_list.append(proj_w.detach().cpu().numpy())
    mlp_c_proj_b_list.append(proj_b.detach().cpu().numpy())

mlp_c_fc_w = np.stack(mlp_c_fc_w_list)  # [12, 768, 3072]
mlp_c_fc_b = np.stack(mlp_c_fc_b_list)  # [12, 3072]
mlp_c_proj_w = np.stack(mlp_c_proj_w_list)  # [12, 3072, 768]
mlp_c_proj_b = np.stack(mlp_c_proj_b_list)  # [12, 768]

np.savez('vit_server_params.npz',
         n_layer=np.array([n_layer]),
         c_attn_w=c_attn_w, c_attn_b=c_attn_b,
         c_proj_w=c_proj_w, c_proj_b=c_proj_b,
         mlp_c_fc_w=mlp_c_fc_w, mlp_c_fc_b=mlp_c_fc_b,
         mlp_c_proj_w=mlp_c_proj_w, mlp_c_proj_b=mlp_c_proj_b)

print("✅ 权重已转置保存！服务器无须改动")
print(f"c_attn_w: {c_attn_w.shape}")  # [12, 768, 2304]
print(f"c_proj_w: {c_proj_w.shape}")   # [12, 768, 768]
print(f"mlp_c_fc_w: {mlp_c_fc_w.shape}")  # [12, 768, 3072]
print(f"mlp_c_proj_w: {mlp_c_proj_w.shape}")  # [12, 3072, 768]