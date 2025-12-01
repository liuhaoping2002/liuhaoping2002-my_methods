# export_vit_client_params.py
import os
import numpy as np
from transformers import ViTModel

os.makedirs("vit_params", exist_ok=True)
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# 每层参数（和 GPT-2 客户端结构一模一样）
for i, layer in enumerate(model.encoder.layer):
    ln1_gamma = layer.layernorm_before.weight.detach().cpu().numpy()
    ln1_beta  = layer.layernorm_before.bias.detach().cpu().numpy()
    ln2_gamma = layer.layernorm_after.weight.detach().cpu().numpy()
    ln2_beta  = layer.layernorm_after.bias.detach().cpu().numpy()
    
    np.savez(f"vit_params/layer_{i}.npz",
             ln1_gamma=ln1_gamma, ln1_beta=ln1_beta,
             ln2_gamma=ln2_gamma, ln2_beta=ln2_beta)

# Final LayerNorm
np.savez("vit_params/final.npz",
         final_gamma=model.layernorm.weight.detach().cpu().numpy(),
         final_beta=model.layernorm.bias.detach().cpu().numpy())

print("ViT 客户端权重已生成到 vit_params/")
