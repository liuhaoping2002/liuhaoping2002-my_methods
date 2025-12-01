# save_vit_ln_for_gpt2_style.py
import os
import numpy as np
from transformers import ViTModel, AutoConfig

def save_vit_ln_params_gpt2_style():
    # 固定目录和文件名，一点都不改！
    save_dir = "vit_params"
    os.makedirs(save_dir, exist_ok=True)

    print("加载 google/vit-base-patch16-224 ...")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    config = AutoConfig.from_pretrained("google/vit-base-patch16-224")

    n_layer = config.num_hidden_layers  # 12

    # === 严格按 GPT-2 风格命名 ===
    # ln1 → attention 前的 LayerNorm (layernorm_before)
    # ln2 → FFN 前的 LayerNorm (layernorm_after)
    # final → 最后的 layernorm

    ln1_gamma = []
    ln1_beta  = []
    ln2_gamma = []
    ln2_beta  = []

    for layer in model.encoder.layer:
        ln1_gamma.append(layer.layernorm_before.weight.detach().cpu().numpy())  # (768,)
        ln1_beta.append(layer.layernorm_before.bias.detach().cpu().numpy())
        ln2_gamma.append(layer.layernorm_after.weight.detach().cpu().numpy())
        ln2_beta.append(layer.layernorm_after.bias.detach().cpu().numpy())

    # 转成 (12, 768) 的数组
    ln1_gamma = np.stack(ln1_gamma)
    ln1_beta  = np.stack(ln1_beta)
    ln2_gamma = np.stack(ln2_gamma)
    ln2_beta  = np.stack(ln2_beta)

    final_gamma = model.layernorm.weight.detach().cpu().numpy()   # (768,)
    final_beta  = model.layernorm.bias.detach().cpu().numpy()     # (768,)

    # 保存到你指定的路径和键名
    np.savez(
        os.path.join(save_dir, "params.npz"),
        n_layer=np.array([n_layer]),        # 你可能也会用到
        ln1_gamma=ln1_gamma,                # shape: (12, 768)
        ln1_beta=ln1_beta,
        ln2_gamma=ln2_gamma,
        ln2_beta=ln2_beta,
        final_gamma=final_gamma,           # shape: (768,)
        final_beta=final_beta
    )

    print("ViT LayerNorm 参数已精确保存！")

if __name__ == "__main__":
    save_vit_ln_params_gpt2_style()