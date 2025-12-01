# save_vit_params.py
import os
import numpy as np
from transformers import AutoConfig, ViTModel, ViTImageProcessor

def save_vit_params():
    # 1. 创建保存目录
    save_dir = "vit_base_params"
    os.makedirs(save_dir, exist_ok=True)

    # 2. 加载模型和 processor（第一次会自动下载，之后走缓存）
    print("正在加载 google/vit-base-patch16-224 ...")
    config = AutoConfig.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # 3. 提取所有需要保存的参数（按层结构组织）
    print("正在提取参数到 NumPy...")

    # Embedding 层
    class_token = model.embeddings.cls_token.detach().cpu().numpy()           # (1, 1, 768)
    patch_embeddings_weight = model.embeddings.patch_embeddings.projection.weight.detach().cpu().numpy()  # (768, 3, 16, 16)
    patch_embeddings_bias   = model.embeddings.patch_embeddings.projection.bias.detach().cpu().numpy()      # (768,)
    position_embeddings = model.embeddings.position_embeddings.detach().cpu().numpy()  # (1, 197, 768)

    # LayerNorm（encoder 前的）
    layernorm_weight = model.layernorm.weight.detach().cpu().numpy()  # (768,)
    layernorm_bias   = model.layernorm.bias.detach().cpu().numpy()    # (768,)

    # 12 个 Transformer Block 的参数
    n_layer = config.num_hidden_layers  # 12

    # 注意：ViT 的每个 block 包含：
    #   attention.attention.{query, key, value, output} + layernorm1
    #   intermediate.dense + output.dense + layernorm2
    attn_q_weight = []
    attn_q_bias   = []
    attn_k_weight = []
    attn_k_bias   = []
    attn_v_weight = []
    attn_v_bias   = []
    attn_out_weight = []
    attn_out_bias   = []
    ln1_gamma = []   # layernorm before attention
    ln1_beta  = []
    ff_dense1_weight = []  # intermediate.dense
    ff_dense1_bias   = []
    ff_dense2_weight = []  # output.dense
    ff_dense2_bias   = []
    ln2_gamma = []   # layernorm before FF
    ln2_beta  = []

    for layer in model.encoder.layer:
        # Attention
        attn_q_weight.append(layer.attention.attention.query.weight.detach().cpu().numpy())
        attn_q_bias.append(layer.attention.attention.query.bias.detach().cpu().numpy())
        attn_k_weight.append(layer.attention.attention.key.weight.detach().cpu().numpy())
        attn_k_bias.append(layer.attention.attention.key.bias.detach().cpu().numpy())
        attn_v_weight.append(layer.attention.attention.value.weight.detach().cpu().numpy())
        attn_v_bias.append(layer.attention.attention.value.bias.detach().cpu().numpy())
        attn_out_weight.append(layer.attention.output.dense.weight.detach().cpu().numpy())
        attn_out_bias.append(layer.attention.output.dense.bias.detach().cpu().numpy())

        # LayerNorm1
        ln1_gamma.append(layer.layernorm_before.weight.detach().cpu().numpy())
        ln1_beta.append(layer.layernorm_before.bias.detach().cpu().numpy())

        # Feed-Forward
        ff_dense1_weight.append(layer.intermediate.dense.weight.detach().cpu().numpy())
        ff_dense1_bias.append(layer.intermediate.dense.bias.detach().cpu().numpy())
        ff_dense2_weight.append(layer.output.dense.weight.detach().cpu().numpy())
        ff_dense2_bias.append(layer.output.dense.bias.detach().cpu().numpy())

        # LayerNorm2
        ln2_gamma.append(layer.layernorm_after.weight.detach().cpu().numpy())
        ln2_beta.append(layer.layernorm_after.bias.detach().cpu().numpy())

    # 转成 numpy array，形状 (num_layers, ...)
    attn_q_weight = np.stack(attn_q_weight)      # (12, 768, 768)
    attn_q_bias   = np.stack(attn_q_bias)        # (12, 768)
    attn_k_weight = np.stack(attn_k_weight)
    attn_k_bias   = np.stack(attn_k_bias)
    attn_v_weight = np.stack(attn_v_weight)
    attn_v_bias   = np.stack(attn_v_bias)
    attn_out_weight = np.stack(attn_out_weight)  # (12, 768, 768)
    attn_out_bias   = np.stack(attn_out_bias)    # (12, 768)

    ln1_gamma = np.stack(ln1_gamma)   # (12, 768)
    ln1_beta  = np.stack(ln1_beta)
    ln2_gamma = np.stack(ln2_gamma)
    ln2_beta  = np.stack(ln2_beta)

    ff_dense1_weight = np.stack(ff_dense1_weight)  # (12, 3072, 768)
    ff_dense1_bias   = np.stack(ff_dense1_bias)    # (12, 3072)
    ff_dense2_weight = np.stack(ff_dense2_weight)  # (12, 768, 3072)
    ff_dense2_bias   = np.stack(ff_dense2_bias)    # (12, 768)

    # 4. 保存为单个 .npz 文件（约 330MB）
    npz_path = os.path.join(save_dir, "vit_params.npz")
    np.savez(npz_path,
             # config
             n_layer=np.array([n_layer]),
             hidden_size=np.array([config.hidden_size]),
             num_attention_heads=np.array([config.num_attention_heads]),
             intermediate_size=np.array([config.intermediate_size]),
             patch_size=np.array([config.patch_size]),
             image_size=np.array([config.image_size]),

             # embeddings
             class_token=class_token,
             patch_embed_weight=patch_embeddings_weight,
             patch_embed_bias=patch_embeddings_bias,
             position_embeddings=position_embeddings,

             # pre-layernorm
             layernorm_weight=layernorm_weight,
             layernorm_bias=layernorm_bias,

             # transformer blocks
             attn_q_weight=attn_q_weight,
             attn_q_bias=attn_q_bias,
             attn_k_weight=attn_k_weight,
             attn_k_bias=attn_k_bias,
             attn_v_weight=attn_v_weight,
             attn_v_bias=attn_v_bias,
             attn_out_weight=attn_out_weight,
             attn_out_bias=attn_out_bias,

             ln1_gamma=ln1_gamma,
             ln1_beta=ln1_beta,
             ff1_weight=ff_dense1_weight,
             ff1_bias=ff_dense1_bias,
             ff2_weight=ff_dense2_weight,
             ff2_bias=ff_dense2_bias,
             ln2_gamma=ln2_gamma,
             ln2_beta=ln2_beta)

    # 5. 保存 processor（只包含 preprocessor_config.json 等几个 KB）
    processor.save_pretrained(os.path.join(save_dir, "processor"))

    print(f"ViT 参数已保存！")
    print(f"   → {npz_path}  ({os.path.getsize(npz_path) / 1024 / 1024:.1f} MB)")
    print(f"   → {save_dir}/processor/  (用于离线图像预处理)")

if __name__ == "__main__":
    save_vit_params()