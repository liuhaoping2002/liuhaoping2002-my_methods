import numpy as np
from transformers import GPT2Model

# 加载GPT-2模型
model = GPT2Model.from_pretrained("gpt2")

# 提取参数到numpy数组
n_layer = np.array([model.config.n_layer])

# 自注意力投影权重和偏置（c_attn: QKV投影）
c_attn_w = np.stack([layer.attn.c_attn.weight.detach().cpu().numpy() for layer in model.h])
c_attn_b = np.stack([layer.attn.c_attn.bias.detach().cpu().numpy() for layer in model.h])

# 自注意力输出投影权重和偏置（c_proj）
c_proj_w = np.stack([layer.attn.c_proj.weight.detach().cpu().numpy() for layer in model.h])
c_proj_b = np.stack([layer.attn.c_proj.bias.detach().cpu().numpy() for layer in model.h])

# 前馈网络第一层权重和偏置（mlp.c_fc）
mlp_c_fc_w = np.stack([layer.mlp.c_fc.weight.detach().cpu().numpy() for layer in model.h])
mlp_c_fc_b = np.stack([layer.mlp.c_fc.bias.detach().cpu().numpy() for layer in model.h])

# 前馈网络第二层权重和偏置（mlp.c_proj）
mlp_c_proj_w = np.stack([layer.mlp.c_proj.weight.detach().cpu().numpy() for layer in model.h])
mlp_c_proj_b = np.stack([layer.mlp.c_proj.bias.detach().cpu().numpy() for layer in model.h])

# LM头权重（wte.weight.T）
lm_head_w = model.wte.weight.detach().cpu().numpy().T  # (d_model, vocab_size)

# 保存到NPZ文件
np.savez('gpt2_server_params.npz',
         n_layer=n_layer,
         c_attn_w=c_attn_w,
         c_attn_b=c_attn_b,
         c_proj_w=c_proj_w,
         c_proj_b=c_proj_b,
         mlp_c_fc_w=mlp_c_fc_w,
         mlp_c_fc_b=mlp_c_fc_b,
         mlp_c_proj_w=mlp_c_proj_w,
         mlp_c_proj_b=mlp_c_proj_b,
         lm_head_w=lm_head_w)

print("GPT-2 parameters downloaded and saved to 'gpt2_server_params.npz'.")