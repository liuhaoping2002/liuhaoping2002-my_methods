import numpy as np

def sample_A_constructive(d, a=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    one = np.ones((d, 1))
    u0 = one / np.sqrt(d)      # (d,1)
    R = rng.normal(size=(d, d-1))
    R = R - u0 @ (u0.T @ R)    # 项目到正交子空间
    Q_full, _ = np.linalg.qr(R, mode='reduced')  # Q_full: d × (d-1)
    Us = Q_full[:, :d-1]  # d × (d-1)
    G = rng.normal(size=(d-1, d-1))
    Qs, Rg = np.linalg.qr(G)
    D = np.sign(np.diag(Rg))
    D[D == 0] = 1.0
    Q_small = Qs @ np.diag(D)           # 正确方式：列符号修正
    Q = Us @ (Q_small @ Us.T)           # embed back
    J = (one @ one.T) / d
    P = np.eye(d) - J
    A = a * J + Q @ P                   # 原代码就是 a*J + Q (因为 QJ=0, QP=Q)
    return A

def sample_orthogonal(d, rng):
    """采样 Haar 分布的 O(d) 矩阵（特殊正交概率1）"""
    G = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q @ np.diag(signs)

# -------------------------- 参数路径请根据实际情况修改 --------------------------
SERVER_PARAMS_PATH = 'gpt2_server_params.npz'
CLIENT_PARAMS_PATH = 'gpt2_params/params.npz'   # 原 client 参数文件夹下的 params.npz

OBF_SERVER_PATH = 'gpt2_server_params_obf.npz'
OBF_CLIENT_PATH = 'gpt2_client_params_obf.npz'

# -------------------------- 加载原参数 --------------------------
data = np.load(SERVER_PARAMS_PATH, allow_pickle=False)
client_data = np.load(CLIENT_PARAMS_PATH, allow_pickle=False)

n_layer = int(data['n_layer'][0])
d_model = 768
head_dim = d_model // 12   # 64，但这里用不到位，只用作切片参考

# -------------------------- 生成全局同一个 A --------------------------
rng_A = np.random.default_rng(2025)
A = sample_A_constructive(d_model, a=6.88, rng=rng_A)
A_inv = np.linalg.inv(A)

# -------------------------- 为每一层准备容器 --------------------------
c_attn_w_new = []
c_attn_b_new = []
c_proj_w_new = []
c_proj_b_new = []   # c_proj_b 保持不变
mlp_fc_w_new = []
mlp_fc_b_new = []
mlp_proj_w_new = []
mlp_proj_b_new = [] # mlp_proj_b 保持不变

# -------------------------- 对每一层执行变换 --------------------------
for layer in range(n_layer):
    # ---------- Attn LN1 (ln1) ----------
    gamma_attn = client_data['ln1_gamma'][layer]      # (768,)
    beta_attn  = client_data['ln1_beta'][layer]       # (768,)
    D_attn = np.diag(gamma_attn)                      # (768,768)
    b_attn_row = beta_attn[None, :]                   # (1,768)

    # 每层独立的 B、C（增强安全性）
    rng_b = np.random.default_rng(2025 + layer * 10)
    rng_c = np.random.default_rng(2025 + layer * 10 + 10000)
    B = sample_orthogonal(d_model, rng_b)             # O(d)
    C = sample_orthogonal(d_model, rng_c)

    QKV = data['c_attn_w'][layer]                   # (768, 2304)
    Wq = QKV[:, :d_model]
    Wk = QKV[:, d_model:2*d_model]
    Wv = QKV[:, 2*d_model:]

    Wq_new = A_inv @ D_attn @ Wq @ B + b_attn_row @ (Wq @ B)
    Wk_new = A_inv @ D_attn @ Wk @ B.T + b_attn_row @ (Wk @ B.T)
    Wv_new = A_inv @ D_attn @ Wv @ C + b_attn_row @ (Wv @ C)

    c_attn_w_layer_new = np.concatenate([Wq_new, Wk_new, Wv_new], axis=1)
    c_attn_w_new.append(c_attn_w_layer_new)

    # bias 吸收（rank-1 update 部分）
    old_attn_b = data['c_attn_b'][layer]             # (2304,)
    add_b_q = (b_attn_row @ (Wq @ B)).flatten()
    add_b_k = (b_attn_row @ (Wk @ B.T)).flatten()
    add_b_v = (b_attn_row @ (Wv @ C)).flatten()
    c_attn_b_layer_new = old_attn_b + np.concatenate([add_b_q, add_b_k, add_b_v])
    c_attn_b_new.append(c_attn_b_layer_new)

    # Wo (c_proj)
    Wo_old = data['c_proj_w'][layer]                 # (768,768)
    Wo_new = np.linalg.inv(C) @ Wo_old @ A
    c_proj_w_new.append(Wo_new)
    c_proj_b_new.append(data['c_proj_b'][layer])       # 不变

    # ---------- FFN LN2 ----------
    gamma_ff = client_data['ln2_gamma'][layer]
    beta_ff  = client_data['ln2_beta'][layer]
    D_ff = np.diag(gamma_ff)
    b_ff_row = beta_ff[None, :]

    W_fc_old = data['mlp_c_fc_w'][layer]            # (768, 3072)
    W_fc_new = A_inv @ D_ff @ W_fc_old + b_ff_row @ W_fc_old
    mlp_fc_w_new.append(W_fc_new)

    add_b_fc = (b_ff_row @ W_fc_old).flatten()
    mlp_fc_b_new.append(data['mlp_c_fc_b'][layer] + add_b_fc)

    W_proj_old = data['mlp_c_proj_w'][layer]         # (3072, 768)
    W_proj_new = W_proj_old @ A
    mlp_proj_w_new.append(W_proj_new)
    mlp_proj_b_new.append(data['mlp_c_proj_b'][layer])  # 不变

# -------------------------- 组装最终数组 --------------------------
c_attn_w_new = np.array(c_attn_w_new)
c_attn_b_new = np.array(c_attn_b_new)
c_proj_w_new = np.array(c_proj_w_new)
c_proj_b_new = np.array(c_proj_b_new)
mlp_fc_w_new = np.array(mlp_fc_w_new)
mlp_fc_b_new = np.array(mlp_fc_b_new)
mlp_proj_w_new = np.array(mlp_proj_w_new)
mlp_proj_b_new = np.array(mlp_proj_b_new)

# -------------------------- 保存 obfuscated server 参数 --------------------------
obf_server = dict(data)  # 复制所有原有字段
obf_server['c_attn_w'] = c_attn_w_new
obf_server['c_attn_b'] = c_attn_b_new
obf_server['c_proj_w'] = c_proj_w_new
obf_server['c_proj_b'] = c_proj_b_new
obf_server['mlp_c_fc_w'] = mlp_fc_w_new
obf_server['mlp_c_fc_b'] = mlp_fc_b_new
obf_server['mlp_c_proj_w'] = mlp_proj_w_new
obf_server['mlp_c_proj_b'] = mlp_proj_b_new
# lm_head_w 保持原样（如果要 rotated logits，可改为 lm_head_w @ A_inv）
obf_server['lm_head_w'] = data['lm_head_w']

np.savez(OBF_SERVER_PATH, **obf_server)
print(f"Server 参数已保存至 {OBF_SERVER_PATH}")

# -------------------------- 保存 obfuscated client 参数（所有 LN → identity） --------------------------
obf_client = dict(client_data)
obf_client['ln1_gamma'] = np.ones_like(client_data['ln1_gamma'])
obf_client['ln1_beta']  = np.zeros_like(client_data['ln1_beta'])
obf_client['ln2_gamma'] = np.ones_like(client_data['ln2_gamma'])
obf_client['ln2_beta']  = np.zeros_like(client_data['ln2_beta'])
obf_client['final_gamma'] = np.ones_like(client_data['final_gamma'])
obf_client['final_beta']  = np.zeros_like(client_data['final_beta'])

np.savez(OBF_CLIENT_PATH, **obf_client)
print(f"Client LN 参数已保存至 {OBF_CLIENT_PATH}")
print("混淆完成！")

# 使用方法：
# 在原 REE server 代码中，只需要把两条加载路径改为：
# data = np.load('gpt2_server_params_obf.npz')
# client_data = np.load('gpt2_client_params_obf.npz')
# 其余代码（包括 A = sample_A_constructive(..., a=6.88, rng=2025) 那段验证打印）完全不用改，仍然可以正常跑且水印验证打印依旧正确。