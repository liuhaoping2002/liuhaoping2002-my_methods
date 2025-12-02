# manual_gpt2_cpu_cuda.py
"""
Manual GPT-2 forward (CPU with numpy OR CUDA with torch).
- 自动检测 CUDA: 如果 torch 可用且 CUDA 可用 => 使用 CUDA 路径 (torch tensors)。
  否则使用 CPU 路径 (numpy).
- 支持从 .npz 加载权重（与 save_gpt2_params.py 导出的结构一致）
  或者直接从 HuggingFace 加载权重并转换为 numpy（可选）。
- 提供 verify_against_hf() 用于与 HuggingFace 官方 model 输出对比验证。

用法示例:
  # 1) 先把 HF 的权重保存为 npz（如果还没保存）
  python save_gpt2_params.py --size 124m --out gpt2_124m_params.npz

  # 2) 再运行手动实现（自动选 GPU 或 CPU）
  python manual_gpt2_cpu_cuda.py --params gpt2_124m_params.npz --text "Hello world"

  # 3) 若想验证和 HF 的输出一致（建议在 CPU 或有单 GPU 的环境）
  python manual_gpt2_cpu_cuda.py --params gpt2_124m_params.npz --text "Hello world" --verify

依赖:
  - numpy
  - transformers (仅用于 tokenizer 和可选的官方模型用于验证)
  - torch (仅当你希望在 CUDA 上运行或做官方 model 对比时)
注意:
  - 大模型在 CPU 上非常慢、内存大。建议先用 small (124m) 做验证。
  - 数值上允许有微小差异（浮点误差），验证时以 max abs error 小于 1e-3~1e-4 为合理。
"""
import argparse
import numpy as np
import sys
import time

# try to import torch; if available we can run CUDA path
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------ Helpers: numpy implementations ------------------
def gelu_np(x: np.ndarray):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi)*(x + 0.044715*np.power(x,3))))

def layer_norm_np(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps=1e-5):
    # x: (..., hidden)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean)**2, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * gamma + beta

def stable_softmax_np(x: np.ndarray, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    s = np.sum(ex, axis=axis, keepdims=True)
    return ex / s

def split_heads_np(x: np.ndarray, n_head: int):
    # x: (batch, seq, hidden) -> (batch, n_head, seq, head_dim)
    batch, seq, hidden = x.shape
    head_dim = hidden // n_head
    x = x.reshape(batch, seq, n_head, head_dim)
    return np.transpose(x, (0,2,1,3))

def merge_heads_np(x: np.ndarray):
    # x: (batch, n_head, seq, head_dim) -> (batch, seq, hidden)
    x = np.transpose(x, (0,2,1,3))
    batch, seq, n_head, head_dim = x.shape
    return x.reshape(batch, seq, n_head * head_dim)

def attention_np(Q, K, V, attn_mask=None):
    # Q,K,V: (batch, n_head, seq, head_dim)
    head_dim = Q.shape[-1]
    scores = np.matmul(Q, np.transpose(K, (0,1,3,2))) / np.sqrt(head_dim)
    if attn_mask is not None:
        scores = scores + attn_mask  # attn_mask should be broadcastable
    probs = stable_softmax_np(scores, axis=-1)
    out = np.matmul(probs, V)
    return out, probs

# ------------------ Helpers: torch implementations ------------------
def gelu_torch(x):
    return 0.5 * x * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0, device=x.device)/torch.tensor(np.pi, device=x.device))*(x + 0.044715*x**3)))

def layer_norm_torch(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean)**2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * gamma + beta

def stable_softmax_torch(x, dim=-1):
    x_max, _ = x.max(dim=dim, keepdim=True)
    ex = torch.exp(x - x_max)
    s = ex.sum(dim=dim, keepdim=True)
    return ex / s

def split_heads_torch(x, n_head):
    # x: (batch, seq, hidden) -> (batch, n_head, seq, head_dim)
    b, seq, hidden = x.shape
    head_dim = hidden // n_head
    x = x.view(b, seq, n_head, head_dim)
    return x.permute(0,2,1,3)

def merge_heads_torch(x):
    # x: (batch, n_head, seq, head_dim) -> (batch, seq, hidden)
    x = x.permute(0,2,1,3).contiguous()
    b, seq, n_head, head_dim = x.shape
    return x.view(b, seq, n_head * head_dim)

def attention_torch(Q, K, V, attn_mask=None):
    head_dim = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2,-1)) / (head_dim ** 0.5)
    if attn_mask is not None:
        scores = scores + attn_mask
    probs = stable_softmax_torch(scores, dim=-1)
    out = torch.matmul(probs, V)
    return out, probs

# ------------------ Load params ------------------
def load_params_npz(path):
    raw = dict(np.load(path, allow_pickle=True))
    params = {}
    for k,v in raw.items():
        params[k] = v
    # ensure float32
    for k,v in list(params.items()):
        if isinstance(v, np.ndarray) and v.dtype == np.float64:
            params[k] = v.astype(np.float32)
    return params

# ------------------ Forward: numpy (CPU) ------------------
def gpt2_forward_numpy(input_ids: np.ndarray, params: dict):
    # input_ids: (batch, seq) ints
    batch, seq = input_ids.shape
    wte = params['wte']       # (vocab, n_embd)
    wpe = params['wpe']       # (n_pos, n_embd)
    n_embd = int(params['n_embd'])
    n_head = int(params['n_head'])
    n_layer = int(params['n_layer'])
    n_pos = int(params['n_positions'])

    assert seq <= n_pos

    x = wte[input_ids]   # (batch, seq, hidden)
    pos_ids = np.arange(seq)[None, :]
    x = x + wpe[pos_ids]

    attn_maps = []
    for i in range(n_layer):
        # LayerNorm1
        ln1_w = params[f'ln1_weight_{i}']
        ln1_b = params[f'ln1_bias_{i}']
        a = layer_norm_np(x, ln1_w, ln1_b)

        c_attn_W = params[f'c_attn_W_{i}']   # (3*hidden, hidden)
        c_attn_b = params[f'c_attn_b_{i}']
        attn_lin = np.dot(a, c_attn_W) + c_attn_b  # (batch, seq, 3*hidden)
        hidden3 = attn_lin.shape[-1]
        hidden = hidden3 // 3
        q = attn_lin[..., :hidden]; k = attn_lin[..., hidden:2*hidden]; v = attn_lin[..., 2*hidden:3*hidden]

        Q = split_heads_np(q, n_head)
        K = split_heads_np(k, n_head)
        V = split_heads_np(v, n_head)

        # causal mask
        seq_len = seq
        causal = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        mask = np.zeros((1,1,seq_len,seq_len), dtype=np.float32)
        mask[:,:,causal] = -1e9

        attn_out, attn_probs = attention_np(Q, K, V, attn_mask=mask)
        attn_maps.append(attn_probs)

        attn_merged = merge_heads_np(attn_out)
        c_proj_W = params[f'c_proj_W_{i}']; c_proj_b = params[f'c_proj_b_{i}']
        proj = np.dot(attn_merged, c_proj_W) + c_proj_b
        x = x + proj

        # MLP
        ln2_w = params[f'ln2_weight_{i}']; ln2_b = params[f'ln2_bias_{i}']
        b_norm = layer_norm_np(x, ln2_w, ln2_b)
        mlp_fc_W = params[f'mlp_fc_W_{i}']; mlp_fc_b = params[f'mlp_fc_b_{i}']
        mlp_proj_W = params[f'mlp_proj_W_{i}']; mlp_proj_b = params[f'mlp_proj_b_{i}']
        hidden_mlp = np.dot(b_norm, mlp_fc_W) + mlp_fc_b
        hidden_act = gelu_np(hidden_mlp)
        mlp_out = np.dot(hidden_act, mlp_proj_W) + mlp_proj_b
        x = x + mlp_out

    # final ln
    ln_f_w = params['ln_f_weight']; ln_f_b = params['ln_f_bias']
    x = layer_norm_np(x, ln_f_w, ln_f_b)

    if 'lm_head' in params:
        lm_head = params['lm_head']
    else:
        lm_head = params['wte']

    logits = np.dot(x, lm_head.T)
    return logits, attn_maps

def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ------------------ Forward: torch (GPU or CPU if user forces) ------------------
def gpt2_forward_torch(input_ids: torch.Tensor, params: dict, device):
    # input_ids: (batch, seq) long on device
    b, seq = input_ids.shape
    # helper to get param as torch tensor on device
    def t(k):
        v = params[k]
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).to(device)
        elif isinstance(v, torch.Tensor):
            return v.to(device)
        else:
            return torch.tensor(v, device=device)
    wte = t('wte')   # (vocab, hidden)
    wpe = t('wpe')
    n_embd = int(params['n_embd'])
    n_head = int(params['n_head'])
    n_layer = int(params['n_layer'])
    n_pos = int(params['n_positions'])

    # embeddings
    x = wte[input_ids]  # advanced indexing
    pos_ids = torch.arange(seq, device=device).unsqueeze(0)
    x = x + wpe[pos_ids]

    attn_maps = []
    for i in range(n_layer):
        ln1_w = t(f'ln1_weight_{i}'); ln1_b = t(f'ln1_bias_{i}')
        a = layer_norm_torch(x, ln1_w, ln1_b)

        c_attn_W = t(f'c_attn_W_{i}'); c_attn_b = t(f'c_attn_b_{i}')
        attn_lin = torch.matmul(a, c_attn_W.t()) + c_attn_b
        hidden3 = attn_lin.shape[-1]; hidden = hidden3 // 3
        q = attn_lin[..., :hidden]; k = attn_lin[..., hidden:2*hidden]; v = attn_lin[..., 2*hidden:3*hidden]

        Q = split_heads_torch(q, n_head)
        K = split_heads_torch(k, n_head)
        V = split_heads_torch(v, n_head)

        seq_len = seq
        # causal mask: shape (1,1,seq,seq)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        mask = torch.zeros((1,1,seq_len,seq_len), device=device, dtype=torch.float32)
        mask[:,:,causal] = -1e9

        attn_out, attn_probs = attention_torch(Q, K, V, attn_mask=mask)
        attn_maps.append(attn_probs.detach().cpu().numpy())

        attn_merged = merge_heads_torch(attn_out)
        c_proj_W = t(f'c_proj_W_{i}'); c_proj_b = t(f'c_proj_b_{i}')
        proj = torch.matmul(attn_merged, c_proj_W.t()) + c_proj_b
        x = x + proj

        ln2_w = t(f'ln2_weight_{i}'); ln2_b = t(f'ln2_bias_{i}')
        b_norm = layer_norm_torch(x, ln2_w, ln2_b)
        mlp_fc_W = t(f'mlp_fc_W_{i}'); mlp_fc_b = t(f'mlp_fc_b_{i}')
        mlp_proj_W = t(f'mlp_proj_W_{i}'); mlp_proj_b = t(f'mlp_proj_b_{i}')
        hidden_mlp = torch.matmul(b_norm, mlp_fc_W.t()) + mlp_fc_b
        hidden_act = gelu_torch(hidden_mlp)
        mlp_out = torch.matmul(hidden_act, mlp_proj_W.t()) + mlp_proj_b
        x = x + mlp_out

    ln_f_w = t('ln_f_weight'); ln_f_b = t('ln_f_bias')
    x = layer_norm_torch(x, ln_f_w, ln_f_b)

    if 'lm_head' in params:
        lm_head = t('lm_head')
    else:
        lm_head = t('wte')

    logits = torch.matmul(x, lm_head.t())
    return logits.detach().cpu().numpy(), attn_maps

# ------------------ Utilities ------------------
def tokenize_texts(texts, tokenizer_name='gpt2'):
    tokenizer = load_tokenizer(tokenizer_name)
    # 如果 tokenizer 没有 pad_token，就把 pad_token 设为 eos_token（常用做法）

    enc = tokenizer(texts, return_tensors='np', padding=True)
    return enc['input_ids'], tokenizer


def verify_against_hf(input_texts, params, device_str, hf_model_name='gpt2'):
    """
    Compute HF model logits and our implementation logits and compare.
    device_str: 'cuda' or 'cpu'
    """
    # tokenizer and hf model
    tokenizer = load_tokenizer(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    hf_model.eval()
    if device_str == 'cuda' and TORCH_AVAILABLE:
        hf_model.to('cuda')

    enc = tokenizer(input_texts, return_tensors='pt', padding=True)
    input_ids_pt = enc['input_ids']
    if device_str == 'cuda' and TORCH_AVAILABLE:
        input_ids_pt = input_ids_pt.cuda()

    # HF logits
    with torch.no_grad():
        out = hf_model(input_ids_pt)
        hf_logits = out.logits.detach().cpu().numpy()

    # Our logits (choose path based on device_str)
    if device_str == 'cuda' and TORCH_AVAILABLE:
        # convert params to numpy if needed (we expect params to have numpy arrays)
        input_ids_np = input_ids_pt.cpu().numpy()
        our_logits, _ = gpt2_forward_torch(torch.from_numpy(input_ids_np).to('cuda'), params, device='cuda')
    else:
        input_ids_np = input_ids_pt.cpu().numpy()
        our_logits, _ = gpt2_forward_numpy(input_ids_np, params)

    # compare
    diff = our_logits - hf_logits
    abs_diff = np.abs(diff)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    mse = float(np.mean(diff**2))
    print("Verification results:")
    print(f"  max abs error = {max_abs:.6e}")
    print(f"  mean abs error = {mean_abs:.6e}")
    print(f"  mse = {mse:.6e}")
    return {'max_abs': max_abs, 'mean_abs': mean_abs, 'mse': mse, 'our_logits': our_logits, 'hf_logits': hf_logits}

# ------------------ Main CLI ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True,
                        help="path to params .npz (from save_gpt2_params.py)")
    parser.add_argument("--text", type=str, default="Hello world")
    parser.add_argument("--verify", action='store_true',
                        help="whether to run verification against HF official model (requires torch)")
    parser.add_argument("--force_cpu", action='store_true', help="force CPU (use numpy)")
    parser.add_argument("--hf_tokenizer", type=str, default="gpt2", help="which HF tokenizer/model to use for tokenization/verify")
    args = parser.parse_args()

    params = load_params_npz(args.params)

    # decide device
    use_cuda = False
    if TORCH_AVAILABLE and torch.cuda.is_available() and not args.force_cpu:
        use_cuda = True
    if args.force_cpu:
        print("force_cpu True -> using numpy CPU path")
    print(f"Torch available: {TORCH_AVAILABLE}, CUDA available: {torch.cuda.is_available() if TORCH_AVAILABLE else False}")
    print(f"Selected device: {'cuda' if use_cuda else 'cpu (numpy)'}")

    # tokenize
    input_ids_np, tokenizer = tokenize_texts([args.text], tokenizer_name=args.hf_tokenizer)
    print("input ids shape:", input_ids_np.shape)

    t0 = time.time()
    if use_cuda:
        device = 'cuda'
        # forward on CUDA using torch path
        input_ids_t = torch.from_numpy(input_ids_np).to(device)
        our_logits, attn_maps = gpt2_forward_torch(input_ids_t, params, device)
    else:
        our_logits, attn_maps = gpt2_forward_numpy(input_ids_np, params)
    t1 = time.time()
    print(f"Inference done in {t1-t0:.3f}s on {'CUDA' if use_cuda else 'CPU (numpy)'}")
    # print topk tokens for last position
    last_logits = our_logits[0, -1]
    topk = 10
    top_idx = np.argsort(-last_logits)[:topk]
    print("Top tokens (id, token, logit):")
    for idx in top_idx:
        tok = tokenizer.decode([int(idx)])
        print(int(idx), repr(tok), float(last_logits[idx]))

    if args.verify:
        if not TORCH_AVAILABLE:
            print("verify requires torch installed (to run HF model). Aborting verify.")
        else:
            device_str = 'cuda' if use_cuda else 'cpu'
            stats = verify_against_hf([args.text], params, device_str, hf_model_name=args.hf_tokenizer)
            print("Verify stats:", stats)

if __name__ == "__main__":
    main()
