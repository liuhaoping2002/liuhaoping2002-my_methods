# client.py (TEE)
import grpc
import numpy as np
import io
import time
import demo_pb2
import demo_pb2_grpc
from transformers import AutoTokenizer
from scipy.special import softmax as sp_softmax

def time_cost(outputs, time_past):
    time_now = time.time()
    time_cost = time_now - time_past
    print(f"{outputs} cost {time_cost*1000:.2f} ms")
    return time_now

def np_to_tensor(arr: np.ndarray) -> demo_pb2.Tensor:
    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=False)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return demo_pb2.Tensor(
        data=buf.getvalue(),
        shape=list(arr.shape),
        dtype=str(arr.dtype)
    )

def tensor_to_np(t: demo_pb2.Tensor) -> np.ndarray:
    buf = io.BytesIO(t.data)
    arr = np.load(buf)
    return arr.copy()

def np_to_state(state_np):
    return {k: np_to_tensor(v) for k, v in state_np.items()}

def state_to_np(state_pb):
    return {k: tensor_to_np(v) for k, v in state_pb.items.items()}

def sample_A_constructive(d, a=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    one = np.ones((d,1))
    u0 = one / np.sqrt(d)      # 均值方向 unit vector (d,1)
    # --- 构造 U_s：d x (d-1) 矩阵，列为与 u0 正交的正交基 ---
    R = rng.normal(size=(d, d-1))
    R = R - u0 @ (u0.T @ R)    # 把列投影到与 u0 正交
    # QR 得到正交基（d x d, 取前 d-1 列）
    Q_full, _ = np.linalg.qr(R, mode='reduced')  # returns d x (d-1)
    Us = Q_full[:, :d-1]  # d x (d-1), orthonormal columns spanning S
    # --- 在 (d-1)-维上采样 Haar 正交矩阵 Q_small ---
    G = rng.normal(size=(d-1, d-1))
    Qs, Rg = np.linalg.qr(G)
    # 调整符号，使得分布是 Haar (对角符号修正)
    D = np.sign(np.diag(Rg))
    D[D==0] = 1.0
    Q_small = Qs * D
    # --- embed Q_small into original space: Q = Us @ Q_small @ Us.T ---
    Q = Us @ (Q_small @ Us.T)
    # 投影 J, P
    J = (one @ one.T) / d
    P = np.eye(d) - J
    # A
    A = a * J + Q   # note QJ = 0, QP = Q, so this equals aJ + QP
    return {
        "A": A, "J": J, "P": P, "Us": Us, "Q_small": Q_small, "Q": Q, "a": a
    }


# 在文件顶部加入
import torch

# TransformerClient.__init__ 的修改版
class TransformerClient:
    def __init__(self, device=None):
        # 选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载参数（只剩 embedding 相关）
        data = np.load('gpt2_params/params.npz')

        self.n_layer = int(data['n_layer'][0])

        # 将 embedding 从 numpy 转为 torch tensor（放到 device）
        # 注意转换 dtype 为 float32，索引用 long
        self.wte = torch.from_numpy(data['wte'].astype(np.float32)).to(self.device)
        self.wpe = torch.from_numpy(data['wpe'].astype(np.float32)).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained('gpt2_params/tokenizer')

        self.hidden_size = 768

    def perform_embedding(self, input_text):
        # tokenizer 返回 numpy input_ids
        input_ids_np = self.tokenizer(input_text, return_tensors="np")["input_ids"]
        # 转为 torch long tensor 并放到 device
        input_ids = torch.from_numpy(input_ids_np).long().to(self.device)
        seq_len = input_ids.shape[1]

        # 进行 embedding（使用 torch 的索引）
        # self.wte shape: [vocab_size, hidden], input_ids indexes first dim
        hidden = self.wte[input_ids] + self.wpe[torch.arange(seq_len, device=self.device)]
        # hidden shape: [batch, seq_len, hidden_size], dtype float32 on device
        return hidden  # 返回 torch.Tensor（在 device 上）


def perform_inference(client, stub, input_text, collect_times=True):
    all_times = {} if collect_times else None

    # --- embedding (on GPU if available) ---
    start_time = time.time() if collect_times else None
    hidden = client.perform_embedding(input_text)  # torch.Tensor on device
    # hidden: shape [batch, seq_len, hidden]
    state = None  # we'll build state after obfuscation

    if collect_times:
        # 若在 GPU 上，先同步再记录时间
        if hidden.device.type == 'cuda':
            torch.cuda.synchronize()
        all_times['embedding'] = (time.time() - start_time) * 1000

    # 构造 A（你的 sample_A_constructive 目前返回 numpy）
    A_np = sample_A_constructive(client.hidden_size, a=1.8)['A']  # numpy (d,d)
    # 将 A 转为 torch 并放到同一 device
    A_t = torch.from_numpy(A_np.astype(np.float32)).to(client.device)

    # 做 obfuscation 在 GPU 上
    start_time = time.time() if collect_times else None
    if client.device.type == 'cuda':
        torch.cuda.synchronize()

    # hidden shape: [batch, seq_len, hidden], A_t: [hidden, hidden]
    # 我们需要进行最后一个维度的矩阵乘法。torch.matmul 支持 batched matmul:
    # result shape: [batch, seq_len, hidden]
    A_obf = torch.matmul(hidden, A_t)
    A_deobf = torch.matmul(A_obf, A_t)

    # 同步并计时
    if client.device.type == 'cuda':
        torch.cuda.synchronize()
    if collect_times:
        all_times['obf'] = (time.time() - start_time) * 1000

    # 把要发送的 state 从 torch 转回 numpy（CPU）供现有 np_to_state 使用
    state = {'input': A_deobf.detach().cpu().numpy()}

    # 唯一一次通信: 发送 input 到 REE 执行所有 blocks + final LN + logits
    if collect_times:
        req = demo_pb2.TransformerRequest(op_id=1000, state=demo_pb2.State(items=np_to_state(state)))
    else:
        req = demo_pb2.TransformerRequest(op_id=999, state=demo_pb2.State(items=np_to_state(state)))

    # gRPC 调用
    start = time.time() if collect_times else None
    resp = stub.Process(req)
    end = time.time() if collect_times else None

    # 将服务端返回的 state 转为 numpy（你已有函数）
    state = state_to_np(resp.state)
    logits = state['logits']
    if collect_times:
        all_times['server'] = (end - start) * 1000

    # 后续 decode 保持原样（使用 numpy）
    start_time = time.time() if collect_times else None
    next_token_id = int(np.argmax(logits[0, -1, :]))
    end_time = time.time() if collect_times else None

    if collect_times:
        print(f"Next token: '{client.tokenizer.decode(next_token_id)}'")
        all_times['decode'] = (end_time - start_time) * 1000

        total_compute = all_times['embedding'] + all_times['decode']
        grpc_time = all_times['server']

        # Print table
        print("\nTime Table:")
        print(f"{'Operation':<20}{'Time (ms)':<10}")
        print("-" * 30)
        print(f"{'embedding':<20}{all_times['embedding']:<10.2f}")
        print(f"{'decode':<20}{all_times['decode']:<10.2f}")
        print(f"{'obf':<20}{all_times['obf']:<10.2f}")
        print(f"{'Total Compute':<20}{total_compute:<10.2f}")
        print(f"{'gRPC Call':<20}{grpc_time:<10.2f}")

    return logits, next_token_id


import argparse
import random

def make_random_token_string(seq_len: int) -> str:
    # 简单的 token（词）列表，可再扩充
    vocab = [
        "the", "of", "and", "to", "a", "in", "that", "is", "for", "on",
        "with", "as", "this", "by", "are", "was", "from", "at", "it", "an",
        "model", "data", "token", "value", "input", "output", "layer",
        "random", "matrix", "state", "hidden", "sequence", "test",
        "generate", "sample", "embedding", "decode", "encode", "compute"
    ]
    
    return " ".join(random.choice(vocab) for _ in range(seq_len))

def run(input_len=5, run_times=1):
    host = 'localhost:50052'
    NNN = 10485760 * 4
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    channel = grpc.insecure_channel(host, options=options)
    stub = demo_pb2_grpc.TransformerServiceStub(channel)
    
    client = TransformerClient()
    
    print(f"Connecting to {host}. Running with SGX Memory Optimization (Lazy Loading)...")
    
    # Warmup - 不开启 Profiler
    print("--- Starting Warmup ---")
    _, _ = perform_inference(client, stub, "Warmup input", collect_times=False)
    print("--- Warmup Finished ---")
    
    input_text = make_random_token_string(input_len)
    for run_idx in range(run_times):
        #input_text = "The capital of France is"
        _, next_token_id = perform_inference(client, stub, input_text, collect_times=True)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer gRPC Client")
    parser.add_argument('--run-times', type=int, default=5,
                        help='Number of inference iterations')
    parser.add_argument('--input-len', type=int, default=10,
                        help='Approximate input text length (words)')
    args = parser.parse_args()
    run(run_times=args.run_times, input_len=args.input_len)