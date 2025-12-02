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


class TransformerClient:
    def __init__(self):
        # 加载参数（只剩 embedding 相关）
        data = np.load('bert_params/params.npz')
        
        self.n_layer = int(data['n_layer'][0])
        
        self.wte = data['wte']
        self.wpe = data['wpe']
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert_params/tokenizer')
        
        self.hidden_size = 768
        
        # 无需 warmup，因为 TEE 不再做 LN

    def perform_embedding(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="np")["input_ids"]
        seq_len = input_ids.shape[1]
        hidden = self.wte[input_ids] + self.wpe[np.arange(seq_len)]
        return hidden.astype(np.float32)

def perform_inference(client, stub, input_text, collect_times=True):

    all_times = {} if collect_times else None

    start_time = time.time() if collect_times else None
    
    # TEE: Embedding
    hidden = client.perform_embedding(input_text)
    state = {'input': hidden}
    
    end_time = time.time() if collect_times else None
    if collect_times:
        all_times['embedding'] = (end_time-start_time)*1000
    
    A = sample_A_constructive(client.hidden_size, a=1.8)['A']
    #print(f"shape of hidden: {hidden.shape} A: {A.shape}")
    start_time = time.time() if collect_times else None
    A_obf = hidden@A
    A_deobf = A_obf@A
    end_time = time.time() if collect_times else None
    if collect_times:
        all_times['obf'] = (end_time - start_time) * 1000
    
    # 唯一一次通信: 发送 input 到 REE 执行所有 blocks + final LN + logits
    if collect_times:
        req = demo_pb2.TransformerRequest(op_id=1000, state=demo_pb2.State(items=np_to_state(state)))
    else:
        req = demo_pb2.TransformerRequest(op_id=999, state=demo_pb2.State(items=np_to_state(state)))

    start = time.time() if collect_times else None
    resp = stub.Process(req)
    end = time.time() if collect_times else None
    
    state = state_to_np(resp.state)
    logits = state['logits']
    if collect_times:
        all_times['server'] = (end - start) * 1000

    start_time = time.time() if collect_times else None
    # TEE: 采样生成 token（这里用 argmax 作为示例）
    next_token_id = int(np.argmax(logits[0, -1, :]))
    end_time = time.time() if collect_times else None
    

    if collect_times:    
        print(f"Next token: '{client.tokenizer.decode(next_token_id)}'")
        all_times['decode'] = (end_time- start_time) * 1000

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