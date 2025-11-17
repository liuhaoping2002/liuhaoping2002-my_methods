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

class TransformerClient:
    def __init__(self):
        # 加载参数（只剩 embedding 相关）
        data = np.load('gpt2_params/params.npz')
        
        self.n_layer = int(data['n_layer'][0])
        
        self.wte = data['wte']
        self.wpe = data['wpe']
        
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2_params/tokenizer')
        
        self.hidden_size = 768
        
        # 无需 warmup，因为 TEE 不再做 LN

    def perform_embedding(self, input_text):
        input_ids = self.tokenizer(input_text, return_tensors="np")["input_ids"]
        seq_len = input_ids.shape[1]
        hidden = self.wte[input_ids] + self.wpe[np.arange(seq_len)]
        return hidden.astype(np.float32)

def perform_inference(client, stub, input_text, collect_times=True):
    start_time = time.time() if collect_times else None
    
    # TEE: Embedding
    hidden = client.perform_embedding(input_text)
    state = {'input': hidden}
    
    all_times = {} if collect_times else None
    
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
        all_times[1000] = ('server_all', (end - start) * 1000)

    # TEE: 采样生成 token（这里用 argmax 作为示例）
    next_token_id = int(np.argmax(logits[0, -1, :]))

    if collect_times:
        print(f"Logits shape: {logits.shape}")
        print(f"Next token: '{client.tokenizer.decode(next_token_id)}'")
        end_time = time.time()
        print(f"Total time cost: {(end_time - start_time) * 1000:.3f} ms")

        with open('time_client.log', 'w') as f:
            print(f"{'op_id':>6} | {'Executor':>8} | {'Time (ms)':>10}", file=f)
            print("-" * 30, file=f)
            for op_id in sorted(all_times.keys()):
                executor, time_ms = all_times[op_id]
                print(f"{op_id:>6} | {executor:>8} | {time_ms:>10.2f}", file=f)

    return logits, next_token_id

def run():
    time_count = time.time()
    NNN = 10485760 * 4
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    channel = grpc.insecure_channel('localhost:50051', options=options)
    stub = demo_pb2_grpc.TransformerServiceStub(channel)
    
    time_count = time_cost("channel establish", time_count)
    client = TransformerClient()
    
    # Warmup
    warmup_runs = 4
    print(f"Performing {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        _, _ = perform_inference(client, stub, "Warmup input", collect_times=False)
    print("Warmup completed.")
    
    # 实际运行
    input_text = "The capital of France is"
    time_count = time_cost("Tokenizer", time_count)
    _, next_token_id = perform_inference(client, stub, input_text, collect_times=True)

if __name__ == '__main__':
    run()