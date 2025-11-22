# client_grpc.py (SGX Memory Optimized)
import grpc
import numpy as np
import io
import time
import gc  # 关键：引入垃圾回收
import demo_pb2
import demo_pb2_grpc
from transformers import AutoTokenizer
from scipy.special import softmax as sp_softmax

# ... (np_to_tensor, tensor_to_np, np_to_state, state_to_np, layer_norm 保持不变) ...
def time_cost(outputs, time_past):
    time_now = time.time()
    time_cost = time_now-time_past
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

def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    norm = (x - mean) / std
    return norm * weight + bias

def filter_state(state_np, keys_to_keep):
    return {k: np_to_tensor(state_np[k]) for k in keys_to_keep if k in state_np}


class TransformerClient:
    def __init__(self):
        # 优化点：初始化时不加载任何大参数
        # 仅加载 Tokenizer 和配置
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2_params/tokenizer')
        self.hidden_size = 768
        self.n_layer = 12 # 硬编码或从配置文件读取，不读取大npz
        self.params_dir = 'gpt2_params_split'
        
        # 预热不再加载大参数，只做假数据计算
        self._warmup_operations()

    def _warmup_operations(self):
        # 保持原逻辑，不依赖真实权重
        warmup_runs = 2 
        dummy_input_ln = np.random.randn(1, 1, self.hidden_size)
        dummy_gamma = np.ones(self.hidden_size)
        dummy_beta = np.zeros(self.hidden_size)
        dummy_scores_softmax = np.random.randn(1, 12, 1, 1)
        dummy_ff1_gelu = np.random.randn(1, 1, self.hidden_size * 4)
        
        for _ in range(warmup_runs):
            layer_norm(dummy_input_ln, dummy_gamma, dummy_beta)
            sp_softmax(dummy_scores_softmax, axis=-1)
            x = dummy_ff1_gelu
            x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    # 辅助函数：按需加载
    def load_embeddings(self):
        return np.load(f'{self.params_dir}/embeddings.npz')

    def load_layer_params(self, layer_idx):
        return np.load(f'{self.params_dir}/layer_{layer_idx}.npz')

    def load_final_params(self):
        return np.load(f'{self.params_dir}/final.npz')

    # 优化点：forward 现在接收 params 字典，不再从 self 读取
    def forward(self, i, state, params):
        current_layer = i // 100
        local_i = i % 100

        # Final LN 处理
        if i == 1201:
            # params 此时应包含 final_gamma/beta
            state['ln_final'] = layer_norm(state['input'], params['final_gamma'], params['final_beta'])
            return 1202, state

        while True:
            print(f"i:{i}   state:{list(state.keys())}")
            if local_i == 1:  # LN1
                gamma = params['ln1_gamma']
                beta = params['ln1_beta']
                state['ln1'] = layer_norm(state['input'], gamma, beta)
                i += 1
                local_i += 1
                break

            elif local_i == 4:  # softmax (无参数)
                state['attn'] = sp_softmax(state['scores'], axis=-1)
                i += 1
                local_i += 1
                break

            elif local_i == 7:
                state['attn_residual'] = state['input'] + state['attn_out']
                # 释放旧内存
                del state['input'], state['attn_out']
                i += 1
                local_i += 1
                
            elif local_i == 8:  # LN2
                gamma = params['ln2_gamma']
                beta = params['ln2_beta']
                state['ln2'] = layer_norm(state['attn_residual'], gamma, beta)
                i += 1
                local_i += 1

            elif local_i == 10:  # GELU (无参数)
                x = state['ff1']
                state['gelu'] = x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
                i += 1
                local_i += 1
                break
            
            elif local_i == 12:
                state['output'] = state['attn_residual'] + state['ff2']
                del state['attn_residual'], state['ff2']
                i += 1
                local_i += 1

            else:
                break
        return i, state

def perform_inference(client, stub, input_text, collect_times=True):
    start_time = time.time() if collect_times else None
    all_times = {} if collect_times else None
    
    # --- 阶段 1: Embedding (按需加载，用完即弃) ---
    input_ids = client.tokenizer(input_text, return_tensors="np")["input_ids"]
    seq_len = input_ids.shape[1]
    
    # Load Embeddings
    embed_params = client.load_embeddings()
    wte = embed_params['wte']
    wpe = embed_params['wpe']
    
    hidden = wte[input_ids] + wpe[np.arange(seq_len)]
    client.seq_len = hidden.shape[1]
    state = {'input': hidden.astype(np.float32)}
    
    # Explicitly delete embedding weights and force GC
    del wte, wpe, embed_params
    gc.collect() 
    # --- Embedding 结束 ---

    i = 1
    
    # --- 阶段 2: 层循环 (按层加载) ---
    for layer in range(client.n_layer):
        state = {'input': hidden}
        i = layer * 100 + 1
        
        layer_params = client.load_layer_params(layer)
        
        try:
            while True:
                old_i = i
                start = time.time() if collect_times else None
                i, state = client.forward(i, state, layer_params)
                
                # 如果完成了本层 (i 增加了且 mod 100 > 12)
                if i % 100 > 12:
                    hidden = state['output']
                    state = {'input': hidden}
                    break
                
                # --- 智能传输逻辑 ---
                # 根据当前的 op_id 决定发送哪些变量给 Server
                keys_to_send = []
                local_i = i % 100
                
                if local_i == 2: 
                    # Server 要算 QKV+Scores。只需要 ln1。
                    # input 留本地做残差，不发！
                    keys_to_send = ['ln1']
                    
                elif local_i == 5:
                    # Server 要算 Attn @ V + Proj。需要 attn 和 V。
                    keys_to_send = ['attn', 'V']
                    
                elif local_i == 9: # (注意 forward 里处理完 LN2 后 i 会变)
                    # Server 要算 MLP Up。需要 ln2。
                    keys_to_send = ['ln2']
                    
                elif local_i == 11:
                    # Server 要算 MLP Down。需要 gelu。
                    keys_to_send = ['gelu']
                
                # 构造精简的 request
                req_state = filter_state(state, keys_to_send)
                print(f"key to send {local_i}: {keys_to_send}")
                print(f"req_state {local_i}: {req_state}")
                req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=req_state))
                
                end = time.time() if collect_times else None
                if collect_times and i > old_i:
                     all_times[old_i] = ('client', (end - start) * 1000)

                # RPC Call
                start = time.time() if collect_times else None
                resp = stub.Process(req)
                end = time.time() if collect_times else None
                
                sent_i = i
                i = resp.op_id
                
                # Server 返回的数据 merge 到本地 state
                received_state = state_to_np(resp.state)
                state.update(received_state)
                
                # 内存清理：发送过的数据如果在后续步骤不再需要，可以清理
                # 注意：V 需要保留到 Step 4，但 Step 1 发送 ln1 后 ln1 可以删了
                if 'ln1' in state and local_i > 1: del state['ln1']
                if 'ln2' in state and local_i > 8: del state['ln2']
                if 'gelu' in state and local_i > 10: del state['gelu']

                if collect_times:
                    all_times[sent_i] = ('server', (end - start) * 1000)
        finally:
            del layer_params
            gc.collect()

    # Send Final Request to Server (Compute Logits)
    #req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=np_to_state(state)))
    req = demo_pb2.TransformerRequest(op_id=1202, state=demo_pb2.State(items=filter_state(state, ['ln_final'])))
    start = time.time() if collect_times else None
    resp = stub.Process(req)
    end = time.time() if collect_times else None
    if collect_times:
        all_times[i] = ('server', (end - start) * 1000)
        
    state = state_to_np(resp.state)
    logits = state['logits']

    if collect_times:
        next_token_id = int(np.argmax(logits[0, -1, :]))
        print(f"Next token: '{client.tokenizer.decode(next_token_id)}'")
        end_time = time.time()
        print(f"Total time cost: {(end_time - start_time) * 1000:.3f} ms")
        # (日志打印逻辑保持不变) ...

    return logits

def run():
    time_count = time.time()
    # ... (gRPC channel 逻辑保持不变) ...
    NNN = 10485760 * 4
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    channel = grpc.insecure_channel('localhost:50051', options=options)
    stub = demo_pb2_grpc.TransformerServiceStub(channel)
    
    client = TransformerClient()
    
    print("Running with SGX Memory Optimization (Lazy Loading)...")
    # Warmup (使用短文本)
    _ = perform_inference(client, stub, "Warmup", collect_times=False)
    
    input_text = "The capital of France is"
    _ = perform_inference(client, stub, input_text, collect_times=True)

if __name__ == '__main__':
    run()