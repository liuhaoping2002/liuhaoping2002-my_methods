# client_grpc.py (SGX Memory Optimized with Detailed Profiling and Aggregation)
import grpc
import numpy as np
import io
import time
import gc
import demo_pb2
import demo_pb2_grpc
from transformers import AutoTokenizer
from scipy.special import softmax as sp_softmax
from collections import defaultdict
import argparse

# --- 基础工具函数保持不变 ---
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

def gelu(x):
    import math
    return x * 0.5 * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))

# --- 统计工具类 (修改了 print_report 逻辑，加入了最终汇总) ---
class Profiler:
    def __init__(self):
        self.embedding_time = 0.0 # seconds
        # structure: {layer_idx: {'client': {op: time_ms}, 'server': {op: time_ms}}}
        self.layer_stats = defaultdict(lambda: {'client': defaultdict(float), 'server': defaultdict(float)})
        self.final_logits_time = 0.0 # seconds (Final Logits Server RPC Call)
        self.final_softmax_time = 0.0 # seconds (Final Softmax/Argmax Client Compute)
    
    def log_client(self, layer, op_name, duration_s):
        self.layer_stats[layer]['client'][op_name] += duration_s * 1000 # to ms

    def log_server(self, layer, op_name, duration_s):
        self.layer_stats[layer]['server'][op_name] += duration_s * 1000 # to ms
        
    def _get_aggregated_stats(self):
        total_client_compute_ms = self.embedding_time * 1000 + self.final_softmax_time * 1000
        total_server_rpc_ms = self.final_logits_time * 1000
        
        # Aggregated stats by layer/stage
        aggregated_data = {}
        
        # 1. Embeddings
        aggregated_data['initial'] = {
            'pure_compute': self.embedding_time * 1000,
            'rpc_total': 0.0,
            'client_total': self.embedding_time * 1000
        }
        
        # 2. Layers
        for layer in sorted(self.layer_stats.keys()):
            stats = self.layer_stats[layer]
            layer_client_time = sum(stats['client'].values())
            layer_server_time = sum(stats['server'].values())
            
            total_client_compute_ms += layer_client_time
            total_server_rpc_ms += layer_server_time
            
            aggregated_data[f'layer_{layer}'] = {
                'pure_compute': layer_client_time,
                'rpc_total': layer_server_time,
                'client_total': layer_client_time + layer_server_time
            }
            
        # 3. Final Logits/Softmax
        # Final Logits Server RPC Call (Pure RPC)
        aggregated_data['logits_call'] = {
            'pure_compute': 0.0,
            'rpc_total': self.final_logits_time * 1000,
            'client_total': self.final_logits_time * 1000
        }
        
        # Final Softmax/Argmax Client Compute (Pure Client)
        aggregated_data['final_output'] = {
            'pure_compute': self.final_softmax_time * 1000,
            'rpc_total': 0.0,
            'client_total': self.final_softmax_time * 1000
        }

        grand_total_client = total_client_compute_ms + total_server_rpc_ms
        
        return aggregated_data, total_client_compute_ms, total_server_rpc_ms, grand_total_client

    def print_report(self):
        aggregated_data, grand_pure_compute, grand_rpc_total, grand_client_total = self._get_aggregated_stats()
        
        print("\n" + "#"*85)
        print(f"{'CLIENT AGGREGATED INFERENCE REPORT (First Token)':^85}")
        print("#"*85)
        
        # 表头
        print(f"{'Layer/Stage':<20} | {'Client Pure Compute (ms)':>24} | {'RPC Total Call (ms)':>20} | {'Client Total (ms)':>15}")
        print("-" * 85)
        
        # 自定义排序键
        def sort_key(key):
            if key == 'initial': return 0
            if key == 'logits_call': return 130 
            if key == 'final_output': return 200
            try: return int(key.split('_')[-1]) + 1
            except ValueError: return 100 
        
        sorted_keys = sorted(aggregated_data.keys(), key=sort_key)
        
        for key in sorted_keys:
            stats = aggregated_data[key]
            
            if key == 'initial':
                layer_label = "Initial Embeddings"
            elif key == 'logits_call':
                layer_label = "Logits RPC Call"
            elif key == 'final_output':
                layer_label = "Final Softmax/Argmax"
            else:
                layer_label = f"Layer {key.split('_')[-1]}"

            print(f"{layer_label:<20} | {stats['pure_compute']:>24.3f} | {stats['rpc_total']:>20.3f} | {stats['client_total']:>15.3f}")

        # 总计摘要
        print("-" * 85)
        print(f"{'GRAND TOTAL':<20} | {grand_pure_compute:>24.3f} | {grand_rpc_total:>20.3f} | {grand_client_total:>15.3f}")
        print("#" * 85)
        
        # 清理状态并空两行
        # 注意：此处不清理 self.layer_stats，以便可以打印更详细的 breakdown
        print("\n")

# --- 算子名称映射辅助 ---
def get_op_name(op_id):
    # Client Ops
    if op_id == 1: return "LN1"
    if op_id == 4: return "Softmax_Attn"
    if op_id == 7: return "Residual_1"
    if op_id == 8: return "LN2"
    if op_id == 10: return "GELU"
    if op_id == 12: return "Residual_2"
    # Server Ops (based on the step BEFORE sending)
    if op_id == 2: return "QKV_Proj"
    if op_id == 5: return "Attn_Matmul" # Client发送attn, V后Server执行attn@V+Proj，Op 5对应Attn Output
    if op_id == 9: return "MLP_Up"
    if op_id == 11: return "MLP_Down"
    return f"Op_{op_id}"

class TransformerClient:
    def __init__(self):
        # 仅加载 Tokenizer 和配置
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2_params/tokenizer')
        self.hidden_size = 768
        self.n_layer = 12 
        self.params_dir = 'gpt2_params_split'
        self.seq_len = 0
        
        self._warmup_operations()

    def _warmup_operations(self):
        warmup_runs = 2 
        dummy_input_ln = np.random.randn(1, 1, self.hidden_size)
        dummy_gamma = np.ones(self.hidden_size)
        dummy_beta = np.zeros(self.hidden_size)
        dummy_scores_softmax = np.random.randn(1, 12, 1, 1)
        dummy_ff1_gelu = np.random.randn(1, 1, self.hidden_size * 4)
        
        for _ in range(warmup_runs):
            layer_norm(dummy_input_ln, dummy_gamma, dummy_beta)
            sp_softmax(dummy_scores_softmax, axis=-1)
            gelu(dummy_ff1_gelu) # 使用 gelu 函数

    # 辅助函数：按需加载
    def load_embeddings(self):
        # 注意：在实际运行中，需要确保 gpt2_params_split/embeddings.npz 存在
        # 否则此处会引发 FileNotFoundError
        return np.load(f'{self.params_dir}/embeddings.npz')

    def load_layer_params(self, layer_idx):
        return np.load(f'{self.params_dir}/layer_{layer_idx}.npz')

    def load_final_params(self):
        return np.load(f'{self.params_dir}/final.npz')

    def forward(self, i, state, params):
        current_layer = i // 100
        local_i = i % 100

        # Final LN 处理
        if i == 1201:
            state['ln_final'] = layer_norm(state['input'], params['final_gamma'], params['final_beta'])
            return 1202, state

        while True:
            if local_i == 1: # LN1
                gamma = params['ln1_gamma']
                beta = params['ln1_beta']
                state['ln1'] = layer_norm(state['input'], gamma, beta)
                i += 1; local_i += 1
                break

            elif local_i == 4: # softmax (无参数)
                state['attn'] = sp_softmax(state['scores'], axis=-1)
                i += 1; local_i += 1
                break

            elif local_i == 7: # Residual 1
                state['attn_residual'] = state['input'] + state['attn_out']
                # 释放旧内存
                del state['input'], state['attn_out']
                i += 1; local_i += 1
                
            elif local_i == 8:# LN2
                gamma = params['ln2_gamma']
                beta = params['ln2_beta']
                state['ln2'] = layer_norm(state['attn_residual'], gamma, beta)
                i += 1; local_i += 1

            elif local_i == 10: # GELU (无参数)
                state['gelu'] = gelu(state['ff1'])
                i += 1; local_i += 1
                break
            
            elif local_i == 12: # Residual 2
                state['output'] = state['attn_residual'] + state['ff2']
                del state['attn_residual'], state['ff2']
                i += 1; local_i += 1

            else:
                break
        return i, state

def perform_inference(client, stub, input_text, profiler=None):
    # --- 阶段 1: Embedding (按需加载，用完即弃) ---
    input_ids = client.tokenizer(input_text, return_tensors="np")["input_ids"]
    seq_len = input_ids.shape[1]
    
    # Load Embeddings (IO 不计入计算时间)
    embed_params = client.load_embeddings()
    wte = embed_params['wte']
    wpe = embed_params['wpe']
    
    # [统计] Embedding 计算
    t0 = time.perf_counter()
    hidden = wte[input_ids] + wpe[np.arange(seq_len)]
    t1 = time.perf_counter()
    if profiler: profiler.embedding_time = t1 - t0
    
    client.seq_len = hidden.shape[1]
    state = {'input': hidden.astype(np.float32)}
    
    del wte, wpe, embed_params
    gc.collect() 
    # --- Embedding 结束 ---

    # --- 阶段 2: 层循环 (按层加载) ---
    current_hidden = hidden # 当前层的输入，上一层的输出
    
    for layer in range(client.n_layer):
        state = {'input': current_hidden}
        i = layer * 100 + 1
        
        # Load Layer Parameters (IO 不计入时间)
        layer_params = client.load_layer_params(layer)
        
        try:
            while True:
                old_i = i
                
                # [统计] Client 本地计算 (TEE)
                t_start = time.perf_counter()
                i, state = client.forward(i, state, layer_params)
                t_end = time.perf_counter()
                
                if profiler and i > old_i:
                    # 只有当 i 增加时，才说明 Client 执行了算子
                    op_name = get_op_name(old_i % 100)
                    profiler.log_client(layer, op_name, t_end - t_start)

                # 完成本层
                if i % 100 > 12 and i < 1201:
                    current_hidden = state.pop('output')
                    break
                
                # --- Server 传输逻辑 ---
                if i % 100 in [2, 5, 9, 11] or i == 1202:
                    
                    keys_to_send = []
                    local_i = i % 100
                    
                    if local_i == 2: keys_to_send = ['ln1']
                    elif local_i == 5: keys_to_send = ['attn', 'V']
                    elif local_i == 9: keys_to_send = ['ln2']
                    elif local_i == 11: keys_to_send = ['gelu']
                    
                    req_state = filter_state(state, keys_to_send)
                    req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=req_state))
                    
                    # [统计] Server 远程调用 (REE)
                    t_start = time.perf_counter()
                    resp = stub.Process(req)
                    t_end = time.perf_counter()
                    
                    if profiler:
                        op_name = get_op_name(local_i)
                        profiler.log_server(layer, op_name, t_end - t_start)
                    
                    i = resp.op_id
                    received_state = state_to_np(resp.state)
                    state.update(received_state)

                    if 'ln1' in state and local_i > 1: del state['ln1']
                    if 'ln2' in state and local_i > 8: del state['ln2']
                    if 'gelu' in state and local_i > 10: del state['gelu']

        finally:
            del layer_params
            gc.collect()

    # --- 阶段 3: Final LN (Client) ---
    final_params = client.load_final_params()
    i = 1201
    state = {'input': current_hidden}
    
    t_start = time.perf_counter()
    i, state = client.forward(i, state, final_params) # 执行 1201: LN_Final
    t_end = time.perf_counter()
    if profiler: profiler.log_client(client.n_layer, "LN_Final", t_end - t_start)
    
    # --- 阶段 4: Final Logits (Server RPC) ---
    req = demo_pb2.TransformerRequest(op_id=1202, state=demo_pb2.State(items=filter_state(state, ['ln_final'])))
    
    t_start = time.perf_counter()
    resp = stub.Process(req)
    t_end = time.perf_counter()
    if profiler: profiler.final_logits_time = t_end - t_start # seconds
        
    state = state_to_np(resp.state)
    logits = state['logits']
    
    # --- 阶段 5: Final Softmax/Argmax (Client) ---
    t_start = time.perf_counter()
    next_token_id = int(np.argmax(logits[0, -1, :]))
    t_end = time.perf_counter()
    if profiler: profiler.final_softmax_time = t_end - t_start # seconds
    
    del final_params, logits, state
    gc.collect()

    return next_token_id

def run():
    parser = argparse.ArgumentParser(description="Transformer gRPC Client")
    parser.add_argument('--host', type=str, default='localhost:50051', help='Server host and port')
    args = parser.parse_args()
    
    NNN = 10485760 * 4
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    channel = grpc.insecure_channel(args.host, options=options)
    stub = demo_pb2_grpc.TransformerServiceStub(channel)
    
    client = TransformerClient()
    
    print(f"Connecting to {args.host}. Running with SGX Memory Optimization (Lazy Loading)...")
    
    # Warmup - 不开启 Profiler
    print("--- Starting Warmup ---")
    _ = perform_inference(client, stub, "Warmup", profiler=None)
    print("--- Warmup Finished ---")
    
    input_text = "The capital of France is"
    
    # 开启 Profiler
    my_profiler = Profiler()
    next_token_id = perform_inference(client, stub, input_text, profiler=my_profiler)
    
    print(f"\nNext token: '{client.tokenizer.decode(next_token_id)}'")
    
    # 打印详细报告和总计汇总
    my_profiler.print_report()

if __name__ == '__main__':
    run()