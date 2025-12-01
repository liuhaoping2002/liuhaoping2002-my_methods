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

# --- 新增：安全缩放工具 ---
SCALING_FACTOR = 1.88  # 缩放因子，可以是任意值，模拟混淆

def scale_state(state_np, mode='encrypt'):
    """
    mode='encrypt': 数据 * 因子 (离开安全区域或计算完成后)
    mode='decrypt': 数据 / 因子 (进入区域或计算开始前)
    """
    factor = SCALING_FACTOR if mode == 'encrypt' else (1.0 / SCALING_FACTOR)
    new_state = {} 
    for i in state_np:
        new_state[i] =  state_np[i] * factor
    return np_to_state(new_state)

# --- 统计工具类 (修改了 print_report 逻辑，加入了最终汇总) ---
class Profiler:
    def __init__(self):
        self.embedding_time = 0.0 
        # structure: {layer_idx: {'client': val, 'server': val, 'scaling': val}}
        # 使用 lambda 自动初始化字典，默认值为 0.0
        self.layer_stats = defaultdict(lambda: defaultdict(float))
        self.final_logits_rpc_time = 0.0 
        self.final_softmax_time = 0.0
        self.final_scaling_time = 0.0 # Final 阶段的 scaling 时间
    
    def log_client(self, layer, op_name, duration_s):
        # op_name 目前没用到细分展示，暂时累加到 'client'
        self.layer_stats[layer]['client'] += duration_s * 1000 

    def log_server(self, layer, op_name, duration_s):
        self.layer_stats[layer]['server'] += duration_s * 1000 
        
    def log_scaling(self, layer, duration_s):
        """新增：记录该层的缩放（加密/解密）耗时"""
        self.layer_stats[layer]['scaling'] += duration_s * 1000
        
    def print_report(self):
        # 准备汇总数据
        # Initial
        total_pure_compute = self.embedding_time * 1000
        total_rpc = 0.0
        total_scaling = 0.0
        
        # Containers for print loop
        aggregated_data = {}
        
        # 1. Initial
        aggregated_data['initial'] = {
            'compute': self.embedding_time * 1000, 'rpc': 0.0, 'scaling': 0.0
        }
        
        # 2. Layers
        for layer in sorted(self.layer_stats.keys()):
            stats = self.layer_stats[layer]
            c = stats['client']
            r = stats['server']
            s = stats['scaling']
            
            total_pure_compute += c
            total_rpc += r
            total_scaling += s
            
            aggregated_data[f'layer_{layer}'] = {'compute': c, 'rpc': r, 'scaling': s}

        # 3. Final Stage
        # Final Compute = Client Final LN (not tracked explicitly in struct but log_client called) + Final Softmax
        # Note: The code below handles Final LN logging via log_client(n_layer, ...) so it might be in layer_12 or separate
        # Let's check perform_inference: log_client(n_layer, "LN_Final") -> layer=12 (if 0-11)
        
        # Final Logits (RPC)
        total_rpc += self.final_logits_rpc_time * 1000
        
        # Final Scaling
        total_scaling += self.final_scaling_time * 1000
        
        # Final Softmax (Compute)
        total_pure_compute += self.final_softmax_time * 1000
        
        # Add explicit final entries for clarity in table
        aggregated_data['final_logits'] = {
            'compute': 0.0, 
            'rpc': self.final_logits_rpc_time * 1000, 
            'scaling': self.final_scaling_time * 1000 # Logits encrypt/decrypt
        }
        aggregated_data['final_output'] = {
            'compute': self.final_softmax_time * 1000, 
            'rpc': 0.0, 
            'scaling': 0.0
        }

        grand_total = total_pure_compute + total_rpc + total_scaling

        print("\n" + "#"*110)
        print(f"{'CLIENT AGGREGATED REPORT (Secure Mode with Scaling Stats)':^110}")
        print("#"*110)
        # 调整表头，增加 Scaling 列
        print(f"{'Layer/Stage':<20} | {'Pure Compute (ms)':>18} | {'Scaling (ms)':>18} | {'RPC Total (ms)':>18} | {'Total (ms)':>15}")
        print("-" * 110)
        
        def sort_key(key):
            if key == 'initial': return -1
            if key == 'final_logits': return 9998
            if key == 'final_output': return 9999
            try: return int(key.split('_')[-1])
            except: return 100
            
        for key in sorted(aggregated_data.keys(), key=sort_key):
            d = aggregated_data[key]
            row_total = d['compute'] + d['rpc'] + d['scaling']
            
            label = key
            if 'layer' in key: label = f"Layer {key.split('_')[-1]}"
            if key == 'initial': label = "Initial Embeddings"
            if key == 'final_logits': label = "Logits Stage"
            if key == 'final_output': label = "Final Softmax"
            
            print(f"{label:<20} | {d['compute']:>18.3f} | {d['scaling']:>18.3f} | {d['rpc']:>18.3f} | {row_total:>15.3f}")
            
        print("-" * 110)
        print(f"{'GRAND TOTAL':<20} | {total_pure_compute:>18.3f} | {total_scaling:>18.3f} | {total_rpc:>18.3f} | {grand_total:>15.3f}")
        print("#" * 110 + "\n")

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
        #self.tokenizer = AutoTokenizer.from_pretrained('vit-base-client')
        self.hidden_size = 768
        self.head_num = 12
        self.head_dim = int(self.hidden_size // self.head_num)
        self.n_layer = 12 
        self.params_dir = 'vit_params'
        self.seq_len = 197
        self.scale = 1.834
        
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
            #print(list(state.keys()))
            if local_i == 1: # LN1
                gamma = params['ln1_gamma']
                beta = params['ln1_beta']
                state['ln1'] = layer_norm(state['input'], gamma, beta)

                ##sca =   scale_state(state['ln1'], mode='encrypt')
                #sca =  scale_state(state['input'], mode='decrypt')

                i += 1; local_i += 1
                break

            elif local_i == 4: # softmax (无参数)
                state['attn'] = sp_softmax(state['scores'], axis=-1)

                #sca =  scale_state(state['attn'], mode='encrypt')
                #sca =  scale_state(state['scores'], mode='decrypt')

                i += 1; local_i += 1
                break

            elif local_i == 6: 
                #Client 执行 Attention Output Projection (W_O)
                w = np.random.rand(self.hidden_size, self.hidden_size).astype(np.float32)  # 模拟权重加载
                b = np.random.rand(self.hidden_size).astype(np.float32)  # 模拟偏置加载

                B, H, S_q, d_v = state['aout'].shape
                aout = state['aout'].transpose(0, 2, 1, 3).reshape(B, S_q, self.hidden_size)
                state['attn_out'] = np.dot(aout, w) + b[None, None, :]                

                #sca =  scale_state(state['aout'], mode='encrypt')

                # 内存清理：aout 很大，算完即弃
                del state['aout']
                
                i += 1
                local_i += 1

            #elif local_i == 7: # Residual 1
                state['attn_residual'] = state['input'] + state['attn_out']
                # 释放旧内存
                del state['input'], state['attn_out']
                i += 1; local_i += 1
                
            #elif local_i == 8:# LN2
                gamma = params['ln2_gamma']
                beta = params['ln2_beta']
                state['ln2'] = layer_norm(state['attn_residual'], gamma, beta)

                #sca =  scale_state(state['ln2'], mode='encrypt')

                i += 1; local_i += 1

            elif local_i == 10: # GELU (无参数)
                state['gelu'] = gelu(state['ff1'])

                #sca =  scale_state(state['gelu'], mode='encrypt')
                #sca =  scale_state(state['ff1'], mode='encrypt')

                i += 1; local_i += 1
                break
            
            elif local_i == 12: # Residual 2
                state['output'] = state['attn_residual'] + state['ff2']

                #sca =  scale_state(state['output'], mode='encrypt')
                #sca =  scale_state(state['ff2'], mode='decrypt')

                del state['attn_residual'], state['ff2']
                i += 1; local_i += 1

            else:
                break
        return i, state

from transformers import ViTImageProcessor
def perform_inference(client, stub, image_pil, profiler=None):
    # --- 阶段 1: Embedding (按需加载，用完即弃) ---
    processor = ViTImageProcessor.from_pretrained("./vit-base-client")
    inputs = processor(images=image_pil, return_tensors="pt")
    pixel_values = inputs.pixel_values  # [1,3,224,224]

    # 2. 只跑官方 ViT 的 embedding 层（获取 [1,197,768]）
    from transformers import ViTModel
    vit = ViTModel.from_pretrained("./vit-base-client")
    with torch.no_grad():
        embeddings = vit.embeddings(pixel_values)  # 包含 cls token + pos embed + patch proj
    hidden = embeddings.cpu().numpy().astype(np.float32)  # [1,197,768]
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

                    req_state_np = state_to_np(demo_pb2.State(items=req_state))
                    t_scale_start = time.perf_counter()
                    encrypted_state = scale_state(req_state_np, 'encrypt')
                    t_scale_end = time.perf_counter()
                    if profiler: profiler.log_scaling(layer, t_scale_end - t_scale_start)


                    req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=encrypted_state))
                    
                    # [统计] Server 远程调用 (REE)
                    t_start = time.perf_counter()
                    resp = stub.Process(req)
                    t_end = time.perf_counter()
                    
                    if profiler:
                        op_name = get_op_name(local_i)
                        profiler.log_server(layer, op_name, t_end - t_start)
                    
                    i = resp.op_id
                    received_state = state_to_np(resp.state)

                    t_scale_start = time.perf_counter()
                    decrypted_state = scale_state(received_state, 'decrypt')
                    t_scale_end = time.perf_counter()
                    if profiler: profiler.log_scaling(layer, t_scale_end - t_scale_start)

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
    req = demo_pb2.TransformerRequest(
        op_id=1202,
        state=demo_pb2.State(items=filter_state(state, ['ln_final']))
    )
    
    t_start = time.perf_counter()
    resp = stub.Process(req)
    t_end = time.perf_counter()
    if profiler: profiler.final_logits_rpc_time = t_end - t_start

    received = state_to_np(resp.state)
    final_hidden = received['final_hidden']  # [1,197,768]

    # 取出 CLS token（第0个）
    cls_output = final_hidden[:, 0, :]  # [1,768]

    return cls_output  # 返回特征向量


from PIL import Image
import torch

def run(input_len=5, run_times=1):
    host = 'localhost:50051'
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
    noise_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # 转成同类型 Image.Image
    noise_img = Image.fromarray(noise_np, mode='RGB')
    _ = perform_inference(client, stub, noise_img, profiler=None)
    print("--- Warmup Finished ---")
    
    #input_text = make_random_token_string(input_len)
    for run_idx in range(run_times):
        #input_text = "The capital of France is"
        
        img = Image.open("ff.jpg").convert("RGB")
        # 开启 Profiler
        my_profiler = Profiler()
        cls_feature = perform_inference(client, stub, img, profiler=my_profiler)

        classifier = torch.nn.Linear(768, 1000)
        # 你可以从 ViTModel.from_pretrained(..., output_attentions=False) 里取 classifier.weight
        logits = classifier(torch.from_numpy(cls_feature))
        pred_id = int(logits.argmax(-1).item())
        print(f"Predicted class: {pred_id}")
        #print(f"\nNext token: '{client.tokenizer.decode(next_token_id)}'")
        
        # 打印详细报告和总计汇总
        #print(f"\nRun {run_idx + 1} Next token: '{client.tokenizer.decode(next_token_id)}'")
        my_profiler.print_report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer gRPC Client")
    parser.add_argument('--run-times', type=int, default=5,
                        help='Number of inference iterations')
    parser.add_argument('--input-len', type=int, default=10,
                        help='Approximate input text length (words)')
    args = parser.parse_args()
    run(run_times=args.run_times, input_len=args.input_len)