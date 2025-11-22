import grpc
import numpy as np
import torch
import io
import time
import sys
import argparse
from concurrent import futures
import threading
import math 
# from collections import defaultdict # 避免额外导入，使用简单字典结构

# 引入生成的 gRPC 代码
import demo_pb2
import demo_pb2_grpc

# --- 基础序列化工具 (保持不变) ---
def tensor_to_np(t: demo_pb2.Tensor) -> np.ndarray:
    buf = io.BytesIO(t.data)
    arr = np.load(buf)
    return arr.copy()

def np_to_tensor(arr: np.ndarray) -> demo_pb2.Tensor:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return demo_pb2.Tensor(
        data=buf.getvalue(),
        shape=list(arr.shape),
        dtype=str(arr.dtype)
    )

def state_to_np(state_pb):
    return {k: tensor_to_np(v) for k, v in state_pb.items.items()}

def np_to_state(state_np):
    return {k: np_to_tensor(v) for k, v in state_np.items()}

# --- 计时辅助工具 ---
local_storage = threading.local()

def get_op_name(step_id):
    """辅助函数：根据 step_id 返回可读的操作名"""
    if step_id >= 1202: return "Logits"
    # step_id % 100
    local = step_id % 100
    if local == 2: return "QKV + Scores"
    if local == 5: return "Attn Output"
    if local == 9: return "MLP Up"
    if local == 11: return "MLP Down"
    return f"Op_{step_id}"

class TransformerService(demo_pb2_grpc.TransformerServiceServicer):
    def __init__(self, device_choice: str = "cpu"):
        # --- 初始化逻辑 (保持不变) ---
        use_cuda = False
        device = None
        if device_choice == "cuda":
            if torch is None:
                print("警告：未安装 torch，无法使用 cuda，回退到 cpu。", file=sys.stderr)
            else:
                if torch.cuda.is_available():
                    use_cuda = True
                    device = torch.device("cuda")
                else:
                    print("警告：torch 找到但没有可用 CUDA 设备，回退到 cpu。", file=sys.stderr)
        
        self.use_cuda = use_cuda
        self.device = device

        try:
            data = np.load('gpt2_server_params.npz')
        except FileNotFoundError:
            raise FileNotFoundError("gpt2_server_params.npz not found. Run download_gpt2_params.py first.")

        self.n_layer = int(data['n_layer'][0])
        self.d_model = data['c_attn_w'].shape[-1] // 3
        self.h = 12
        self.d_k = self.d_model // self.h

        # 加载权重 (代码保持不变，略去冗长部分)
        c_attn_w_np = [data['c_attn_w'][i] for i in range(self.n_layer)]
        c_attn_b_np = [data['c_attn_b'][i] for i in range(self.n_layer)]
        c_proj_w_np = [data['c_proj_w'][i] for i in range(self.n_layer)]
        c_proj_b_np = [data['c_proj_b'][i] for i in range(self.n_layer)]
        mlp_c_fc_w_np = [data['mlp_c_fc_w'][i] for i in range(self.n_layer)]
        mlp_c_fc_b_np = [data['mlp_c_fc_b'][i] for i in range(self.n_layer)]
        mlp_c_proj_w_np = [data['mlp_c_proj_w'][i] for i in range(self.n_layer)]
        mlp_c_proj_b_np = [data['mlp_c_proj_b'][i] for i in range(self.n_layer)]
        lm_head_w_np = data['lm_head_w']

        if self.use_cuda:
            # 权重转移到 device
            self.c_attn_w = [torch.from_numpy(w).to(self.device) for w in c_attn_w_np]
            self.c_attn_b = [torch.from_numpy(b).to(self.device) for b in c_attn_b_np]
            self.c_proj_w = [torch.from_numpy(w).to(self.device) for w in c_proj_w_np]
            self.c_proj_b = [torch.from_numpy(b).to(self.device) for b in c_proj_b_np]
            self.mlp_c_fc_w = [torch.from_numpy(w).to(self.device) for w in mlp_c_fc_w_np]
            self.mlp_c_fc_b = [torch.from_numpy(b).to(self.device) for b in mlp_c_fc_b_np]
            self.mlp_c_proj_w = [torch.from_numpy(w).to(self.device) for w in mlp_c_proj_w_np]
            self.mlp_c_proj_b = [torch.from_numpy(b).to(self.device) for b in mlp_c_proj_b_np]
            self.lm_head_w = torch.from_numpy(lm_head_w_np).to(self.device)
        else:
            # 权重保持 numpy
            self.c_attn_w = c_attn_w_np
            self.c_attn_b = c_attn_b_np
            self.c_proj_w = c_proj_w_np
            self.c_proj_b = c_proj_b_np
            self.mlp_c_fc_w = mlp_c_fc_w_np
            self.mlp_c_fc_b = mlp_c_fc_b_np
            self.mlp_c_proj_w = mlp_c_proj_w_np
            self.mlp_c_proj_b = mlp_c_proj_b_np
            self.lm_head_w = lm_head_w_np

        print("Device chosen:", "cuda" if self.use_cuda else "cpu")
        print("Number of layers:", self.n_layer)

    def _to_torch_state(self, state_np: dict):
        # ... (保持不变) ...
        if not self.use_cuda: return state_np
        torch_state = {}
        for k, v in state_np.items():
            if isinstance(v, np.ndarray):
                torch_state[k] = torch.from_numpy(v).to(self.device)
            else:
                arr = np.asarray(v)
                if np.issubdtype(arr.dtype, np.floating): arr = arr.astype(np.float32, copy=False)
                torch_state[k] = torch.from_numpy(arr).to(self.device)
        return torch_state

    def _to_numpy_state(self, state_mixed: dict):
        # ... (保持不变) ...
        out = {}
        for k, v in state_mixed.items():
            if self.use_cuda and isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                out[k] = v
            else:
                out[k] = np.asarray(v)
        return out
    
    def _sync(self):
        """同步 CUDA 流，确保计时准确"""
        if self.use_cuda:
            torch.cuda.synchronize()

    def Process(self, request, context):
        # [统计] 服务端总计时开始
        t_server_start = time.perf_counter()
        
        op_id = request.op_id
        
        # 线程局部存储初始化
        if not hasattr(local_storage, 'all_times'):
            # 结构: {key: {'compute': X, 'total': Y}}
            local_storage.all_times = {}

        # 数据反序列化
        state = state_to_np(request.state)
        
        if self.use_cuda:
             context_manager = torch.no_grad()
        else:
             import contextlib
             context_manager = contextlib.nullcontext()

        # [统计] 纯计算时间累加器 (针对当前 RPC 请求)
        pure_compute_time = 0.0

        with context_manager:
            i = op_id
            current_layer = i // 100
            
            s = self._to_torch_state(state)
            response_state = {}

            while True:
                local_i = i % 100
                
                if i >= 1202: # Logits
                    # [计时] Logits Matmul
                    t0 = time.perf_counter()
                    if self.use_cuda:
                        logits = torch.matmul(s['ln_final'], self.lm_head_w)
                        self._sync()
                    else:
                        logits = np.dot(s['ln_final'], self.lm_head_w)
                    t1 = time.perf_counter()
                    pure_compute_time += (t1 - t0)
                    
                    response_state['logits'] = logits
                    i = 9999
                    break

                if local_i == 2:  # QKV
                    # [计时] QKV Matmul
                    t0 = time.perf_counter()
                    if self.use_cuda:
                        inp = s['ln1']
                        w, b = self.c_attn_w[current_layer], self.c_attn_b[current_layer]
                        proj = torch.matmul(inp, w) + b[None, None, :]
                        self._sync()
                    else:
                        inp = s['ln1']
                        w, b = self.c_attn_w[current_layer], self.c_attn_b[current_layer]
                        proj = np.dot(inp, w) + b[None, None, :]
                    t1 = time.perf_counter()
                    pure_compute_time += (t1 - t0)

                    # Reshape (不计入纯计算)
                    B, S, _ = inp.shape
                    if self.use_cuda:
                        Q = proj[:, :, :self.d_model].reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                        K = proj[:, :, self.d_model:2*self.d_model].reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                        V = proj[:, :, 2*self.d_model:].reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                    else:
                        Q = proj[:, :, :self.d_model].reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                        K = proj[:, :, self.d_model:2*self.d_model].reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                        V = proj[:, :, 2*self.d_model:].reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                    
                    s['Q'], s['K'], s['V'] = Q, K, V
                    i += 1

                elif local_i == 3:  # Scores + Mask
                    # [计时] Attention Scores
                    t0 = time.perf_counter()
                    if self.use_cuda:
                        scores = torch.matmul(s['Q'], s['K'].transpose(-2, -1)) / math.sqrt(self.d_k)
                        S_q = s['Q'].shape[2]
                        mask = torch.triu(torch.ones((S_q, S_q), device=self.device, dtype=scores.dtype), diagonal=1) * -1e9
                        scores = scores + mask[None, None, :, :]
                        self._sync()
                    else:
                        scores = np.matmul(s['Q'], s['K'].transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
                        S_q = s['Q'].shape[2]
                        mask = np.triu(np.ones((S_q, S_q), dtype=scores.dtype), k=1) * -1e9
                        scores += mask[None, None, :, :]
                    t1 = time.perf_counter()
                    pure_compute_time += (t1 - t0)

                    s['scores'] = scores
                    response_state['scores'] = scores
                    response_state['V'] = s['V']
                    i += 1
                    # Server 在此处完成了一次完整的请求处理，准备返回

                elif local_i == 5: # Attn @ V
                    # [计时] Attn * V
                    t0 = time.perf_counter()
                    if self.use_cuda:
                        s['aout'] = torch.matmul(s['attn'], s['V'])
                        self._sync()
                    else:
                        s['aout'] = np.matmul(s['attn'], s['V'])
                    t1 = time.perf_counter()
                    pure_compute_time += (t1 - t0)
                    
                    i += 1

                elif local_i == 6: # Attn Output Proj
                    # [计时] Attn Output Projection
                    t0 = time.perf_counter()
                    if self.use_cuda:
                        B, H, S_q, d_v = s['aout'].shape
                        aout = s['aout'].permute(0, 2, 1, 3).reshape(B, S_q, self.d_model)
                        w, b = self.c_proj_w[current_layer], self.c_proj_b[current_layer]
                        s['attn_out'] = torch.matmul(aout, w) + b[None, None, :]
                        self._sync()
                    else:
                        B, H, S_q, d_v = s['aout'].shape
                        aout = s['aout'].transpose(0, 2, 1, 3).reshape(B, S_q, self.d_model)
                        w, b = self.c_proj_w[current_layer], self.c_proj_b[current_layer]
                        s['attn_out'] = np.dot(aout, w) + b[None, None, :]
                    t1 = time.perf_counter()
                    pure_compute_time += (t1 - t0)

                    response_state['attn_out'] = s['attn_out']
                    i += 1

                elif local_i == 9: # MLP Up
                    # [计时] MLP Up
                    t0 = time.perf_counter()
                    if self.use_cuda:
                        w, b = self.mlp_c_fc_w[current_layer], self.mlp_c_fc_b[current_layer]
                        s['ff1'] = torch.matmul(s['ln2'], w) + b[None, None, :]
                        self._sync()
                    else:
                        w, b = self.mlp_c_fc_w[current_layer], self.mlp_c_fc_b[current_layer]
                        s['ff1'] = np.dot(s['ln2'], w) + b[None, None, :]
                    t1 = time.perf_counter()
                    pure_compute_time += (t1 - t0)

                    response_state['ff1'] = s['ff1']
                    i += 1

                elif local_i == 11: # MLP Down
                    # [计时] MLP Down
                    t0 = time.perf_counter()
                    if self.use_cuda:
                        w, b = self.mlp_c_proj_w[current_layer], self.mlp_c_proj_b[current_layer]
                        s['ff2'] = torch.matmul(s['gelu'], w) + b[None, None, :]
                        self._sync()
                    else:
                        w, b = self.mlp_c_proj_w[current_layer], self.mlp_c_proj_b[current_layer]
                        s['ff2'] = np.dot(s['gelu'], w) + b[None, None, :]
                    t1 = time.perf_counter()
                    pure_compute_time += (t1 - t0)

                    response_state['ff2'] = s['ff2']
                    i += 1
                
                else:
                    break
        
        # 转换回 numpy
        out_state_np = self._to_numpy_state(response_state)
        
        # [统计] 服务端总计时结束

        pure_compute_ms = pure_compute_time * 1000
        
        # 确定统计键
        agg_key = f"layer_{op_id // 100}" if op_id < 1202 else "final_logits"
        
        # 累积数据
        if agg_key not in local_storage.all_times:
            local_storage.all_times[agg_key] = {'compute': 0.0, 'total': 0.0}
        
        local_storage.all_times[agg_key]['compute'] += pure_compute_ms

        t_server_end = time.perf_counter()
        server_total_ms = (t_server_end - t_server_start) * 1000

        local_storage.all_times[agg_key]['total'] += server_total_ms

        # 打印当前 Step 的性能日志 (请求级别)
        op_name = get_op_name(op_id)
        print(f"[Req Layer {op_id // 100} - {op_name:^15}] Server Total: {server_total_ms:.3f} ms | Pure Compute: {pure_compute_ms:.3f} ms")


        if i == 9999:
            # --- 完整推理聚合报告 ---
            
            # 计算 Grand Totals
            grand_total_compute = sum(d['compute'] for d in local_storage.all_times.values())
            grand_total_server = sum(d['total'] for d in local_storage.all_times.values())

            print("\n" + "="*70)
            print(f"{'SERVER AGGREGATED INFERENCE REPORT':^70}")
            print("="*70)
            
            # 分层详细报告
            print(f"{'Layer/Stage':<15} | {'Pure Compute (ms)':>20} | {'Total Server Call (ms)':>24}")
            print("-" * 70)
            
            # 排序并打印
            for key in sorted(local_storage.all_times.keys()):
                if key == 'final_logits':
                    layer_label = "FINAL LOGITS"
                else:
                    layer_label = f"Layer {key.split('_')[-1]}"
                
                stats = local_storage.all_times[key]
                print(f"{layer_label:<15} | {stats['compute']:>20.3f} | {stats['total']:>24.3f}")

            # 总计摘要
            print("-" * 70)
            print(f"{'GRAND TOTAL':<15} | {grand_total_compute:>20.3f} | {grand_total_server:>24.3f}")
            print("=" * 70)
            
            # 清理状态并空两行
            local_storage.all_times = {} 
            print("\n") 
        
        return demo_pb2.TransformerResponse(op_id=i, state=demo_pb2.State(items=np_to_state(out_state_np)), status="ok")

def serve(device_choice: str):
    # ... (保持不变) ...
    NNN = 10485760 * 4
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    demo_pb2_grpc.add_TransformerServiceServicer_to_server(TransformerService(device_choice), server)
    server.add_insecure_port('[::]:50051')
    print(f"Server listening on :50051 using device: {device_choice}")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer gRPC server")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='device to run computations on: "cpu" or "cuda" (requires torch and CUDA).')
    args = parser.parse_args()
    
    # 检查必要的库
    if args.device == 'cuda' and torch is None:
        print("Error: PyTorch not installed. Cannot run in CUDA mode.", file=sys.stderr)
        sys.exit(1)
        
    serve(args.device)