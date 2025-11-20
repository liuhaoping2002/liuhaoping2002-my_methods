# server_grpc.py (支持通过 --device 指定 cpu 或 cuda)
import grpc
import numpy as np
import io
from concurrent import futures
import demo_pb2
import demo_pb2_grpc
from transformers import AutoConfig, GPT2Model
import argparse
import math
import sys
from scipy.special import softmax as sp_softmax

# 可选的 GPU 支持
try:
    import torch
except Exception:
    torch = None

def tensor_to_np(t: demo_pb2.Tensor) -> np.ndarray:
    buf = io.BytesIO(t.data)
    arr = np.load(buf)
    return arr.copy()

def np_to_tensor(arr: np.ndarray) -> demo_pb2.Tensor:
    buf = io.BytesIO()
    # 保持原来的 allow_pickle=False
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

import time  # 新增导入 time
import threading
# 全局字典，用于收集时间（在 serve 函数中定义）
local_storage = threading.local()

def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    norm = (x - mean) / std
    return norm * weight + bias

class TransformerService(demo_pb2_grpc.TransformerServiceServicer):
    def __init__(self, device_choice: str = "cpu"):
        # 选择 device
        use_cuda = False
        device = None
        if device_choice == "cuda":
            if torch is None:
                print("警告：未安装 torch，无法使用 cuda，回退到 cpu。", file=sys.stderr)
                use_cuda = False
                device = None
            else:
                if torch.cuda.is_available():
                    use_cuda = True
                    device = torch.device("cuda")
                else:
                    print("警告：torch 找到但没有可用 CUDA 设备，回退到 cpu。", file=sys.stderr)
                    use_cuda = False
                    device = None
        else:
            use_cuda = False
            device = None

        self.use_cuda = use_cuda
        self.device = device
        
        # 从本地NPZ加载参数（替换原from_pretrained）
        try:
            data = np.load('gpt2_server_params.npz')
            data_c = np.load('gpt2_params/params.npz')
        except FileNotFoundError:
            raise FileNotFoundError("gpt2_server_params.npz not found. Run download_gpt2_params.py first.")

        self.n_layer = int(data['n_layer'][0])
        self.d_model = data['c_attn_w'].shape[-1] // 3  # 从c_attn_w推断 (d_model * 3)
        self.h = 12  # GPT-2 small的固定head数，可硬编码或从config推断
        self.d_k = self.d_model // self.h

        # 加载numpy权重
        c_attn_w_np = [data['c_attn_w'][i] for i in range(self.n_layer)]
        c_attn_b_np = [data['c_attn_b'][i] for i in range(self.n_layer)]

        c_proj_w_np = [data['c_proj_w'][i] for i in range(self.n_layer)]
        c_proj_b_np = [data['c_proj_b'][i] for i in range(self.n_layer)]

        mlp_c_fc_w_np = [data['mlp_c_fc_w'][i] for i in range(self.n_layer)]
        mlp_c_fc_b_np = [data['mlp_c_fc_b'][i] for i in range(self.n_layer)]

        mlp_c_proj_w_np = [data['mlp_c_proj_w'][i] for i in range(self.n_layer)]
        mlp_c_proj_b_np = [data['mlp_c_proj_b'][i] for i in range(self.n_layer)]

        lm_head_w_np = data['lm_head_w']
        


        # 根据是否使用 cuda，将权重转换为 torch tensors（并移动到 device）或保持 numpy
        if self.use_cuda:
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
        """如果使用 CUDA，则把 numpy state 转为 torch tensors 并搬到 device；否则直接返回 numpy dict"""
        if not self.use_cuda:
            return state_np
        torch_state = {}
        for k, v in state_np.items():
            if isinstance(v, np.ndarray):
                torch_state[k] = torch.from_numpy(v).to(self.device)
            else:
                arr = np.asarray(v)
                if np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float32, copy=False)
                torch_state[k] = torch.from_numpy(arr).to(self.device)
        return torch_state

    def _to_numpy_state(self, state_mixed: dict):
        """把可能为 torch tensors 的 state 转为 numpy（用于返回）"""
        out = {}
        for k, v in state_mixed.items():
            if self.use_cuda and isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                out[k] = v
            else:
                if torch is not None and isinstance(v, torch.Tensor):
                    out[k] = v.detach().cpu().numpy()
                else:
                    out[k] = np.asarray(v)
        return out

    def Process(self, request, context):
        if not hasattr(local_storage, 'all_times'):
            local_storage.all_times = {}
        
        op_id = request.op_id
        state = state_to_np(request.state)  # numpy dict incoming
        i = op_id
        current_layer = i // 100

        # 如果用 cuda，则把 state 一次性转为 torch tensors
        s = self._to_torch_state(state)

        while True:
            local_i = i % 100
            print(f"i={i}")

            if current_layer >= self.n_layer:  # final linear
                start = time.time()
                if local_i == 2:
                    # logits = ln_final @ lm_head_w
                    if self.use_cuda:
                        logits = torch.matmul(s['ln_final'], self.lm_head_w)
                        s['logits'] = logits
                    else:
                        s['logits'] = np.dot(s['ln_final'], self.lm_head_w)
                    end = time.time()  # 结束测量
                    time_ms = (end - start) * 1000
                    local_storage.all_times[i] = ('server', time_ms)  # 记录到当前 i
                    i = 9999
                break
            start = time.time()  # 每个操作前测量（除 final 外）
            if local_i == 2:  # QKV
                # inp = state['ln1']
                if self.use_cuda:
                    inp = s['ln1']  # torch tensor on device
                    w = self.c_attn_w[current_layer]
                    b = self.c_attn_b[current_layer]
                    proj = torch.matmul(inp, w) + b[None, None, :]
                    B, S, _ = inp.shape
                    Q = proj[:, :, :self.d_model]
                    K = proj[:, :, self.d_model:2*self.d_model]
                    V = proj[:, :, 2*self.d_model:]
                    # reshape and transpose: (B, S, h, d_k) -> (B, h, S, d_k)
                    Q = Q.reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                    K = K.reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                    V = V.reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                    s['Q'] = Q
                    s['K'] = K
                    s['V'] = V
                else:
                    inp = s['ln1']
                    w = self.c_attn_w[current_layer]
                    b = self.c_attn_b[current_layer]
                    proj = np.dot(inp, w) + b[None, None, :]
                    B, S, _ = inp.shape
                    Q = proj[:, :, :self.d_model]
                    K = proj[:, :, self.d_model:2*self.d_model]
                    V = proj[:, :, 2*self.d_model:]
                    Q = Q.reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                    K = K.reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                    V = V.reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                    s['Q'] = Q
                    s['K'] = K
                    s['V'] = V
                i += 1

            elif local_i == 3:  # scores + causal mask
                if self.use_cuda:
                    scores = torch.matmul(s['Q'], s['K'].transpose(-2, -1)) / math.sqrt(self.d_k)
                    S_q = s['Q'].shape[2]
                    # causal mask
                    mask = torch.triu(torch.ones((S_q, S_q), device=self.device, dtype=scores.dtype), diagonal=1) * -1e9
                    scores = scores + mask[None, None, :, :]
                    s['scores'] = scores
                else:
                    scores = np.matmul(s['Q'], s['K'].transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
                    S_q = s['Q'].shape[2]
                    mask = np.triu(np.ones((S_q, S_q), dtype=scores.dtype), k=1) * -1e9
                    scores += mask[None, None, :, :]
                    s['scores'] = scores
                i += 1
                

            elif local_i == 5:
                # state['aout'] = np.matmul(state['attn'], state['V'])
                if self.use_cuda:
                    s['aout'] = torch.matmul(s['attn'], s['V'])
                else:
                    s['aout'] = np.matmul(s['attn'], s['V'])
                i += 1


            elif local_i == 7:
                # state['attn_residual'] = state['input'] + state['attn_out']
                s['attn_residual'] = s['input'] + s['attn_out']
                i += 1
                

            elif local_i == 9:
                # w = self.mlp_c_fc_w[current_layer]
                # state['ff1'] = np.dot(state['ln2'], w) + b[None, None, :]
                if self.use_cuda:
                    w = self.mlp_c_fc_w[current_layer]
                    b = self.mlp_c_fc_b[current_layer]
                    s['ff1'] = torch.matmul(s['ln2'], w) + b[None, None, :]
                else:
                    w = self.mlp_c_fc_w[current_layer]
                    b = self.mlp_c_fc_b[current_layer]
                    s['ff1'] = np.dot(s['ln2'], w) + b[None, None, :]
                i += 1

            elif local_i == 11:
                if self.use_cuda:
                    w = self.mlp_c_proj_w[current_layer]
                    b = self.mlp_c_proj_b[current_layer]
                    s['ff2'] = torch.matmul(s['gelu'], w) + b[None, None, :]
                else:
                    w = self.mlp_c_proj_w[current_layer]
                    b = self.mlp_c_proj_b[current_layer]
                    s['ff2'] = np.dot(s['gelu'], w) + b[None, None, :]
                i += 1



            else:
                break
            end = time.time()  # 每个操作后测量（除 final 外）
            if 'start' in locals():
                time_ms = (end - start) * 1000
                local_storage.all_times[i - 1] = ('server', time_ms)  # 归属到刚完成的 i（增前）

        # 在返回之前，把 state（可能含 torch tensors）转为 numpy
        out_state_np = self._to_numpy_state(s)
        
        if i == 9999:
            #print("\n服务端本次输入执行时间统计:")
            with open('time_server.log', 'w') as f:
                print(f"{'op_id':>6} | {'Executor':>8} | {'Time (ms)':>10}", file=f)
                print("-" * 30, file=f)
                for op_id in sorted(local_storage.all_times.keys()):
                    executor, time_ms = local_storage.all_times[op_id]
                    print(f"{op_id:>6} | {executor:>8} | {time_ms:>10.2f}", file=f)
            # 重置 for 下一个输入
            local_storage.all_times = {}
        
        return demo_pb2.TransformerResponse(op_id=i, state=demo_pb2.State(items=np_to_state(out_state_np)), status="ok")

def serve(device_choice: str):
    global all_times
    all_times = {}  # 初始化全局字典
    NNN = 10485760 * 4  # 更大一点，logits 可能大
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    demo_pb2_grpc.add_TransformerServiceServicer_to_server(TransformerService(device_choice), server)
    server.add_insecure_port('[::]:50051')
    print("Server listening on :50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer gRPC server")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='device to run computations on: "cpu" or "cuda" (requires torch and CUDA).')
    args = parser.parse_args()
    serve(args.device)
