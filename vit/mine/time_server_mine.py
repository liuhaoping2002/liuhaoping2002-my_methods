# server.py (REE)
import grpc
import numpy as np
import io
from concurrent import futures
import demo_pb2
import demo_pb2_grpc
import argparse
import math
import sys
import time
from scipy.special import softmax as sp_softmax
import threading


try:
    import torch
except Exception:
    torch = None

import demo_pb2  # Assuming this is the generated protobuf file
import demo_pb2_grpc  # Assuming this is the generated gRPC file
import grpc
from concurrent import futures
import argparse

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

local_storage = threading.local()

class TransformerService(demo_pb2_grpc.TransformerServiceServicer):
    def __init__(self, device_choice: str = "cpu"):
        # device 选择
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

        # 加载参数 (server + client 的所有参数)
        data = np.load('vit_server_params.npz')
        client_data = np.load('vit_params/params.npz')  # 加载 LN 和 final LN 参数

        self.n_layer = int(data['n_layer'][0])
        self.d_model = data['c_attn_w'].shape[-1] // 3
        self.h = 12
        self.d_k = self.d_model // self.h
        self.seq_len = 0

        # 线性层参数
        c_attn_w_np = [data['c_attn_w'][i] for i in range(self.n_layer)]
        c_attn_b_np = [data['c_attn_b'][i] for i in range(self.n_layer)]

        c_proj_w_np = [data['c_proj_w'][i] for i in range(self.n_layer)]
        c_proj_b_np = [data['c_proj_b'][i] for i in range(self.n_layer)]

        mlp_c_fc_w_np = [data['mlp_c_fc_w'][i] for i in range(self.n_layer)]
        mlp_c_fc_b_np = [data['mlp_c_fc_b'][i] for i in range(self.n_layer)]

        mlp_c_proj_w_np = [data['mlp_c_proj_w'][i] for i in range(self.n_layer)]
        mlp_c_proj_b_np = [data['mlp_c_proj_b'][i] for i in range(self.n_layer)]

        #lm_head_w_np = data['lm_head_w']

        # LN 参数（包括 final）
        self.ln1_gamma = [client_data['ln1_gamma'][i] for i in range(self.n_layer)]
        self.ln1_beta = [client_data['ln1_beta'][i] for i in range(self.n_layer)]
        self.ln2_gamma = [client_data['ln2_gamma'][i] for i in range(self.n_layer)]
        self.ln2_beta = [client_data['ln2_beta'][i] for i in range(self.n_layer)]
        self.final_gamma = client_data['final_gamma']
        self.final_beta = client_data['final_beta']

        # 转 torch 或保持 numpy
        if self.use_cuda:
            self.c_attn_w = [torch.from_numpy(w).to(self.device) for w in c_attn_w_np]
            self.c_attn_b = [torch.from_numpy(b).to(self.device) for b in c_attn_b_np]
            self.c_proj_w = [torch.from_numpy(w).to(self.device) for w in c_proj_w_np]
            self.c_proj_b = [torch.from_numpy(b).to(self.device) for b in c_proj_b_np]
            self.mlp_c_fc_w = [torch.from_numpy(w).to(self.device) for w in mlp_c_fc_w_np]
            self.mlp_c_fc_b = [torch.from_numpy(b).to(self.device) for b in mlp_c_fc_b_np]
            self.mlp_c_proj_w = [torch.from_numpy(w).to(self.device) for w in mlp_c_proj_w_np]
            self.mlp_c_proj_b = [torch.from_numpy(b).to(self.device) for b in mlp_c_proj_b_np]
            #self.lm_head_w = torch.from_numpy(lm_head_w_np).to(self.device)

            self.ln1_gamma = [torch.from_numpy(g).to(self.device) for g in self.ln1_gamma]
            self.ln1_beta = [torch.from_numpy(b).to(self.device) for b in self.ln1_beta]
            self.ln2_gamma = [torch.from_numpy(g).to(self.device) for g in self.ln2_gamma]
            self.ln2_beta = [torch.from_numpy(b).to(self.device) for b in self.ln2_beta]
            self.final_gamma = torch.from_numpy(client_data['final_gamma']).to(self.device)
            self.final_beta = torch.from_numpy(client_data['final_beta']).to(self.device)
        else:
            self.c_attn_w = c_attn_w_np
            self.c_attn_b = c_attn_b_np
            self.c_proj_w = c_proj_w_np
            self.c_proj_b = c_proj_b_np
            self.mlp_c_fc_w = mlp_c_fc_w_np
            self.mlp_c_fc_b = mlp_c_fc_b_np
            self.mlp_c_proj_w = mlp_c_proj_w_np
            self.mlp_c_proj_b = mlp_c_proj_b_np
            #self.lm_head_w = lm_head_w_np

            self.final_gamma = client_data['final_gamma']
            self.final_beta = client_data['final_beta']

        print("Device chosen:", "cuda" if self.use_cuda else "cpu")
        # ... (打印形状代码不变，可省略以简化)

    def _to_torch_state(self, state_np: dict):
        if not self.use_cuda:
            return state_np
        torch_state = {}
        for k, v in state_np.items():
            torch_state[k] = torch.from_numpy(np.asarray(v, dtype=np.float32)).to(self.device)
        return torch_state

    def _to_numpy_state(self, state_mixed: dict):
        out = {}
        for k, v in state_mixed.items():
            if self.use_cuda and isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().numpy()
            else:
                out[k] = np.asarray(v)
        return out

    def layer_norm(self, x, weight, bias, eps=1e-5):
        if self.use_cuda:
            mean = x.mean(dim=-1, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            std = torch.sqrt(var + eps)
            norm = (x - mean) / std
            self._sync()
            return norm * weight + bias
        else:
            mean = x.mean(axis=-1, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
            std = np.sqrt(var + eps)
            norm = (x - mean) / std
            return norm * weight + bias

    def gelu(self, x):
        if self.use_cuda:
            return x * 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        else:
            return x * 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def softmax(self, x, axis=-1):
        if self.use_cuda:
            return torch.softmax(x, dim=axis)
        else:
            return sp_softmax(x, axis=axis)
        
    def _sync(self):
        """同步 CUDA 流，确保计时准确"""
        if self.use_cuda:
            torch.cuda.synchronize()

    def full_forward_all(self, input_hidden, whether_warmup=False):
        """
        返回：
        x: 最终输出
        layer_time: dict, 每层的“纯计算”耗时（毫秒）
        计时策略：只记录主要数值计算（matmul/dot, softmax, gelu, layer_norm 等）。
        不记录：reshape/transpose/permute/astype/随机数或mask的生成等开销。
        """
        layer_time = {}
        # 将输入转换为内部状态（保持原逻辑）
        s = self._to_torch_state({'input': input_hidden})
        x = s['input']

        # 用于高精度计时
        perf = time.perf_counter

        for layer in range(self.n_layer):
            compute_time = 0.0  # 本层只计数“纯计算”，单位秒

            # ========== LN1 ==========
            # layer_norm 被视为模型计算的一部分 => 计时
            t0 = perf()
            ln1 = self.layer_norm(x, self.ln1_gamma[layer], self.ln1_beta[layer])
            t1 = perf(); compute_time += (t1 - t0)

            # ========== QKV projection ==========
            # proj 的矩阵乘/加被计时；reshape/transposes 放在计时外
            if self.use_cuda:
                t0 = perf()
                proj = torch.matmul(ln1, self.c_attn_w[layer]) + self.c_attn_b[layer][None, None, :]
                self._sync()
                t1 = perf(); compute_time += (t1 - t0)

                # 下面的 reshape / permute 不计时（只做形状调整）
                B, S, _ = ln1.shape
                Q = proj[:, :, :self.d_model].reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                K = proj[:, :, self.d_model:2*self.d_model].reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                V = proj[:, :, 2*self.d_model:].reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
            else:
                t0 = perf()
                proj = np.dot(ln1, self.c_attn_w[layer]) + self.c_attn_b[layer][None, None, :]
                t1 = perf(); compute_time += (t1 - t0)

                B, S, _ = ln1.shape
                # reshape/transpose 不计时
                Q = proj[:, :, :self.d_model].reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                K = proj[:, :, self.d_model:2*self.d_model].reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                V = proj[:, :, 2*self.d_model:].reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)

            # ========== scores = Q @ K^T / sqrt(d_k) 以及 mask add ==========
            # 计算 scores 的矩阵乘是重要计算 -> 计时
            if self.use_cuda:
                t0 = perf()
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                self._sync()
                t1 = perf(); compute_time += (t1 - t0)

                # mask 的生成（形状操作）不计时，但 mask 加到 scores 是数值计算，应计时
                S_q = Q.shape[2]
                # 注意：为了避免在计时中包含 mask 的生成开销，先生成 mask（不计时）
                mask = torch.triu(torch.ones((S_q, S_q), device=self.device) * -1e9, diagonal=1)
                # 把 mask 加到 scores 视为数值计算，计时
                t0 = perf()
                scores = scores + mask[None, None, :, :]
                t1 = perf(); compute_time += (t1 - t0)
            else:
                t0 = perf()
                # 注意：K.transpose(0,1,3,2) 是 shape 操作，K already prepared; matmul 计时
                scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
                t1 = perf(); compute_time += (t1 - t0)

                S_q = Q.shape[2]
                mask = np.triu(np.ones((S_q, S_q)) * -1e9, k=1)  # 生成 mask 不计时
                t0 = perf()
                scores = scores + mask[None, None, :, :]
                t1 = perf(); compute_time += (t1 - t0)

            # ========== softmax ==========
            t0 = perf()
            attn = self.softmax(scores, axis=-1)
            if self.use_cuda:
                self._sync()
            t1 = perf(); compute_time += (t1 - t0)

            # ========== attn @ V ==========
            t0 = perf()
            if self.use_cuda:
                aout = torch.matmul(attn, V)
                self._sync()
            else:
                aout = np.matmul(attn, V)
            t1 = perf(); compute_time += (t1 - t0)

            # ========== c_proj: aout -> attn_out (matmul + bias) ==========
            # reshape / permute 视为形状操作，不计时
            if self.use_cuda:
                B, H, S_q, d_v = aout.shape
                aout = aout.permute(0, 2, 1, 3).reshape(B, S_q, self.d_model)
                t0 = perf()
                attn_out = torch.matmul(aout, self.c_proj_w[layer]) + self.c_proj_b[layer][None, None, :]
                self._sync()
                t1 = perf(); compute_time += (t1 - t0)
            else:
                B, H, S_q, d_v = aout.shape
                aout = aout.transpose(0, 2, 1, 3).reshape(B, S_q, self.d_model)
                t0 = perf()
                attn_out = np.dot(aout, self.c_proj_w[layer]) + self.c_proj_b[layer][None, None, :]
                t1 = perf(); compute_time += (t1 - t0)

            # ========== residual after attn ==========
            # 这里 x + attn_out 是一次简单加法（数值计算），我们也计时它
            t0 = perf()
            attn_residual = x + attn_out
            if self.use_cuda:
                self._sync()
            t1 = perf(); compute_time += (t1 - t0)

            # ========== 随机矩阵相关（rand1的生成不计时，但矩阵乘计时） ==========
            # 生成 rand1（随机生成视为非计时开销）
            if self.use_cuda:
                rand1 = torch.rand(self.d_model, self.d_model, device=self.device)
                t0 = perf()
                ir1 = torch.matmul(attn_residual, rand1)
                ir2 = torch.matmul(ir1, rand1)
                self._sync()
                t1 = perf(); compute_time += (t1 - t0)
            else:
                rand1 = np.random.rand(self.d_model, self.d_model)
                t0 = perf()
                ir1 = np.dot(attn_residual, rand1)
                ir2 = np.dot(ir1, rand1)
                t1 = perf(); compute_time += (t1 - t0)

            # ========== LN2 ==========
            t0 = perf()
            ln2 = self.layer_norm(attn_residual, self.ln2_gamma[layer], self.ln2_beta[layer])
            t1 = perf(); compute_time += (t1 - t0)

            # ========== FF1 ==========
            t0 = perf()
            if self.use_cuda:
                ff1 = torch.matmul(ln2, self.mlp_c_fc_w[layer]) + self.mlp_c_fc_b[layer][None, None, :]
                self._sync()
            else:
                ff1 = np.dot(ln2, self.mlp_c_fc_w[layer]) + self.mlp_c_fc_b[layer][None, None, :]
            t1 = perf(); compute_time += (t1 - t0)

            # ========== GELU ==========
            t0 = perf()
            gelu = self.gelu(ff1)
            if self.use_cuda:
                self._sync()
            t1 = perf(); compute_time += (t1 - t0)

            # ========== FF2 ==========
            t0 = perf()
            if self.use_cuda:
                ff2 = torch.matmul(gelu, self.mlp_c_proj_w[layer]) + self.mlp_c_proj_b[layer][None, None, :]
                self._sync()
            else:
                ff2 = np.dot(gelu, self.mlp_c_proj_w[layer]) + self.mlp_c_proj_b[layer][None, None, :]
            t1 = perf(); compute_time += (t1 - t0)

            # ========== final residual (数值相加，计时) ==========
            t0 = perf()
            x = attn_residual + ff2
            if self.use_cuda:
                self._sync()
            t1 = perf(); compute_time += (t1 - t0)

            # 存储本层的纯计算耗时（毫秒）
            layer_time[f"layer {layer}"] = compute_time * 1000.0

        st = time.time()
        ln_final = self.layer_norm(x, self.final_gamma, self.final_beta)
        ed = time.time()
        layer_time[f"last LN"] = (ed - st) * 1000

        total_time = sum(layer_time.values())

        return ln_final, layer_time, total_time

    def Process(self, request, context):
        process_start = time.time()
        #print("Server start: ", process_start)

        if not hasattr(local_storage, 'all_times'):
            local_storage.all_times = {}
        
        op_id = request.op_id
        state = state_to_np(request.state)
        
        if op_id == 1000:  # 执行所有 blocks + final LN + logits
            #print(state)
            start = time.time()
            input_hidden = state['input']
            #print(f"input hidden shape: {input_hidden.shape[1]}")
            self.seq_len = input_hidden.shape[1]
            logits, layer_time, total_time = self.full_forward_all(input_hidden)
            end = time.time()
            local_storage.all_times[op_id] = ('server', (end - start) * 1000)
            #print('server', (end - start) * 1000, "ms")
            out_state_np = self._to_numpy_state({'final_hidden': logits})
            #print(out_state_np)

            # Print table
            grpc_total = (time.time() - process_start) * 1000
            print("\nTime Table:")
            print(f"{'Operation':<20}{'Time (ms)':<10}")
            print("-" * 30)
            for op, t in layer_time.items():
                print(f"{op:<20}{t:<10.2f}")
            print(f"{'Total Compute':<20}{total_time:<10.2f}")
            print(f"{'gRPC Total':<20}{grpc_total:<10.2f}")

            #print("Server end: ", time.time())
            return demo_pb2.TransformerResponse(op_id=1001, state=demo_pb2.State(items=np_to_state(out_state_np)), status="ok")
        elif op_id == 999:
            start = time.time()
            input_hidden = state['input']
            logits, _, _ = self.full_forward_all(input_hidden, whether_warmup=True)
            end = time.time()
            local_storage.all_times[op_id] = ('server', (end - start) * 1000)
            #print('server', (end - start) * 1000, "ms")
            out_state_np = self._to_numpy_state({'final_hidden': logits})
            #print("Server end: ", time.time())
            return demo_pb2.TransformerResponse(op_id=1001, state=demo_pb2.State(items=np_to_state(out_state_np)), status="ok")
            

        # 如果需要写 log
        with open('time_server.log', 'w') as f:
            print(f"{'op_id':>6} | {'Executor':>8} | {'Time (ms)':>10}", file=f)
            print("-" * 30, file=f)
            for op_id in sorted(local_storage.all_times.keys()):
                executor, time_ms = local_storage.all_times[op_id]
                print(f"{op_id:>6} | {executor:>8} | {time_ms:>10.2f}", file=f)
        local_storage.all_times = {}
        

def serve(device_choice: str):
    NNN = 10485760 * 4
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    demo_pb2_grpc.add_TransformerServiceServicer_to_server(TransformerService(device_choice), server)
    server.add_insecure_port('[::]:50052')
    print("Server listening on :50052")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer gRPC server")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    serve(args.device)