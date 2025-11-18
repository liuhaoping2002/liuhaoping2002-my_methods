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
        data = np.load('gpt2_server_params.npz')
        client_data = np.load('gpt2_params/params.npz')  # 加载 LN 和 final LN 参数

        self.n_layer = int(data['n_layer'][0])
        self.d_model = data['c_attn_w'].shape[-1] // 3
        self.h = 12
        self.d_k = self.d_model // self.h

        # 线性层参数
        c_attn_w_np = [data['c_attn_w'][i] for i in range(self.n_layer)]
        c_attn_b_np = [data['c_attn_b'][i] for i in range(self.n_layer)]

        c_proj_w_np = [data['c_proj_w'][i] for i in range(self.n_layer)]
        c_proj_b_np = [data['c_proj_b'][i] for i in range(self.n_layer)]

        mlp_c_fc_w_np = [data['mlp_c_fc_w'][i] for i in range(self.n_layer)]
        mlp_c_fc_b_np = [data['mlp_c_fc_b'][i] for i in range(self.n_layer)]

        mlp_c_proj_w_np = [data['mlp_c_proj_w'][i] for i in range(self.n_layer)]
        mlp_c_proj_b_np = [data['mlp_c_proj_b'][i] for i in range(self.n_layer)]

        lm_head_w_np = data['lm_head_w']

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
            self.lm_head_w = torch.from_numpy(lm_head_w_np).to(self.device)

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
            self.lm_head_w = lm_head_w_np

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

    def full_forward_all(self, input_hidden, whether_warmup=False):
        layer_time = {}
        s = self._to_torch_state({'input': input_hidden})
        x = s['input']

        for layer in range(self.n_layer):
            st = time.time()
            # LN1
            ln1 = self.layer_norm(x, self.ln1_gamma[layer], self.ln1_beta[layer])

            # QKV
            if self.use_cuda:
                proj = torch.matmul(ln1, self.c_attn_w[layer]) + self.c_attn_b[layer][None, None, :]
                B, S, _ = ln1.shape
                Q = proj[:, :, :self.d_model].reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                K = proj[:, :, self.d_model:2*self.d_model].reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
                V = proj[:, :, 2*self.d_model:].reshape(B, S, self.h, self.d_k).permute(0, 2, 1, 3)
            else:
                proj = np.dot(ln1, self.c_attn_w[layer]) + self.c_attn_b[layer][None, None, :]
                B, S, _ = ln1.shape
                Q = proj[:, :, :self.d_model].reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                K = proj[:, :, self.d_model:2*self.d_model].reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                V = proj[:, :, 2*self.d_model:].reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)

            # scores + mask
            if self.use_cuda:
                scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
                S_q = Q.shape[2]
                mask = torch.triu(torch.ones((S_q, S_q), device=self.device) * -1e9, diagonal=1)
                scores = scores + mask[None, None, :, :]
            else:
                scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
                S_q = Q.shape[2]
                mask = np.triu(np.ones((S_q, S_q)) * -1e9, k=1)
                scores += mask[None, None, :, :]

            # softmax
            attn = self.softmax(scores, axis=-1)

            # attn @ V
            if self.use_cuda:
                aout = torch.matmul(attn, V)
            else:
                aout = np.matmul(attn, V)

            # c_proj
            if self.use_cuda:
                B, H, S_q, d_v = aout.shape
                aout = aout.permute(0, 2, 1, 3).reshape(B, S_q, self.d_model)
                attn_out = torch.matmul(aout, self.c_proj_w[layer]) + self.c_proj_b[layer][None, None, :]
            else:
                B, H, S_q, d_v = aout.shape
                aout = aout.transpose(0, 2, 1, 3).reshape(B, S_q, self.d_model)
                attn_out = np.dot(aout, self.c_proj_w[layer]) + self.c_proj_b[layer][None, None, :]

            # residual after attn
            attn_residual = x + attn_out

            # LN2
            ln2 = self.layer_norm(attn_residual, self.ln2_gamma[layer], self.ln2_beta[layer])

            # FF1
            if self.use_cuda:
                ff1 = torch.matmul(ln2, self.mlp_c_fc_w[layer]) + self.mlp_c_fc_b[layer][None, None, :]
            else:
                ff1 = np.dot(ln2, self.mlp_c_fc_w[layer]) + self.mlp_c_fc_b[layer][None, None, :]

            # GELU
            gelu = self.gelu(ff1)

            # FF2
            if self.use_cuda:
                ff2 = torch.matmul(gelu, self.mlp_c_proj_w[layer]) + self.mlp_c_proj_b[layer][None, None, :]
            else:
                ff2 = np.dot(gelu, self.mlp_c_proj_w[layer]) + self.mlp_c_proj_b[layer][None, None, :]

            # final residual
            x = attn_residual + ff2
            ed = time.time()
            layer_time[f"layer {layer}"] = (ed - st) * 1000

        # Final LN (现在在 REE)
        st = time.time()
        ln_final = self.layer_norm(x, self.final_gamma, self.final_beta)
        ed = time.time()
        layer_time[f"last LN"] = (ed - st) * 1000

        # LM Head (logits)
        st = time.time()
        if self.use_cuda:
            logits = torch.matmul(ln_final, self.lm_head_w)
        else:
            logits = np.dot(ln_final, self.lm_head_w)
        ed = time.time()
        layer_time[f"logits"] = (ed - st) * 1000

        total_time = 0
        if whether_warmup == False:
            for op in layer_time:
                print(f"{op} cost {layer_time[op]} ms")
                total_time += layer_time[op]
            print(f"Total time cost {total_time} ms")
        return logits

    def Process(self, request, context):
        print("Server start: ", time.time())
        if not hasattr(local_storage, 'all_times'):
            local_storage.all_times = {}
        
        op_id = request.op_id
        state = state_to_np(request.state)
        
        if op_id == 1000:  # 执行所有 blocks + final LN + logits
            print(state)
            start = time.time()
            input_hidden = state['input']
            logits = self.full_forward_all(input_hidden)
            end = time.time()
            local_storage.all_times[op_id] = ('server', (end - start) * 1000)
            print('server', (end - start) * 1000, "ms")
            out_state_np = self._to_numpy_state({'logits': logits})
            print(out_state_np)
            print("Server end: ", time.time())
            return demo_pb2.TransformerResponse(op_id=1001, state=demo_pb2.State(items=np_to_state(out_state_np)), status="ok")
        elif op_id == 999:
            start = time.time()
            input_hidden = state['input']
            logits = self.full_forward_all(input_hidden, whether_warmup=True)
            end = time.time()
            local_storage.all_times[op_id] = ('server', (end - start) * 1000)
            #print('server', (end - start) * 1000, "ms")
            out_state_np = self._to_numpy_state({'logits': logits})
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
    server.add_insecure_port('[::]:50051')
    print("Server listening on :50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer gRPC server")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    serve(args.device)