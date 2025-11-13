# server_grpc.py (完整可运行版，支持GPT-2 Small全12层 + embedding + final LM head，带causal mask、bias处理、权重从HF加载)
import grpc
import numpy as np
import io
from concurrent import futures
import demo_pb2
import demo_pb2_grpc
from transformers import AutoConfig, GPT2Model

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
    return {k: tensor_to_np(v) for k, v in state_pb.items.items()}  # 修复版（用户原代码可能已修复）

def np_to_state(state_np):
    return {k: np_to_tensor(v) for k, v in state_np.items()}

class TransformerService(demo_pb2_grpc.TransformerServiceServicer):
    def __init__(self):
        config = AutoConfig.from_pretrained("gpt2")
        self.n_layer = config.n_layer
        self.d_model = config.n_embd
        self.h = config.n_head
        self.d_k = self.d_model // self.h

        model = GPT2Model.from_pretrained("gpt2")

        self.c_attn_w = [layer.attn.c_attn.weight.detach().cpu().numpy() for layer in model.h]  # (768, 2304)
        self.c_attn_b = [layer.attn.c_attn.bias.detach().cpu().numpy() for layer in model.h]

        self.c_proj_w = [layer.attn.c_proj.weight.detach().cpu().numpy() for layer in model.h]  # (768, 768)
        self.c_proj_b = [layer.attn.c_proj.bias.detach().cpu().numpy() for layer in model.h]

        self.mlp_c_fc_w = [layer.mlp.c_fc.weight.detach().cpu().numpy() for layer in model.h]  # (768, 3072)
        self.mlp_c_fc_b = [layer.mlp.c_fc.bias.detach().cpu().numpy() for layer in model.h]

        self.mlp_c_proj_w = [layer.mlp.c_proj.weight.detach().cpu().numpy() for layer in model.h]  # (3072, 768)
        self.mlp_c_proj_b = [layer.mlp.c_proj.bias.detach().cpu().numpy() for layer in model.h]

        self.lm_head_w = model.wte.weight.detach().cpu().numpy().T  # (768, 50257)
        
        print("Number of layers:", self.n_layer)
        print("\nShapes for c_attn_w (per layer):")
        for i, w in enumerate(self.c_attn_w):
            print(f"Layer {i}: {w.shape}")
        
        print("\nShapes for c_attn_b (per layer):")
        for i, b in enumerate(self.c_attn_b):
            print(f"Layer {i}: {b.shape}")
        
        print("\nShapes for c_proj_w (per layer):")
        for i, w in enumerate(self.c_proj_w):
            print(f"Layer {i}: {w.shape}")
        
        print("\nShapes for c_proj_b (per layer):")
        for i, b in enumerate(self.c_proj_b):
            print(f"Layer {i}: {b.shape}")
        
        print("\nShapes for mlp_c_fc_w (per layer):")
        for i, w in enumerate(self.mlp_c_fc_w):
            print(f"Layer {i}: {w.shape}")
        
        print("\nShapes for mlp_c_fc_b (per layer):")
        for i, b in enumerate(self.mlp_c_fc_b):
            print(f"Layer {i}: {b.shape}")
        
        print("\nShapes for mlp_c_proj_w (per layer):")
        for i, w in enumerate(self.mlp_c_proj_w):
            print(f"Layer {i}: {w.shape}")
        
        print("\nShapes for mlp_c_proj_b (per layer):")
        for i, b in enumerate(self.mlp_c_proj_b):
            print(f"Layer {i}: {b.shape}")
        
        print("\nShape for lm_head_w:")
        print(self.lm_head_w.shape)

    def Process(self, request, context):
        op_id = request.op_id
        state = state_to_np(request.state)
        i = op_id
        current_layer = i // 100

        while True:
            local_i = i % 100

            if current_layer >= self.n_layer:  # final linear
                if local_i == 2:
                    state['logits'] = np.dot(state['ln_final'], self.lm_head_w)  # no bias
                    i = 9999
                break

            if local_i == 2:  # QKV
                inp = state['ln1']
                #print("i= ", i, "inp shape: ", inp.shape)
                w = self.c_attn_w[current_layer]
                b = self.c_attn_b[current_layer]
                #print("input w shape: ", w.shape,"input b shape: ", b.shape)
                proj = np.dot(inp, w) + b[None, None, :]
                B, S, _ = inp.shape
                Q = proj[:, :, :self.d_model]
                K = proj[:, :, self.d_model:2*self.d_model]
                V = proj[:, :, 2*self.d_model:]
                Q = Q.reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                K = K.reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                V = V.reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                state['Q'] = Q
                state['K'] = K
                state['V'] = V
                i += 1

            elif local_i == 3:  # scores + causal mask
                scores = np.matmul(state['Q'], state['K'].transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
                S_q = state['Q'].shape[2]
                mask = np.triu(np.ones((S_q, S_q), dtype=scores.dtype), k=1) * -1e9
                scores += mask[None, None, :, :]
                state['scores'] = scores
                i += 1

            # 其余部分保持不变，只改 np.dot为 np.dot(..., w)
            elif local_i == 5:
                state['aout'] = np.matmul(state['attn'], state['V'])
                i += 1

            elif local_i == 6:
                B, H, S_q, d_v = state['aout'].shape
                aout = state['aout'].transpose(0, 2, 1, 3).reshape(B, S_q, self.d_model)
                w = self.c_proj_w[current_layer]
                b = self.c_proj_b[current_layer]
                state['attn_out'] = np.dot(aout, w) + b[None, None, :]
                i += 1

            elif local_i == 7:
                state['attn_residual'] = state['input'] + state['attn_out']
                i += 1

            elif local_i == 9:
                w = self.mlp_c_fc_w[current_layer]
                b = self.mlp_c_fc_b[current_layer]
                state['ff1'] = np.dot(state['ln2'], w) + b[None, None, :]
                i += 1

            elif local_i == 11:
                w = self.mlp_c_proj_w[current_layer]
                b = self.mlp_c_proj_b[current_layer]
                state['ff2'] = np.dot(state['gelu'], w) + b[None, None, :]
                i += 1

            elif local_i == 12:
                state['output'] = state['attn_residual'] + state['ff2']
                i += 1

            else:
                break

        return demo_pb2.TransformerResponse(op_id=i, state=demo_pb2.State(items=np_to_state(state)), status="ok")
# serve() 保持不变
def serve():
    NNN = 10485760 * 4  # 更大一点，logits 可能大
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    demo_pb2_grpc.add_TransformerServiceServicer_to_server(TransformerService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server listening on :50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()