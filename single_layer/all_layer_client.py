# client_grpc.py
import grpc
import numpy as np
import io
import demo_pb2
import demo_pb2_grpc
from scipy.special import softmax

# ---------- Tensor <-> bytes ----------
def np_to_tensor(arr: np.ndarray) -> demo_pb2.Tensor:
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

# LayerNorm 函数
def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    norm = (x - mean) / std
    return norm * weight + bias

# ---------- Transformer Client ----------
class TransformerClient:
    def __init__(self):
        # 加载 LN 参数（客户端使用）
        self.ln1_weight = np.load('ln1_weight.npy')
        self.ln1_bias = np.load('ln1_bias.npy')
        self.ln2_weight = np.load('ln2_weight.npy')
        self.ln2_bias = np.load('ln2_bias.npy')

    def forward(self, i, state):
        while True:
            if i == 1:  # LN1
                state['ln1'] = layer_norm(state['input'], self.ln1_weight, self.ln1_bias)
                print("i=", i)
                i += 1
            elif i == 4:  # softmax
                state['attn'] = softmax(state['scores'], axis=-1)
                print("i=", i)
                i += 1
            elif i == 8:  # LN2
                state['ln2'] = layer_norm(state['attn_residual'], self.ln2_weight, self.ln2_bias)
                print("i=", i)
                i += 1
            elif i == 10:  # GELU
                x = state['ff1']
                state['gelu'] = x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
                print("i=", i)
                i += 1
            else:
                print("BREAK: i=", i)
                break
        return i, state

# ---------- RPC ----------
def run():
    B, S, D = 2, 64, 768
    state = {'input': np.random.randn(B, S, D).astype(np.float32)}

    NNN = 10485760*2
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    channel = grpc.insecure_channel('localhost:50051', options=options)
    stub = demo_pb2_grpc.TransformerServiceStub(channel)

    i = 1  # 从 LN1 开始
    transformer = TransformerClient()
    while True:
        i, state = transformer.forward(i, state)
        if i > 12:
            print(f"Finished, final output shape={state['output'].shape}")
            break

        req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=np_to_state(state)))
        resp = stub.Process(req)

        i = resp.op_id
        state = state_to_np(resp.state)
        
        if 'aout' in state:
            print(f"[op_id={i}] Attention output shape: {state['aout'].shape}")
        if 'ff2' in state:
            print(f"[op_id={i}] FFN output shape: {state['ff2'].shape}")
        if 'output' in state:
            print(f"[op_id={i}] Final output shape: {state['output'].shape}")
        print(list(state.keys()))

if __name__ == '__main__':
    run()