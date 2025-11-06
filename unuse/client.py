# client_grpc.py
import grpc
import numpy as np
import io
import demo_pb2
import demo_pb2_grpc
from scipy.special import softmax
import time

# ---------- Tensor <-> bytes ----------
def np_to_tensor(arr: np.ndarray) -> demo_pb2.Tensor:
    buf = io.BytesIO()
    np.save(buf, arr)
    return demo_pb2.Tensor(
        data=buf.getvalue(),
        shape=arr.shape,
        dtype=str(arr.dtype)
    )

def tensor_to_np(t: demo_pb2.Tensor) -> np.ndarray:
    #buf = io.BytesIO(t.data)
    #return np.load(buf)
    buf = io.BytesIO(t.data)
    arr = np.load(buf, mmap_mode=None)
    return arr.copy()  # 强制转成 resident array
    

def np_to_state(state_np):
    return {k: np_to_tensor(v) for k, v in state_np.items()}

def state_to_np(state_pb):
    return {k: tensor_to_np(v) for k, v in state_pb.items.items()}

# ---------- Transformer Client ----------
class TransformerClient:
    def forward(self, i, state):
        while True:
            if i == 3:
                # non-linear softmax
                state['attn'] = softmax(state['scores'], axis=-1)
                i += 1
            elif i == 7:
                # non-linear GELU approximation
                x = state['ff1']
                state['gelu'] = x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
                i += 1
            else:
                # not a client op
                break
        return i, state

# ---------- RPC ----------
def run():
    # 模拟数据
    B, S, D = 2, 64, 64
    state = {'input': np.random.randn(B, S, D).astype(np.float32)}

    channel = grpc.insecure_channel('localhost:50051')
    stub = demo_pb2_grpc.TransformerServiceStub(channel)

    i = 1
    transformer = TransformerClient()
    while True:
        i, state = transformer.forward(i, state)
        if i > 8:
            print(f"Finished, final output shape={state['ff2'].shape}")
            break

        req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=np_to_state(state)))
        resp = stub.Process(req)

        i = resp.op_id
        state = state_to_np(resp.state)
        
        if 'aout' in state:
            #arr = state['aout']
            #print("type:", type(arr))
            #print("is memmap:", isinstance(arr, np.memmap))
            #print("filename:", getattr(arr, 'filename', 'N/A'))
            print(f"[op_id={i}] Final FFN output: {state['aout']}")
        if 'ff2' in state:
            print(f"[op_id={i}] Final FFN output: {state['ff2']}")
        print(list(state.keys()))

if __name__ == '__main__':
    run()