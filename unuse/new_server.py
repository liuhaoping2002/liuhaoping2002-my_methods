# server_grpc.py 
import grpc
import numpy as np
import io
from concurrent import futures
import demo_pb2
import demo_pb2_grpc
from scipy.special import softmax

def tensor_to_np(t: demo_pb2.Tensor) -> np.ndarray:
    #buf = io.BytesIO(t.data)
    #return np.load(buf)
    buf = io.BytesIO(t.data)
    arr = np.load(buf, mmap_mode=None)
    return arr.copy()  # 强制转成 resident array

def np_to_tensor(arr: np.ndarray) -> demo_pb2.Tensor:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return demo_pb2.Tensor(
        data=buf.getvalue(),
        shape=list(arr.shape),      # 关键：list(arr.shape)
        dtype=str(arr.dtype)
    )

def state_to_np(state_pb):
    return {k: tensor_to_np(v) for k, v in state_pb.items.items()}

def np_to_state(state_np):
    return {k: np_to_tensor(v) for k, v in state_np.items()}

class TransformerService(demo_pb2_grpc.TransformerServiceServicer):
    def __init__(self):
        self.d_model = 64
        self.h = 8
        self.d_k = self.d_model // self.h
        self.d_ff = self.d_model * 4
        self.w_q = np.random.randn(self.d_model, self.d_model)
        self.w_k = np.random.randn(self.d_model, self.d_model)
        self.w_v = np.random.randn(self.d_model, self.d_model)
        self.w_o = np.random.randn(self.d_model, self.d_model)
        self.ff_linear1 = np.random.randn(self.d_model, self.d_ff)
        self.ff_linear2 = np.random.randn(self.d_ff, self.d_model)

    def Process(self, request, context):
        op_id = request.op_id
        state = state_to_np(request.state)
        i = op_id
        while True:
            if i == 1:
                # linear projections for Q, K, V
                inp = state['input']
                B, S, D = inp.shape
                Q = np.dot(inp, self.w_q).reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                K = np.dot(inp, self.w_k).reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                V = np.dot(inp, self.w_v).reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                state['Q'] = Q
                state['K'] = K
                state['V'] = V
                i += 1
            elif i == 2:
                # linear matmul and scale for scores
                state['scores'] = np.matmul(state['Q'], state['K'].transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
                i += 1
            elif i == 4:
                # linear matmul for attention output
                state['aout'] = np.matmul(state['attn'], state['V'])
                i += 1
            elif i == 5:
                # linear output projection
                B, H, S_q, d_v = state['aout'].shape
                aout = state['aout'].transpose(0, 2, 1, 3).reshape(B, S_q, self.d_model)
                state['attn_out'] = np.dot(aout, self.w_o)
                i += 1
            elif i == 6:
                # linear FFN first layer
                state['ff1'] = np.dot(state['attn_out'], self.ff_linear1)
                i += 1
            elif i == 8:
                # linear FFN second layer
                state['ff2'] = np.dot(state['gelu'], self.ff_linear2)
                i += 1
            else:
                # not a server op or end
                break
        if i > 8:
            i = 999
        return demo_pb2.TransformerResponse(op_id=i, state=demo_pb2.State(items=np_to_state(state)), status="ok")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    demo_pb2_grpc.add_TransformerServiceServicer_to_server(TransformerService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server listening on :50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()