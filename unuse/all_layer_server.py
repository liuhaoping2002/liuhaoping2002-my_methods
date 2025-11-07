
# server_grpc.py 
import grpc
import numpy as np
import io
from concurrent import futures
import demo_pb2
import demo_pb2_grpc

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

class TransformerService(demo_pb2_grpc.TransformerServiceServicer):
    def __init__(self):
        self.d_model = 768
        self.h = 12
        self.d_k = self.d_model // self.h
        self.d_ff = 3072

        # 加载权重
        self.w_q = np.load('w_q.npy')
        self.w_k = np.load('w_k.npy')
        self.w_v = np.load('w_v.npy')
        self.w_o = np.load('w_o.npy')
        self.ff_linear1 = np.load('ff_linear1.npy')
        self.ff_linear2 = np.load('ff_linear2.npy')
        # LN 参数（服务器不直接用，但加载以备）
        self.ln1_weight = np.load('ln1_weight.npy')
        self.ln1_bias = np.load('ln1_bias.npy')
        self.ln2_weight = np.load('ln2_weight.npy')
        self.ln2_bias = np.load('ln2_bias.npy')

    def Process(self, request, context):
        op_id = request.op_id
        state = state_to_np(request.state)
        i = op_id
        while True:
            if i == 2:  # QKV proj (after LN1 in client)
                inp = state['ln1']  # 使用 normed input
                B, S, D = inp.shape
                Q = np.dot(inp, self.w_q).reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                K = np.dot(inp, self.w_k).reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                V = np.dot(inp, self.w_v).reshape(B, S, self.h, self.d_k).transpose(0, 2, 1, 3)
                state['Q'] = Q
                state['K'] = K
                state['V'] = V
                print("i=", i)
                i += 1
            elif i == 3:
                state['scores'] = np.matmul(state['Q'], state['K'].transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
                print("i=", i)
                i += 1
            elif i == 5:
                state['aout'] = np.matmul(state['attn'], state['V'])
                print("i=", i)
                i += 1
            elif i == 6:
                B, H, S_q, d_v = state['aout'].shape
                aout = state['aout'].transpose(0, 2, 1, 3).reshape(B, S_q, self.d_model)
                state['attn_out'] = np.dot(aout, self.w_o)
                print("i=", i)
                i += 1
            elif i == 7:  # attn residual: input + attn_out
                state['attn_residual'] = state['input'] + state['attn_out']
                print("i=", i)
                i += 1
            elif i == 9:  # FF1 dot (after LN2 in client)
                state['ff1'] = np.dot(state['ln2'], self.ff_linear1)
                print("i=", i)
                i += 1
            elif i == 11:
                state['ff2'] = np.dot(state['gelu'], self.ff_linear2)
                print("i=", i)
                i += 1
            elif i == 12:  # ff residual: attn_residual + ff2
                state['output'] = state['attn_residual'] + state['ff2']
                print("i=", i)
                i += 1
            else:
                print(type(i))
                print("BREAK: i=", i)
                break
        if i > 12:
            i = 999
        return demo_pb2.TransformerResponse(op_id=i, state=demo_pb2.State(items=np_to_state(state)), status="ok")

def serve():
    options = [
        ('grpc.max_send_message_length', 10485760),
        ('grpc.max_receive_message_length', 10485760)
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    demo_pb2_grpc.add_TransformerServiceServicer_to_server(TransformerService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server listening on :50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()