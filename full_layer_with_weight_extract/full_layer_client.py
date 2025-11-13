# client_grpc.py (完整可运行版，支持完整GPT-2 Small推理)
import grpc
import numpy as np
import io
import time
import demo_pb2
import demo_pb2_grpc
from transformers import AutoConfig, GPT2Model, AutoTokenizer
from scipy.special import softmax as sp_softmax


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
    return {k: tensor_to_np(v) for k, v in state_pb.items.items()}  # 修复版

def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    norm = (x - mean) / std
    return norm * weight + bias

class TransformerClient:
    def __init__(self):
        '''config = AutoConfig.from_pretrained("gpt2")
        self.n_layer = config.n_layer  # 12

        model = GPT2Model.from_pretrained("gpt2")

        self.ln1_gamma = [layer.ln_1.weight.detach().cpu().numpy() for layer in model.h]
        self.ln1_beta = [layer.ln_1.bias.detach().cpu().numpy() for layer in model.h]
        self.ln2_gamma = [layer.ln_2.weight.detach().cpu().numpy() for layer in model.h]
        self.ln2_beta = [layer.ln_2.bias.detach().cpu().numpy() for layer in model.h]

        self.final_gamma = model.ln_f.weight.detach().cpu().numpy()
        self.final_beta = model.ln_f.bias.detach().cpu().numpy()

        self.wte = model.wte.weight.detach().cpu().numpy()
        self.wpe = model.wpe.weight.detach().cpu().numpy()

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")'''
        # 加载所有参数从单一.npz
        data = np.load('gpt2_params/params.npz')
        
        self.n_layer = int(data['n_layer'][0])
        
        # 切分栈数组回列表
        self.ln1_gamma = [data['ln1_gamma'][i] for i in range(self.n_layer)]
        self.ln1_beta = [data['ln1_beta'][i] for i in range(self.n_layer)]
        self.ln2_gamma = [data['ln2_gamma'][i] for i in range(self.n_layer)]
        self.ln2_beta = [data['ln2_beta'][i] for i in range(self.n_layer)]
        
        self.final_gamma = data['final_gamma']
        self.final_beta = data['final_beta']
        
        self.wte = data['wte']
        self.wpe = data['wpe']
        
        # 加载 tokenizer 从目录
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2_params/tokenizer')

    def forward(self, i, state):
        current_layer = i // 100
        local_i = i % 100

        # 先处理 final LN，避免 index error
        if i == 1201:
            state['ln_final'] = layer_norm(state['input'], self.final_gamma, self.final_beta)
            return 1202, state

        while True:
            if local_i == 1:  # LN1
                gamma = self.ln1_gamma[current_layer]
                beta = self.ln1_beta[current_layer]
                state['ln1'] = layer_norm(state['input'], gamma, beta)
                i += 1
                local_i += 1

            elif local_i == 4:  # softmax
                state['attn'] = sp_softmax(state['scores'], axis=-1)
                i += 1
                local_i += 1

            elif local_i == 8:  # LN2
                gamma = self.ln2_gamma[current_layer]
                beta = self.ln2_beta[current_layer]
                state['ln2'] = layer_norm(state['attn_residual'], gamma, beta)
                i += 1
                local_i += 1

            elif local_i == 10:  # GELU
                x = state['ff1']
                state['gelu'] = x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
                i += 1
                local_i += 1

            else:
                break
        return i, state

def run():
    time_count = time.time()
    NNN = 10485760 * 4
    options = [
        ('grpc.max_send_message_length', NNN),
        ('grpc.max_receive_message_length', NNN)
    ]
    channel = grpc.insecure_channel('localhost:50051', options=options)
    stub = demo_pb2_grpc.TransformerServiceStub(channel)
    
    time_count = time_cost("channel establiash", time_count)
    client = TransformerClient()
    
    # 输入示例
    input_text = "The capital of France is"
    input_ids = client.tokenizer(input_text, return_tensors="np")["input_ids"]  # (1, seq_len)
    seq_len = input_ids.shape[1]

    hidden = client.wte[input_ids] + client.wpe[np.arange(seq_len)]

    state = {'input': hidden.astype(np.float32)}
    time_count = time_cost("Tokenizer", time_count)
    
    i = 1  # start block 0

    for layer in range(client.n_layer):
        state = {'input': hidden}  # clean state, keep only input for residual
        i = layer * 100 + 1

        while True:
            i, state = client.forward(i, state)

            if i % 100 > 12:  # server finished the block
                hidden = state['output']
                state = {'input': hidden}  # clean for next layer
                break

            req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=np_to_state(state)))
            resp = stub.Process(req)
            i = resp.op_id
            state = state_to_np(resp.state)
            if i%100 == 13:
                time_count = time_cost(f"Layer {i//100}", time_count)
            #print(f"[client] layer {layer} received op_id={i} keys={list(state.keys())}")

    # Final LN
    i = 1201
    i, state = client.forward(i, state)
    req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=np_to_state(state)))
    resp = stub.Process(req)
    state = state_to_np(resp.state)
    logits = state['logits']
    time_count = time_cost(f"Final LN", time_count)

    print(f"Logits shape: {logits.shape}")
    # 示例生成下一个token
    next_token_id = int(np.argmax(logits[0, -1, :]))
    print(f"Next token: '{client.tokenizer.decode(next_token_id)}'")
    time_count = time_cost(f"Decode", time_count)

if __name__ == '__main__':
    run()
