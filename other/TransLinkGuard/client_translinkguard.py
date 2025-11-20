# client_grpc.py (完整可运行版，支持完整GPT-2 Small推理)
import grpc
import numpy as np
import io
import time
import demo_pb2
import demo_pb2_grpc
from transformers import AutoTokenizer
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
        # 加载所有参数从单一.npz（原代码不变）
        data = np.load('gpt2_params/params.npz')
        
        self.n_layer = int(data['n_layer'][0])
        
        # 切分栈数组回列表（原代码不变）
        self.ln1_gamma = [data['ln1_gamma'][i] for i in range(self.n_layer)]
        self.ln1_beta = [data['ln1_beta'][i] for i in range(self.n_layer)]
        #self.ln2_gamma = [data['ln2_gamma'][i] for i in range(self.n_layer)]
        #self.ln2_beta = [data['ln2_beta'][i] for i in range(self.n_layer)]
        
        self.final_gamma = data['final_gamma']
        self.final_beta = data['final_beta']
        
        self.wte = data['wte']
        self.wpe = data['wpe']
        
        # 加载 tokenizer 从目录（原代码不变）
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2_params/tokenizer')
        
        # 新增：假设hidden_size（GPT-2 small: 768）
        self.hidden_size = 768
        self.seq_len = 1
        self.mask_gelu = np.random.rand(1, self.seq_len, self.hidden_size*4)
        self.mask_resi = np.random.rand(1, self.seq_len, self.hidden_size)
        self.perm_gelu = np.random.permutation(self.hidden_size*4)
        self.perm_resi = np.random.permutation(self.hidden_size)
        
        # 新增：全局warmup，在初始化时预热所有操作类型（不计入时间测量）
        self._warmup_operations()

    def _warmup_operations(self):
        warmup_runs = 5  # warmup次数
        # Dummy数据：匹配典型形状 (batch=1, seq=1, hidden=768; attn_heads=12; ffn=3072)
        dummy_input_ln = np.random.randn(1, 1, self.hidden_size)
        dummy_gamma = np.ones(self.hidden_size)
        dummy_beta = np.zeros(self.hidden_size)
        
        dummy_scores_softmax = np.random.randn(1, 12, 1, 1)  # attn scores
        
        dummy_ff1_gelu = np.random.randn(1, 1, self.hidden_size * 4)  # FFN intermediate
        
        for _ in range(warmup_runs):
            # Pre-warmup LayerNorm (for LN1, LN2, final)
            layer_norm(dummy_input_ln, dummy_gamma, dummy_beta)
            
            # Pre-warmup Softmax
            sp_softmax(dummy_scores_softmax, axis=-1)
            
            # Pre-warmup GELU
            x = dummy_ff1_gelu
            x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def forward(self, i, state):
        current_layer = i // 100
        local_i = i % 100
        #print(f"i={i}")

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


            elif local_i == 10:  # GELU
                x = state['ff1']
                gelu_res = x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
                
                st = time.time()
                gelu_obf = (gelu_res + self.mask_gelu)[:, :, self.perm_gelu]
                et = time.time()
                #print(f"GELU_obf cost {(et-st)*1000:.3f} ms")
                state['gelu'] = gelu_res
                
                i += 1
                local_i += 1

            elif local_i == 12:
                # state['output'] = state['attn_residual'] + state['ff2']
                st = time.time()
                
                attn_resi = state['attn_residual'] 
                ff2 = state['ff2']
                attn_resi_res = (attn_resi-self.mask_resi)[:, :, self.perm_resi]
                ff2_res = ff2[:, :, self.perm_resi]
                
                et = time.time()
                #print(f"resi_obf cost {(et-st)*1000:.3f} ms")
                
                state['output'] = state['attn_residual'] + state['ff2']
                #state['output'] = attn_resi + ff2
                i += 1
                local_i += 1
            else:
                break
        return i, state


def perform_inference(client, stub, input_text, collect_times=True):
    start_time = time.time() if collect_times else None
    input_ids = client.tokenizer(input_text, return_tensors="np")["input_ids"]  # (1, seq_len)
    seq_len = input_ids.shape[1]
    
    client.seq_len = seq_len
    client.mask_gelu = np.random.rand(1, client.seq_len, client.hidden_size*4)

    hidden = client.wte[input_ids] + client.wpe[np.arange(seq_len)]
    
    client.seq_len = hidden.shape[1]

    state = {'input': hidden.astype(np.float32)}
    
    all_times = {} if collect_times else None
    
    i = 1  # start block 0

    for layer in range(client.n_layer):
        state = {'input': hidden}  # clean state, keep only input for residual
        i = layer * 100 + 1

        while True:
            old_i = i
            start = time.time() if collect_times else None
            i, state = client.forward(i, state)
            end = time.time() if collect_times else None
            if collect_times and i > old_i:  # Client processed this op_id
                all_times[old_i] = ('client', (end - start) * 1000)

            if i % 100 > 12:  # server finished the block
                hidden = state['output']
                state = {'input': hidden}  # clean for next layer
                break

            # Send to server
            req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=np_to_state(state)))
            start = time.time() if collect_times else None
            resp = stub.Process(req)
            end = time.time() if collect_times else None
            sent_i = i
            i = resp.op_id
            state = state_to_np(resp.state)
            if collect_times:
                all_times[sent_i] = ('server', (end - start) * 1000)  # Attribute to sent op_id

    # Final LN
    i = 1201
    old_i = i
    start = time.time() if collect_times else None
    i, state = client.forward(i, state)
    end = time.time() if collect_times else None
    if collect_times and i > old_i:
        all_times[old_i] = ('client', (end - start) * 1000)
    
    req = demo_pb2.TransformerRequest(op_id=i, state=demo_pb2.State(items=np_to_state(state)))
    start = time.time() if collect_times else None
    resp = stub.Process(req)
    end = time.time() if collect_times else None
    if collect_times:
        all_times[i] = ('server', (end - start) * 1000) 
    state = state_to_np(resp.state)
    logits = state['logits']

    if collect_times:
        #print(f"Logits shape: {logits.shape}")
        next_token_id = int(np.argmax(logits[0, -1, :]))
        print(f"Next token: '{client.tokenizer.decode(next_token_id)}'")
        end_time = time.time()
        print(f"Total time cost: {(end_time - start_time) * 1000:.3f} ms")

        with open('time_client.log', 'w') as f:
            print(f"{'op_id':>6} | {'Executor':>8} | {'Time (ms)':>10}", file=f)
            print("-" * 30, file=f)
            for op_id in sorted(all_times.keys()):
                executor, time_ms = all_times[op_id]
                print(f"{op_id:>6} | {executor:>8} | {time_ms:>10.2f}", file=f)

    return logits  # 返回logits以便warmup时丢弃

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
    
    warmup_runs = 1  # 可根据需要调整
    #print(f"Performing {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        _ = perform_inference(client, stub, "Warmup input", collect_times=False)  # 用短输入预热

    time_count = time_cost("warm up", time_count)
    
    input_text = "The capital of France is"
    _ = perform_inference(client, stub, input_text, collect_times=True)
    
if __name__ == '__main__':
    run()
