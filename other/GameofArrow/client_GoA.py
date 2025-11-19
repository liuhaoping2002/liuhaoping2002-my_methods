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
    time_cost = time_now - time_past
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

def permutation_from_key(key: str, n: int = 4096):
    rng = np.random.RandomState(hash(key) % (2**32))
    arr = np.arange(n)
    rng.shuffle(arr)
    return arr


class TransformerClient:
    def __init__(self):
        # 加载所有参数从单一.npz（原代码不变）
        data = np.load('gpt2_params/params.npz')
        
        self.n_layer = int(data['n_layer'][0])
        
        # 切分栈数组回列表（原代码不变）
        self.ln1_gamma = [data['ln1_gamma'][i] for i in range(self.n_layer)]
        self.ln1_beta = [data['ln1_beta'][i] for i in range(self.n_layer)]
        self.ln2_gamma = [data['ln2_gamma'][i] for i in range(self.n_layer)]
        self.ln2_beta = [data['ln2_beta'][i] for i in range(self.n_layer)]
        
        self.final_gamma = data['final_gamma']
        self.final_beta = data['final_beta']
        
        self.wte = data['wte']
        self.wpe = data['wpe']
        
        # 加载 tokenizer 从目录（原代码不变）
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2_params/tokenizer')
        
        # 新增：假设hidden_size（GPT-2 small: 768）
        self.hidden_size = 768
        self.head_num = 12
        self.head_dim = self.hidden_size//self.head_num
        
        self.obf1 = np.random.rand(self.hidden_size, 1)
        self.seq_len = 1
        self.obf2 = np.random.rand(self.seq_len, 1)
        self.perm1 = permutation_from_key(key="perm1", n=self.hidden_size)
        self.perm2 = permutation_from_key(key="perm2", n=self.seq_len)
        self.perm3 = permutation_from_key(key="perm3", n=self.head_dim)
        self.perm4 = permutation_from_key(key="perm4", n=self.hidden_size*4)
        self.ir1 = np.random.rand(self.seq_len, self.hidden_size)
        self.ir2 = np.random.rand(self.seq_len, self.seq_len)
        self.v1 = np.random.rand(1, self.hidden_size)
        self.v2 = np.random.rand(1, self.seq_len)
        self.v4 = np.random.rand(1, self.hidden_size*4)
        self.D1 = np.diag(np.random.randn(self.hidden_size)).reshape(self.head_num, self.head_dim, self.hidden_size) # d*d维对角
        self.D2 = np.diag(np.random.randn(self.seq_len))
        self.D3 = np.diag(np.random.randn(self.hidden_size))
        self.D4 = np.diag(np.random.randn(self.hidden_size*4))
        # 新增：全局warmup，在初始化时预热所有操作类型（不计入时间测量）
        self._warmup_operations()

    def _warmup_operations(self):
        warmup_runs = 5  # warmup次数
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

        # 先处理 final LN，避免 index error
        if i == 1201:
            state['ln_final'] = layer_norm(state['input'], self.final_gamma, self.final_beta)
            return 1202, state

        while True:
            print(f"i:{i}   state:{list(state.keys())}")
            if local_i == 1:  # LN1
                
                
                
                gamma = self.ln1_gamma[current_layer]
                beta = self.ln1_beta[current_layer]
                state['ln1'] = layer_norm(state['input'], gamma, beta)
                i += 1
                local_i += 1
                
            elif local_i ==3:
                # 返回后对KQV去混淆后再混淆
                st = time.time()
                #Qobf = self.ir1@self.obf1@(np.ones((1, self.hidden_size)))+self.ir1[:,self.perm1]
                #Kobf = self.ir1@self.obf1@(np.ones((1, self.hidden_size)))+self.ir1[:,self.perm1]
                #Vobf = self.ir1@self.obf1@(np.ones((1, self.hidden_size)))+self.ir1[:,self.perm1]
                obf1 = self.obf1.reshape(self.head_num, self.head_dim ,1)
                Q_reobf = state['Q']@obf1@(np.ones((1, self.head_dim)))+(state['Q']@self.D1)[:, :, :, self.perm3]
                K_reobf = state['K']@obf1@(np.ones((1, self.head_dim)))+(state['K']@self.D1)[:, :, :, self.perm3]
                V_reobf = state['V']@obf1@(np.ones((1, self.head_dim)))+(state['V']@self.D1)[:, :, :, self.perm3]
                Qobf = Q_reobf@obf1@(np.ones((1, self.head_dim)))+(Q_reobf@self.D1)[:, :, :, self.perm3]
                Kobf = K_reobf@obf1@(np.ones((1, self.head_dim)))+(K_reobf@self.D1)[:, :, :, self.perm3]
                Vobf = V_reobf@obf1@(np.ones((1, self.head_dim)))+(V_reobf@self.D1)[:, :, :, self.perm3]
                state['Q']=Qobf
                state['K']=Kobf
                state['V']=Vobf
                et = time.time()
                print(f"obf QKV cost {(et-st)*1000} ms")
                break
                

            elif local_i == 4:  # softmax
                st = time.time()
                K_ori = (state['K'].transpose(0, 2, 1, 3).reshape(1, self.seq_len, self.hidden_size)).transpose(0, 2, 1)
                Q_ori = (state['Q'].transpose(0, 2, 1, 3).reshape(1, self.seq_len, self.hidden_size)).transpose(0, 2, 1)
                
                K_reobf = self.D2@np.ones((self.seq_len, 1))@self.v1@K_ori@self.D2
                Q_reobf = self.D2@np.ones((self.seq_len, 1))@self.v1@Q_ori@self.D2
                
                score_reobf = (self.D2@(state['scores']-K_reobf-Q_reobf)@self.D2)[:, :, self.perm2]
                
                #state['attn'] = sp_softmax(score_reobf, axis=-1)
                attn = sp_softmax(score_reobf, axis=-1)
                one_expand = np.expand_dims(np.expand_dims(np.ones((self.seq_len, 1)), axis=0), axis=0)
                obf_expand = np.expand_dims(np.expand_dims(self.v2, axis=0), axis=0)
                attn_obf = np.matmul(np.matmul(attn, one_expand), obf_expand)
                state['attn'] = attn_obf
                
                et = time.time()
                print(f"obf Attention cost {(et-st)*1000} ms")
                
                i += 1
                local_i += 1
                break

            elif local_i == 5:
                # pos5
                st = time.time()

                V_ori = (state['V'].transpose(0, 2, 1, 3).reshape(1, self.seq_len, self.hidden_size))
                V_reobf = self.D2@np.ones((self.seq_len, 1))@self.v2@V_ori@self.D3
                Vp = V_reobf.reshape(1,self.seq_len, self.head_num, self.head_dim).transpose(0, 2, 1, 3)
                attn_ori = (state['attn'].transpose(0, 2, 1, 3).reshape(1, self.seq_len, self.head_num*self.seq_len))
                attn_reobf = self.D2@np.ones((self.seq_len, 1))@self.v2@attn_ori
                aout_reobf = (self.D2@(state['aout']-Vp))[:, :,  :, self.perm3]
                et = time.time()
                print(f"VW_0 cost {(et-st)*1000} ms")
                i += 1
                local_i += 1
                break
                
            elif local_i == 7:
                attn_out = state['attn_out']
                aout_reshape = state['aout'].transpose(0, 2, 1, 3).reshape(1, self.seq_len, self.hidden_size)
                
                attn_out_deobf = attn_out@self.D3[ :, self.perm1] - aout_reshape@self.v1.T@np.ones((1, self.hidden_size))@self.D3
                #print((attn_out_deobf).shape)
                state['attn_residual'] = state['input'] + attn_out_deobf
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
                
                gelu_deobf = x@self.D4[ :, self.perm4] - state['ln2']@self.v1.T@np.ones((1, self.hidden_size*4))@self.D4
                
                
                state['gelu'] = gelu_deobf * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gelu_deobf + 0.044715 * np.power(gelu_deobf, 3))))
                i += 1
                local_i += 1

            elif local_i == 12:
                # state['output'] = state['attn_residual'] + state['ff2'].
                ff2_out = state['ff2']
                
                ff2_out_deobf = ff2_out@self.D3[ :, self.perm1] - state['gelu']@self.v4.T@np.ones((1, self.hidden_size))@self.D3
                
                state['output'] = state['attn_residual'] + ff2_out_deobf
                i += 1
                local_i += 1
            
            else:
                break
        return i, state

def perform_inference(client, stub, input_text, collect_times=True):
    start_time = time.time() if collect_times else None
    input_ids = client.tokenizer(input_text, return_tensors="np")["input_ids"]  # (1, seq_len)
    seq_len = input_ids.shape[1]

    hidden = client.wte[input_ids] + client.wpe[np.arange(seq_len)]
    
    client.seq_len = hidden.shape[1]
    client.obf2 = np.random.rand(client.seq_len, 1)
    client.perm2 = permutation_from_key(key="perm2", n=client.seq_len)
    client.D2 = np.diag(np.random.randn(client.seq_len))
    client.v2 = np.random.rand(1, client.seq_len)

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
    
    # Warmup: 运行几次dummy推理来预热整个管道，包括通信（不收集时间）
    warmup_runs = 1  # 可根据需要调整
    #print(f"Performing {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        _ = perform_inference(client, stub, "Warmup input", collect_times=False)  # 用短输入预热
    #print("Warmup completed.")
    time_count = time_cost("warm up", time_count)
    # 实际运行并收集时间
    input_text = "The capital of France is"
    time_count = time_cost("Tokenizer", time_count)  # 原tokenizer时间（实际运行前）
    _ = perform_inference(client, stub, input_text, collect_times=True)

if __name__ == '__main__':
    #for i in range(0,10):
    #    print(f"--------------{i}------------------")
    run()