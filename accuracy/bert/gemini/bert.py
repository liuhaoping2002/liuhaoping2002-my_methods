import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json

import torch
import os
from transformers import BertModel, BertConfig

# ------------------ 新增：HF 模型 ID 映射 ------------------
HF_MODEL_MAP = {
    '110m': 'bert-base-uncased',
    '336m': 'bert-large-uncased',
    # 4M/41M 没有标准对应，若要验证，可使用随机权重或近似模型
    '41m': 'google/bert_uncased_L-6_H-512_A-8', # 41M closest public model, requires verification
    '4m': 'google/bert_uncased_L-2_H-128_A-2' # TinyBERT-like
}

# --- 新增：预训练参数下载和保存工具 ---

def download_and_save_pretrained_weights(size_key, weights_path):
    """
    下载指定的 HF 预训练模型，并将其权重保存到本地。
    """
    if size_key not in HF_MODEL_MAP:
        print(f"[Warning] Size {size_key} is non-standard. Using random weights for verification.")
        return False
        
    hf_model_id = HF_MODEL_MAP[size_key]
    if os.path.exists(weights_path):
        print(f"[*] Local pretrained weights already exist at {weights_path}. Skipping download.")
        return True

    print(f"[*] Downloading pre-trained BERT weights ({hf_model_id}) for {size_key}...")
    try:
        # 使用 from_pretrained 下载模型和权重
        ref_model = BertModel.from_pretrained(hf_model_id)
        torch.save(ref_model.state_dict(), weights_path)
        print(f"[SUCCESS] Weights saved to {weights_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download pre-trained weights: {e}")
        # 如果下载失败，退回到随机权重
        return False

# --- 1. 模型配置 (Model Configurations) ---
# 定义精确的参数以匹配要求的规模
# 4M: Tiny, 41M: Small/Medium mix, 110M: Base, 336M: Large
BERT_CONFIGS = {
    '4m':   {'hidden_size': 128,  'num_hidden_layers': 2,  'num_attention_heads': 2,  'intermediate_size': 512,  'vocab_size': 30522},
    '41m':  {'hidden_size': 512,  'num_hidden_layers': 6,  'num_attention_heads': 8,  'intermediate_size': 2048, 'vocab_size': 30522},
    '110m': {'hidden_size': 768,  'num_hidden_layers': 12, 'num_attention_heads': 12, 'intermediate_size': 3072, 'vocab_size': 30522},
    '336m': {'hidden_size': 1024, 'num_hidden_layers': 24, 'num_attention_heads': 16, 'intermediate_size': 4096, 'vocab_size': 30522}
}

# --- 2. 手动实现组件 (Manual Components) ---
# 注意：所有类名和变量名尽量对齐 Hugging Face 源码，以便权重自动匹配

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=0)
        self.position_embeddings = nn.Embedding(512, config['hidden_size']) # Max len 512
        self.token_type_embeddings = nn.Embedding(2, config['hidden_size']) # Segment IDs

        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['hidden_size'] % config['num_attention_heads'] != 0:
            raise ValueError("Hidden size not divisible by attention heads")
            
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / config['num_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # BERT use separated Linear layers, unlike ViT's fused
        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        # (B, Seq, Hidden) -> (B, Seq, Heads, HeadDim) -> (B, Heads, Seq, HeadDim)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Attention scores = Q * K^T / sqrt(d)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the mask (adding large negative number to masked positions)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        # (B, Heads, Seq, HeadDim) -> (B, Seq, Heads, HeadDim) -> (B, Seq, Hidden)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # Residual
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config['num_hidden_layers'])])

    def forward(self, hidden_states, attention_mask=None):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ManualBertModel(nn.Module):
    """
    手动实现的 BERT 主类。
    结构完全对齐 Hugging Face Transformers 的 BertModel。
    """
    def __init__(self, config_key='110m'):
        super().__init__()
        self.config_dict = BERT_CONFIGS[config_key]
        
        self.embeddings = BertEmbeddings(self.config_dict)
        self.encoder = BertEncoder(self.config_dict)
        self.pooler = BertPooler(self.config_dict)
        
        # 初始化权重 (简单的截断正态分布，实际运行时会被 load_state_dict 覆盖)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_extended_attention_mask(self, attention_mask, dtype):
        """
        处理 Attention Mask。
        Input Mask: 1.0 for keep, 0.0 for mask.
        Output: 0.0 for keep, -10000.0 for mask (added to attention scores).
        """
        # (B, Seq) -> (B, 1, 1, Seq)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask.to(dtype=dtype)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        input_ids: (Batch, Seq_Len)
        attention_mask: (Batch, Seq_Len), 1=valid, 0=pad
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 1. Mask Preprocessing (Crucial for Verification!)
        # HF models convert 0/1 mask to large negative numbers internally
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, self.embeddings.word_embeddings.weight.dtype)

        # 2. Main Flow
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoder_outputs)

        return encoder_outputs, pooled_output

# --- 3. 参数保存与加载 ---

def save_model(model, path):
    print(f"[*] Saving model params to {path}")
    torch.save(model.state_dict(), path)

def load_model(model, path):
    print(f"[*] Loading model params from {path}")
    state_dict = torch.load(path)
    # 因为我们的命名与 HF 一致，直接加载即可
    keys = model.load_state_dict(state_dict, strict=False)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 4. 验证逻辑 (Verification) ---

def verify_implementation(manual_model, size_key):
    """
    通过实例化一个 Hugging Face BertModel，复制权重，对比输出。
    """
    try:
        from transformers import BertConfig, BertModel
    except ImportError:
        print("Transformers library not installed.")
        return

    print(f"\n--- Verifying {size_key.upper()} against Hugging Face ---")
    
    cfg_dict = BERT_CONFIGS[size_key]
    
    # 1. 创建 HF 参考模型 (Reference Model)
    hf_config = BertConfig(
        vocab_size=cfg_dict['vocab_size'],
        hidden_size=cfg_dict['hidden_size'],
        num_hidden_layers=cfg_dict['num_hidden_layers'],
        num_attention_heads=cfg_dict['num_attention_heads'],
        intermediate_size=cfg_dict['intermediate_size'],
        hidden_act="gelu", # Manual uses GELU
        max_position_embeddings=512,
        type_vocab_size=2,
    )
    ref_model = BertModel(hf_config)
    
    local_weights_path = "./bert_110m_pretrained.pth"
    state_dict = torch.load(local_weights_path)

    # 加载权重到模型
    ref_model.load_state_dict(state_dict)
    
    ref_model.eval()

    # 2. 权重同步 (Weight Sync)
    # 我们的命名完全镜像了 HF，所以可以直接使用 ref_model 的 state_dict
    # 但需要注意：HF 的 keys 可能包含 "bert." 前缀 (如果从 AutoModel 加载)，或者没有 (如果直接用 BertModel)
    # 这里我们直接用 BertModel，所以 keys 应该匹配。
    
    ref_state_dict = ref_model.state_dict()
    manual_model.load_state_dict(ref_state_dict, strict=True)
    manual_model.eval()

    # 3. 构造输入
    input_ids = torch.randint(0, cfg_dict['vocab_size'], (1, 32)) # Batch=1, Seq=32
    mask = torch.ones((1, 32))
    # 模拟 Padding：把最后 5 个设为 0
    mask[:, -5:] = 0
    
    # 4. 推理
    with torch.no_grad():
        # HF Output
        ref_out = ref_model(input_ids, attention_mask=mask)
        ref_last_hidden = ref_out.last_hidden_state
        ref_pooled = ref_out.pooler_output
        
        # Manual Output
        man_last_hidden, man_pooled = manual_model(input_ids, attention_mask=mask)

    # 5. 比较误差
    diff_hidden = (man_last_hidden - ref_last_hidden).abs().max().item()
    diff_pooled = (man_pooled - ref_pooled).abs().max().item()

    print(f"Diff Last Hidden: {diff_hidden:.8f}")
    print(f"Diff Pooled:      {diff_pooled:.8f}")

    if diff_hidden < 1e-5:
        print(">>> SUCCESS: Verification Passed!")
    else:
        print(">>> WARNING: Verification Failed. Check eps, gelu, or mask logic.")

# --- Main ---
def main():
    target_size = '110m' # 选择一个需要下载预训练权重的规模
    weights_path = f"bert_{target_size}_pretrained.pth"
    
    # 步骤 1: 确保预训练权重已下载并保存到本地
    #download_success = download_and_save_pretrained_weights(target_size, weights_path)
    download_success = True
    # 如果下载成功，则使用下载的权重进行后续操作
    if download_success:
        print(f"\n=== Loading Pre-trained BERT Model ({target_size}) ===")
        
        # 步骤 2: 初始化手动模型并加载本地保存的权重
        loaded_model = ManualBertModel(target_size)
        
        # 严格加载预训练参数
        load_model(loaded_model, weights_path) 
        loaded_model.eval()
        
        # 步骤 3: 验证（使用我们本地加载的预训练权重与 HF 的同一份模型进行对比）
        # 注意：这里 verify_implementation 需要修改为使用 HF_MODEL_MAP 中的 ID 再次实例化模型进行对比
        verify_implementation(loaded_model, target_size)
        
    else:
        print("\nVerification skipped due to download failure or non-standard size.")
    
    # 2. 初始化 & 保存 (模拟先有参数)
    model = ManualBertModel(target_size)
    print(f"Parameters: {count_parameters(model) / 1e6:.2f} M")
    
    save_model(model, weights_path)
    del model # 清空内存
    
    # 3. 加载 & 计算
    loaded_model = ManualBertModel(target_size)
    load_model(loaded_model, weights_path)
    loaded_model.eval()
    
    # 演示计算
    dummy_input = torch.randint(0, 30522, (1, 128))
    dummy_mask = torch.ones((1, 128))
    out_seq, out_pool = loaded_model(dummy_input, attention_mask=dummy_mask)
    print(f"Calculation Output Shape: {out_seq.shape}")

    # 4. 验证 (Verify)
    verify_implementation(loaded_model, target_size)

import random
import numpy as np

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU的话
    #torch.backends.cudnn.deterministic = True  # 确保卷积操作的确定性
    #torch.backends.cudnn.benchmark = False  # 禁用 `cudnn.benchmark` 以避免非确定性行为
    main()