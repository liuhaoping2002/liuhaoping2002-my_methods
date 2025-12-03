import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import os
import tqdm
import numpy as np

# --- 1. 配置与模型 ID ---
# 为了测试准确率，我们必须使用在该任务上微调过的模型
# 这里以 110M (Bert-Base) 为例，使用 textattack/bert-base-uncased-SST-2
MODEL_CONFIGS = {
    '110m': {
        'hf_id': 'textattack/bert-base-uncased-SST-2', # 这是一个在 SST-2 上微调好的 Base 模型
        'hidden_size': 768, 'num_layers': 12, 'heads': 12, 'vocab': 30522
    },
    # 注意：4m/41m/336m 如果没有公开的 SST-2 微调权重，
    # 只能加载架构，准确率会是随机猜测 (约 50%)。
    # 此处代码通用，只要你有对应的权重文件即可。
}

# --- 2. 手动实现的 BERT (包含分类头) ---

# 复用之前的 ManualBertModel 组件（为了节省篇幅，这里假设已导入/定义）
# 必须确保 ManualBertModel 的命名与 HF BertModel 内部完全一致
# 这里重新简写核心部分以展示 Classification Head 的集成

class ManualBertEmbeddings(nn.Module):
    # ... (与之前相同，略以节省篇幅，实际运行时需包含) ...
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        # 简化版前向
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None: token_type_ids = torch.zeros_like(input_ids)
        
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids) + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.LayerNorm(embeddings))

class ManualBertEncoder(nn.Module):
    # ... (完全复用 HF 结构) ...
    def __init__(self, config):
        super().__init__()
        # 使用 ModuleList 模拟 BERT 层
        from transformers.models.bert.modeling_bert import BertLayer
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            # HF layer output is a tuple (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
        return hidden_states

class ManualBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ManualBertEmbeddings(config)
        self.encoder = ManualBertEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

    def get_extended_attention_mask(self, attention_mask):
        # 处理 mask (0/1 -> 0/-10000)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None: attention_mask = torch.ones_like(input_ids)
        
        ext_mask = self.get_extended_attention_mask(attention_mask)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, ext_mask)
        
        # Pooler logic
        first_token_tensor = encoder_outputs[:, 0]
        pooled_output = self.pooler_activation(self.pooler(first_token_tensor))
        return encoder_outputs, pooled_output

class ManualBertForSequenceClassification(nn.Module):
    """
    手动实现的带分类头的 BERT。
    结构对应 transformers.BertForSequenceClassification
    """
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        # 关键：命名必须叫 self.bert 以匹配 HF 权重
        self.bert = ManualBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 1. Base Model Forward
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1] # 取 Pooler output

        # 2. Classification Head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# --- 3. 工具函数：下载与加载 ---

def download_and_save_weights(model_id, local_path):
    if os.path.exists(local_path):
        print(f"[*] Local weights found at {local_path}")
        return
    
    print(f"[*] Downloading fine-tuned weights from {model_id}...")
    # 下载带分类头的模型
    model = BertForSequenceClassification.from_pretrained(model_id)
    torch.save(model.state_dict(), local_path)
    print(f"[*] Saved to {local_path}")

def load_manual_model(local_path, config):
    print("[*] Instantiating Manual Model...")
    model = ManualBertForSequenceClassification(config)
    
    print(f"[*] Loading weights from {local_path}...")
    state_dict = torch.load(local_path)
    
    # HF 的 Pooler 权重通常在 state_dict 里叫 'bert.pooler.dense.weight'
    # 我们的 ManualBertModel 里叫 'bert.pooler.weight' (如果是 nn.Linear)
    # 这是一个常见的命名差异点，这里做一个简单的修正逻辑
    new_state_dict = {}
    for k, v in state_dict.items():
        if "bert.pooler.dense" in k:
            new_key = k.replace("bert.pooler.dense", "bert.pooler")
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    # 加载
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if len(missing) > 0:
        print(f"[Warning] Missing keys: {missing[:5]}...") # 通常不应该缺失关键权重
    model.eval()
    return model

# --- 4. 评估循环 (Evaluation Loop) ---

def evaluate_accuracy_and_diff(manual_model, hf_model, dataset, tokenizer, device='cpu'):
    manual_model.to(device)
    hf_model.to(device)
    hf_model.eval()
    manual_model.eval()

    total = 0
    correct_manual = 0
    correct_hf = 0
    max_diff_logits = 0.0

    print("\n=== Starting Evaluation on SST-2 Validation Set ===")
    
    # 仅测试前 100 条以快速验证 (全量测试需去掉 slice)
    subset_data = dataset.select(range(100)) 
    
    for example in tqdm.tqdm(subset_data):
        inputs = tokenizer(example['sentence'], return_tensors="pt", padding='max_length', max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label = example['label']

        with torch.no_grad():
            # 1. HF Forward
            hf_outputs = hf_model(**inputs)
            hf_logits = hf_outputs.logits
            
            # 2. Manual Forward
            man_logits = manual_model(inputs['input_ids'], inputs['attention_mask'])

            # 3. 误差统计
            diff = (man_logits - hf_logits).abs().max().item()
            if diff > max_diff_logits:
                max_diff_logits = diff

            # 4. 准确率统计
            hf_pred = torch.argmax(hf_logits, dim=-1).item()
            man_pred = torch.argmax(man_logits, dim=-1).item()
            
            if hf_pred == label: correct_hf += 1
            if man_pred == label: correct_manual += 1
            total += 1

    acc_hf = correct_hf / total
    acc_man = correct_manual / total
    
    print("\n=== Evaluation Results ===")
    print(f"Total Samples Tested: {total}")
    print(f"Max Logits Error (Manual vs HF): {max_diff_logits:.8f}")
    print(f"HF Official Accuracy:     {acc_hf:.2%}")
    print(f"Manual Impl Accuracy:     {acc_man:.2%}")
    
    if abs(acc_hf - acc_man) < 1e-4 and max_diff_logits < 1e-4:
        print(">>> SUCCESS: Accuracy matches perfectly!")
    else:
        print(">>> WARNING: Discrepancy detected.")

# --- Main ---

def main():
    target_size = '110m' # 这里使用 Base (110M) 因为它有现成的 SST-2 权重
    config_info = MODEL_CONFIGS[target_size]
    hf_model_id = config_info['hf_id']
    weights_path = f"bert_{target_size}_sst2.pth"

    # 1. 准备数据 (SST-2)
    print("[*] Loading SST-2 dataset...")
    # GLUE benchmark 的 SST-2 子集
    dataset = load_dataset("glue", "sst2", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    # 2. 保存权重到本地 (模拟用户下载过程)
    download_and_save_weights(hf_model_id, weights_path)

    # 3. 加载 HF 官方模型 (作为对照组)
    print(f"[*] Loading Reference HF Model: {hf_model_id}")
    hf_model = BertForSequenceClassification.from_pretrained(hf_model_id)

    # 4. 加载手动模型 (从本地文件)
    config = BertConfig.from_pretrained(hf_model_id) # 获取对应的 Config 结构
    manual_model = load_manual_model(weights_path, config)

    # 5. 运行对比测试
    evaluate_accuracy_and_diff(manual_model, hf_model, dataset, tokenizer, device='cpu')

if __name__ == "__main__":
    main()