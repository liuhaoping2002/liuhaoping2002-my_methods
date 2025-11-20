# save_params.py
import os
import numpy as np
from transformers import AutoConfig, GPT2Model, AutoTokenizer

def save_params():
    # 创建保存目录
    os.makedirs('gpt2_params', exist_ok=True)
    
    # 加载配置和模型
    config = AutoConfig.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 准备数据：栈列表参数
    ln1_gamma = np.stack([layer.ln_1.weight.detach().cpu().numpy() for layer in model.h])  # (12, 768)
    ln1_beta = np.stack([layer.ln_1.bias.detach().cpu().numpy() for layer in model.h])
    ln2_gamma = np.stack([layer.ln_2.weight.detach().cpu().numpy() for layer in model.h])
    ln2_beta = np.stack([layer.ln_2.bias.detach().cpu().numpy() for layer in model.h])
    
    # 保存所有参数到一个.npz文件
    np.savez('gpt2_params/params.npz',
             n_layer=np.array([config.n_layer]),
             ln1_gamma=ln1_gamma,
             ln1_beta=ln1_beta,
             ln2_gamma=ln2_gamma,
             ln2_beta=ln2_beta,
             final_gamma=model.ln_f.weight.detach().cpu().numpy(),
             final_beta=model.ln_f.bias.detach().cpu().numpy(),
             wte=model.wte.weight.detach().cpu().numpy(),
             wpe=model.wpe.weight.detach().cpu().numpy())
    
    # 保存 tokenizer 到目录
    tokenizer.save_pretrained('gpt2_params/tokenizer')

if __name__ == '__main__':
    save_params()
    print("Parameters saved: 'gpt2_params/params.npz' and 'gpt2_params/tokenizer/' directory.")