# split_params_for_sgx.py
import numpy as np
import os

def split_params():
    # 假设原文件在 gpt2_params/params.npz
    if not os.path.exists('gpt2_params/params.npz'):
        print("Error: gpt2_params/params.npz not found.")
        return

    print("Loading original params...")
    data = np.load('gpt2_params/params.npz')
    n_layer = int(data['n_layer'][0])
    
    output_dir = 'gpt2_params_split'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存 Embedding (最大的一块)
    print("Saving embeddings...")
    np.savez_compressed(f'{output_dir}/embeddings.npz', 
                        wte=data['wte'], 
                        wpe=data['wpe'])

    # 2. 保存每一层的参数
    for i in range(n_layer):
        print(f"Saving layer {i}...")
        np.savez_compressed(f'{output_dir}/layer_{i}.npz',
                            ln1_gamma=data['ln1_gamma'][i],
                            ln1_beta=data['ln1_beta'][i],
                            ln2_gamma=data['ln2_gamma'][i],
                            ln2_beta=data['ln2_beta'][i])

    # 3. 保存 Final LayerNorm
    print("Saving final layer...")
    np.savez_compressed(f'{output_dir}/final.npz',
                        final_gamma=data['final_gamma'],
                        final_beta=data['final_beta'])
    
    print(f"Done! Split parameters saved to '{output_dir}/'")

if __name__ == '__main__':
    split_params()