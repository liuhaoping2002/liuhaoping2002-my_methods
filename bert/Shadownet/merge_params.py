import numpy as np
import os
import shutil

def merge_server_weights_to_client():
    # 路径配置
    SERVER_PARAMS_PATH = 'bert_server_params.npz'
    CLIENT_PARAMS_DIR = 'bert_params_split'
    
    print(f"正在读取服务端总参数文件: {SERVER_PARAMS_PATH} ...")
    if not os.path.exists(SERVER_PARAMS_PATH):
        print("错误：未找到服务端参数文件。请确保 bert_server_params.npz 存在。")
        return

    # 1. 加载服务端大文件
    server_data = np.load(SERVER_PARAMS_PATH)
    
    # 获取层数
    try:
        n_layer = int(server_data['n_layer'][0])
    except:
        n_layer = 12 # 默认 GPT-2 Small
    
    print(f"检测到模型层数: {n_layer}")
    
    # 2. 遍历每一层进行合并
    for i in range(n_layer):
        client_layer_file = os.path.join(CLIENT_PARAMS_DIR, f'layer_{i}.npz')
        
        if not os.path.exists(client_layer_file):
            print(f"警告: 客户端文件 {client_layer_file} 不存在，跳过。")
            continue
            
        print(f"正在处理第 {i} 层 -> {client_layer_file}")
        
        # 3. 读取现有的客户端参数 (LN gamma/beta 等)
        # 注意：np.load 返回的对象是不可变的，我们需要将其转化为 dict
        with np.load(client_layer_file) as existing_data:
            layer_dict = dict(existing_data)
        
        # 4. 从服务端数据中提取当前层的权重，并注入到字典中
        # 这些是 Stateless Server 模式下客户端需要发送给服务端的权重
        
        # QKV 权重 (Attention Inputs)
        layer_dict['c_attn_w'] = server_data['c_attn_w'][i]
        layer_dict['c_attn_b'] = server_data['c_attn_b'][i]
        
        # Attn Output Projection 权重 (之前您决定移回TEE计算的)
        layer_dict['c_proj_w'] = server_data['c_proj_w'][i]
        layer_dict['c_proj_b'] = server_data['c_proj_b'][i]
        
        # MLP Up Projection (c_fc)
        layer_dict['mlp_c_fc_w'] = server_data['mlp_c_fc_w'][i]
        layer_dict['mlp_c_fc_b'] = server_data['mlp_c_fc_b'][i]
        
        # MLP Down Projection (c_proj)
        layer_dict['mlp_c_proj_w'] = server_data['mlp_c_proj_w'][i]
        layer_dict['mlp_c_proj_b'] = server_data['mlp_c_proj_b'][i]
        
        # 5. 覆盖保存回原文件
        # 使用 savez_compressed 可以减小体积，或者 savez
        np.savez(client_layer_file, **layer_dict)
        
    print("\n合并完成！")
    print("现在客户端的 layer_x.npz 文件已包含所有计算所需的权重。")
    print("您可以运行 client_grpc_secure_profiled.py 了。")

if __name__ == "__main__":
    # 为了安全，建议先备份 bert_params_split 文件夹
    if os.path.exists('bert_params_split'):
        if not os.path.exists('bert_params_split_backup'):
            print("正在创建参数备份 (bert_params_split_backup)...")
            shutil.copytree('bert_params_split', 'bert_params_split_backup')
    
    merge_server_weights_to_client()