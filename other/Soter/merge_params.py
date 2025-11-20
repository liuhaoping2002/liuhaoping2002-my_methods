import numpy as np

# 1. 读取原始参数
params = np.load('gpt2_params/params.npz')

# 把里面的所有键值取出来
params_dict = {key: params[key] for key in params.files}

# 2. 读取 c_proj_x 参数
server_params = np.load('gpt2_server_params.npz')

c_proj_w = server_params['c_proj_w']
c_proj_b = server_params['c_proj_b']

# 3. 合并进字典
params_dict['c_proj_w'] = c_proj_w
params_dict['c_proj_b'] = c_proj_b

# 4. 统一保存为新的 npz 文件
np.savez('gpt2_params/params.npz', **params_dict)

print("合并完成！")
