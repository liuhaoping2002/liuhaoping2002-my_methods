import torch
import torch.nn as nn
import os
import math
import warnings

# --- 1. 模型配置 (Model Configurations) ---
# 这些配置对应标准的 ViT-Base, ViT-Large, ViT-Huge
VIT_CONFIGS = {
    '86m':  {'img_size': 224, 'patch_size': 16, 'embed_dim': 768,  'depth': 12, 'num_heads': 12, 'mlp_ratio': 4},
    '307m': {'img_size': 224, 'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'mlp_ratio': 4},
    '632m': {'img_size': 224, 'patch_size': 14, 'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'mlp_ratio': 4}
    # 注意: 632M (ViT-H) 通常使用 patch_size=14
}

# --- 2. 手动实现组件 (Manual Components) ---

class PatchEmbed(nn.Module):
    """ 将图像切块并投影到 Embedding 维度 """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 使用 Conv2d 实现切块投影是最高效的方法
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, Embed_Dim, Grid_H, Grid_W) -> (B, Embed_Dim, Num_Patches)
        x = self.proj(x).flatten(2)
        # -> (B, Num_Patches, Embed_Dim)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    """ 多头自注意力机制 (Manual Multi-Head Self Attention) """
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 融合 QKV 投影以提高效率，匹配 timm 结构
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # 生成 Q, K, V
        # shape: (B, N, 3, Heads, Head_Dim) -> (3, B, Heads, N, Head_Dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention Score = (Q @ K.T) / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Output = Attn @ V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    """ 前馈神经网络 """
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        # Residual connection 1
        x = x + self.attn(self.norm1(x))
        # Residual connection 2
        x = x + self.mlp(self.norm2(x))
        return x

class ManualViT(nn.Module):
    """ 主 Vision Transformer 类 """
    def __init__(self, config_key='86m', num_classes=1000):
        super().__init__()
        cfg = VIT_CONFIGS[config_key]
        self.embed_dim = cfg['embed_dim']
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=cfg['img_size'], patch_size=cfg['patch_size'], embed_dim=cfg['embed_dim']
        )
        
        # 2. Class Token & Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg['embed_dim']))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, cfg['embed_dim']))
        
        # 3. Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            Block(dim=cfg['embed_dim'], num_heads=cfg['num_heads'], mlp_ratio=cfg['mlp_ratio'])
            for _ in range(cfg['depth'])
        ])
        
        # 4. Final Norm & Head
        self.norm = nn.LayerNorm(cfg['embed_dim'], eps=1e-6)
        self.head = nn.Linear(cfg['embed_dim'], num_classes)

        # 初始化权重 (简单的截断正态分布)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # Concatenate Class Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Position Embedding
        x = x + self.pos_embed
        
        # Blocks
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0] # 只取 Class Token 的输出

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# --- 3. 参数保存与加载工具 (Save/Load Utils) ---

def save_model_params(model, filepath):
    print(f"[*] Saving model parameters to {filepath}...")
    torch.save(model.state_dict(), filepath)

def load_model_params(model, filepath):
    print(f"[*] Loading model parameters from {filepath}...")
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 4. 验证与运行脚本 (Verification & Execution) ---

def verify_against_timm(manual_model, model_size_key):
    """
    为了验证手动实现的正确性，我们将从 timm (成熟库) 加载权重，
    复制到我们的模型中，然后对比输出。
    """
    try:
        import timm
    except ImportError:
        print("[!] Timm library not found. Skipping verification.")
        return

    # 映射配置名到 timm 的模型名
    timm_map = {
        '86m': 'vit_base_patch16_224',
        '307m': 'vit_large_patch16_224',
        '632m': 'vit_huge_patch14_224'
    }
    
    timm_name = timm_map[model_size_key]
    print(f"\n[Verification] Loading reference model: {timm_name} from timm...")
    
    # 加载预训练的 timm 模型 (pretrained=False 只是为了架构对比，也可以 True 来测试真实权重)
    # 这里我们使用 pretrained=False 初始化随机权重，然后同步给手动模型，验证计算逻辑一致性
    ref_model = timm.create_model(timm_name, pretrained=False)
    ref_model.eval()
    
    # --- 权重同步 (Weight Mapping) ---
    # 这是一个非常 tricky 的部分，因为命名习惯可能不同。
    # 幸运的是，我们的手动实现特意模仿了 timm 的结构。
    
    manual_state = manual_model.state_dict()
    ref_state = ref_model.state_dict()
    
    # 强制将 ref_model 的权重覆盖到 manual_model
    # 前提：我们代码中的层级命名必须和 timm 基本一致
    new_state_dict = {}
    for key in manual_state.keys():
        if key in ref_state:
            # 检查形状是否一致
            if manual_state[key].shape == ref_state[key].shape:
                new_state_dict[key] = ref_state[key]
            else:
                print(f"  [Warning] Shape mismatch for {key}: Manual {manual_state[key].shape} vs Ref {ref_state[key].shape}")
        else:
            print(f"  [Warning] Key missing in reference model: {key}")
            
    manual_model.load_state_dict(new_state_dict, strict=False)
    manual_model.eval()
    
    # --- 运行对比 ---
    input_tensor = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        out_manual = manual_model(input_tensor)
        out_ref = ref_model(input_tensor)
        
    diff = torch.abs(out_manual - out_ref).max().item()
    print(f"first 15 outputs Manual: {out_manual.flatten()[:15]}")
    print(f"first 15 outputs Timm:   {out_ref.flatten()[:15]}")
    print(f"[Verification] Max absolute difference between Manual and Timm: {diff:.8f}")
    
    if diff < 1e-5:
        print(">>> SUCCESS: Manual implementation matches mature implementation!")
    else:
        print(">>> WARNING: Outputs differ. Check layer initialization or epsilon values.")

def main():
    # 1. 选择模型规模
    selected_size = '86m' # 可选: '86m', '307m', '632m'
    weights_path = f"vit_{selected_size}_weights.pth"
    
    print(f"--- Initializing Manual ViT ({selected_size}) ---")
    model = ManualViT(config_key=selected_size)
    
    # 打印实际参数量
    total_params = count_parameters(model)
    print(f"Model Parameters: {total_params / 1e6:.2f} M")
    
    # 2. 先保存参数
    save_model_params(model, weights_path)
    
    # 3. 销毁模型，重新从本地加载参数进行计算
    del model
    print("Model deleted from memory.")
    
    model_loaded = ManualViT(config_key=selected_size)
    load_model_params(model_loaded, weights_path)
    model_loaded.eval()
    
    # 4. 进行一次推理计算
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model_loaded(dummy_input)
    print(f"Inference Output Shape: {output.shape}")
    
    # 5. (可选) 验证逻辑正确性
    # 注意：验证需要 timm 库，且会对比计算结果是否一致
    verify_against_timm(model_loaded, selected_size)

if __name__ == '__main__':
    main()