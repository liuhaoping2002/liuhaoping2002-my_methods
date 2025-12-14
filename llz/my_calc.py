

def calc_SOTER(mod_dim=768, seq_len=128, num_layers=12):
    print(f"SOTER Calculation - mod_dim: {mod_dim}, seq_len: {seq_len}")
    
    # KQV
    kqv_flops = 3 * (mod_dim * mod_dim) * seq_len * 2

    # Attention
    attn_flops = 2 * (seq_len * mod_dim * seq_len)

    # softmax
    softmax_flops = 5 * seq_len * seq_len

    # softmax@V
    attn_v_flops = 2 * (seq_len * mod_dim * seq_len)

    # Output projection
    out_proj_flops = 2 * mod_dim * mod_dim * seq_len

    # LN1
    ln1_flops = 5 * mod_dim * seq_len

    # FFN1
    ffn1_flops = 2 * (mod_dim * 4 * mod_dim) * seq_len

    # GELU
    gelu_flops = 4 * 4 * mod_dim * seq_len

    # FFN2
    ffn2_flops = 2 * (4 * mod_dim * mod_dim) * seq_len

    # LN2
    ln2_flops = 5 * mod_dim * seq_len

    total_flops_layer = (kqv_flops + attn_flops + softmax_flops + attn_v_flops +
                   out_proj_flops + ln1_flops + ffn1_flops + gelu_flops +
                   ffn2_flops + ln2_flops)    
    print(f"SOTER Total FLOPs per layer: {total_flops_layer}")

    total_flops = total_flops_layer * num_layers
    print(f"SOTER Total FLOPs for {num_layers} layers: {total_flops}")

    print(f"SOTER Total FLOPs per Layer (in billions): {total_flops_layer / 1e9:.5f} GFLOPs")
    print(f"SOTER Total FLOPs for {num_layers} Layers (in billions): {total_flops / 1e9:.5f} GFLOPs")
    return total_flops_layer

def calc(mod_dim=768, seq_len=128, num_layers=12):
    print(f"mod_dim: {mod_dim}, seq_len: {seq_len}")
    
    # KQV
    kqv_flops = 3 * (mod_dim * mod_dim) * seq_len * 2

    # Attention
    attn_flops = 2 * (seq_len * mod_dim * seq_len)

    # softmax
    softmax_flops = 5 * seq_len * seq_len

    # softmax@V
    attn_v_flops = 2 * (seq_len * mod_dim * seq_len)

    # Output projection
    out_proj_flops = 2 * mod_dim * mod_dim * seq_len

    # LN1
    ln1_flops = 5 * mod_dim * seq_len

    # FFN1
    ffn1_flops = 2 * (mod_dim * 4 * mod_dim) * seq_len

    # GELU
    gelu_flops = 4 * 4 * mod_dim * seq_len

    # FFN2
    ffn2_flops = 2 * (4 * mod_dim * mod_dim) * seq_len

    # LN2
    ln2_flops = 5 * mod_dim * seq_len

    total_flops_linear_layer = kqv_flops + attn_flops + attn_v_flops + out_proj_flops + ffn1_flops + ffn2_flops
    total_flops_nonlinear_layer = softmax_flops + ln1_flops + gelu_flops + ln2_flops
    SOTER_TEE_flops = total_flops_linear_layer * 0.2
    total_flops_layer = total_flops_linear_layer + total_flops_nonlinear_layer
    print(f"Total FLOPs per layer: {total_flops_layer}")

    total_flops = total_flops_layer * num_layers
    print(f"Total FLOPs for {num_layers} layers: {total_flops}")

    print(f"Total FLOPs per Layer (in billions): {total_flops_layer / 1e9:.5f} GFLOPs")
    print(f"Total FLOPs for {num_layers} Layers (in billions): {total_flops / 1e9:.5f} GFLOPs")
    return total_flops_layer

if __name__ == '__main__':
    calc()
    calc_SOTER()