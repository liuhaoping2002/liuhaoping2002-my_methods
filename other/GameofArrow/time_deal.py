import re
import sys
import csv

# 在 main() 函数中，parse_table 之后、处理 all_data 之前添加：
OP_NAME_MAP = {
    1: "LayerNorm1",
    2: "QKV",
    3: "QK^T",
    4: "Softmax",
    5: "*V",
    6: "*Wo",
    7: "Residual_Attn",
    8: "LayerNorm2",
    9: "FFN1",
    10: "GELU",
    11: "FFN2",
    12: "Residual_FFN"
}

def parse_table(log_content, is_client=False):
    data = {}
    lines = log_content.strip().split('\n')
    start_parsing = False
    for line in lines:
        if '---' in line or '-' * 30 in line:  # 匹配分隔线
            start_parsing = True
            continue
        if start_parsing and '|' in line:
            parts = [p.strip() for p in re.split(r'\s*\|\s*', line) if p.strip()]
            if len(parts) < 3:
                continue
            try:
                op_id = int(parts[0])
                time_ms = float(parts[2])
                data[op_id] = {'time': time_ms, 'executor': parts[1]}
            except ValueError:
                continue  # 跳过无效行
    return data

def main(client_log_file, server_log_file):
    with open(client_log_file, 'r') as f:
        client_content = f.read()
    with open(server_log_file, 'r') as f:
        server_content = f.read()

    client_data = parse_table(client_content, is_client=True)
    server_data = parse_table(server_content)

    # 合并所有数据
    all_data = {}
    for op_id in set(list(client_data.keys()) + list(server_data.keys())):
        layer = op_id // 100
        local_i = op_id % 100
        if layer not in all_data:
            all_data[layer] = []
        entry = {'op_id': op_id, 'local_i': local_i}
        if op_id in client_data:
            entry['client_executor'] = client_data[op_id]['executor']
            entry['client_time'] = client_data[op_id]['time']
        if op_id in server_data:
            entry['server_time'] = server_data[op_id]['time']
        all_data[layer].append(entry)

    # 准备CSV数据列表
    csv_data = []
    csv_data.append(['Layer', 'ID/Type', 'Time (ms)'])

    all_layers_total = 0.0  # 所有层总用时
    all_layers = []
    
    # 按层处理 0-11 层
    for layer in range(12):  # 0-11
        if layer not in all_data:
            continue
        print(f"Layer {layer}:")
        sorted_ops = sorted(all_data[layer], key=lambda x: x['local_i'])
        
        i = 0
        
        layer_total = 0.0
        while i < len(sorted_ops):
            
            op = sorted_ops[i]
            op_id = op['op_id']
            local_i = op['local_i']
            
            if 'client_time' in op and op['client_executor'] == 'server' and 'server_time' in op:
                # 同时有 Client (server) 和 Server (server)
                op_time = op['server_time']
                layer_total += op_time
                print(f"  op {op_id} {OP_NAME_MAP[op_id%100]} :\n    Server: {op_time:.2f} ms")
                
                csv_data.append([layer, f"{OP_NAME_MAP[op_id%100]}", op_time])
                
                # 计算通信开销
                rtt = op['client_time']
                subtract = op_time
                j = i + 1
                while j < len(sorted_ops):
                    next_op = sorted_ops[j]
                    if 'client_time' in next_op and next_op['client_executor'] == 'client' and 'server_time' not in next_op:
                        break
                    if 'server_time' in next_op:
                        subtract += next_op['server_time']
                    j += 1
                
                overhead = rtt - subtract
                layer_total += overhead
                print(f"    Commun: {overhead:.2f} ms")
                
                # 添加到CSV: 通信开销，用时
                csv_data.append([layer, "Communication", overhead])
            
            else:
                # 其他情况，直接打印并累加
                print(f"  op {op_id} {OP_NAME_MAP[op_id%100]}:")
                if 'client_time' in op:
                    print(f"    Client: {op['client_time']:.2f} ms")
                    layer_total += op['client_time']
                    csv_data.append([layer, f"{OP_NAME_MAP[op_id%100]}", op['client_time']])
                if 'server_time' in op:
                    print(f"    Server: {op['server_time']:.2f} ms")
                    layer_total += op['server_time']
                    csv_data.append([layer, f"{OP_NAME_MAP[op_id%100]}", op['server_time']])
            
            i += 1
        
        # 输出每层总用时
        print(f"Layer {layer} 总用时: {layer_total:.3f} ms")
        print("---")
        
        # 添加到CSV
        #csv_data.append([layer, "Layer Total", layer_total])
        all_layers.append([layer, "Layer Total", layer_total])
        all_layers_total += layer_total

    # 输出所有层总用时

    for per_layers in all_layers:
        csv_data.append(per_layers)
    print(f"所有层总用时: {all_layers_total:.3f} ms")
    
    # 添加到CSV
    csv_data.append(["All Layers", "Total", all_layers_total])

    # 保存到CSV文件
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python time_deal.py client.log server.log")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])