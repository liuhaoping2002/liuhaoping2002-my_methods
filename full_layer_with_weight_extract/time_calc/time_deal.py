import re
import sys

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

    # 按层处理 0-11 层
    for layer in range(12):  # 0-11
        if layer not in all_data:
            continue
        print(f"Layer {layer}:")
        sorted_ops = sorted(all_data[layer], key=lambda x: x['local_i'])
        
        i = 0
        while i < len(sorted_ops):
            op = sorted_ops[i]
            op_id = op['op_id']
            local_i = op['local_i']
            
            if 'client_time' in op and op['client_executor'] == 'server' and 'server_time' in op:
                # 同时有 Client (server) 和 Server (server)
                op_time = op['server_time']
                print(f"  op_id {op_id} (local_i {local_i}):\n    算子用时 = {op_time:.2f} ms")
                
                # 计算通信开销
                rtt = op['client_time']
                subtract = op_time  # 减去当前 server
                j = i + 1
                while j < len(sorted_ops):
                    next_op = sorted_ops[j]
                    if 'client_time' in next_op and next_op['client_executor'] == 'client' and 'server_time' not in next_op:
                        # 遇到仅含 Client (client)，停下，不减
                        break
                    if 'server_time' in next_op:
                        subtract += next_op['server_time']
                    j += 1
                
                overhead = rtt - subtract
                print(f"    通信开销 = {overhead:.2f} ms")
            else:
                # 其他情况，直接打印
                print(f"  op_id {op_id} (local_i {local_i}):")
                if 'client_time' in op:
                    print(f"    Client ({op['client_executor']}): {op['client_time']:.2f} ms")
                if 'server_time' in op:
                    print(f"    Server: {op['server_time']:.2f} ms")
            
            i += 1
        print("---")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python analyze_logs.py client.log server.log")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])