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
            entry['client_time'] = client_data[op_id]['time']
            entry['client_executor'] = client_data[op_id]['executor']
        if op_id in server_data:
            entry['server_time'] = server_data[op_id]['time']
            entry['server_executor'] = server_data[op_id]['executor']
        all_data[layer].append(entry)

    # 按层输出，按 local_i 排序
    for layer in sorted(all_data.keys()):
        print(f"Layer {layer}:")
        sorted_ops = sorted(all_data[layer], key=lambda x: x['local_i'])
        for op in sorted_ops:
            print(f"  op_id: {op['op_id']}")
            if 'client_time' in op:
                print(f"    Client ({op['client_executor']}): {op['client_time']:.2f} ms")
            if 'server_time' in op:
                print(f"    Server ({op['server_executor']}): {op['server_time']:.2f} ms")
        print("---")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python analyze_logs.py client.log server.log")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])