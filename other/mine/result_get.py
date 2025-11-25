import re
import csv
from prettytable import PrettyTable


# ---------------------------
# 解析 tee_client.log
# ---------------------------
def parse_client_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[::-1]

    total_list = []
    obf_list = []
    grpc_list = []

    re_total = re.compile(r"Total Compute\s+([\d\.]+)")
    re_obf = re.compile(r"obf\s+([\d\.]+)")
    re_grpc = re.compile(r"gRPC Call\s+([\d\.]+)")

    cur_total = cur_obf = cur_grpc = None

    for line in lines:
        if cur_total is None:
            m = re_total.search(line)
            if m:
                cur_total = float(m.group(1))
                continue

        if cur_obf is None:
            m = re_obf.search(line)
            if m:
                cur_obf = float(m.group(1))
                continue

        if cur_grpc is None:
            m = re_grpc.search(line)
            if m:
                cur_grpc = float(m.group(1))
                continue

        if cur_total is not None and cur_obf is not None and cur_grpc is not None:
            total_list.append(cur_total)
            obf_list.append(cur_obf)
            grpc_list.append(cur_grpc)
            cur_total = cur_obf = cur_grpc = None

    return total_list, obf_list, grpc_list


# ---------------------------
# 解析 tee_server.log
# ---------------------------
def parse_server_log(file_path, expected_count):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[::-1]

    total_list = []
    grpc_total_list = []

    re_total = re.compile(r"Total Compute\s+([\d\.]+)")
    re_grpc_total = re.compile(r"gRPC Total\s+([\d\.]+)")

    cur_total = cur_grpc = None

    for line in lines:
        if cur_total is None:
            m = re_total.search(line)
            if m:
                cur_total = float(m.group(1))
                continue

        if cur_grpc is None:
            m = re_grpc_total.search(line)
            if m:
                cur_grpc = float(m.group(1))
                continue

        if cur_total is not None and cur_grpc is not None:
            total_list.append(cur_total)
            grpc_total_list.append(cur_grpc)
            cur_total = cur_grpc = None

        if len(total_list) >= expected_count:
            break

    return total_list, grpc_total_list


# ---------------------------
# 主逻辑：计算、打印、保存 CSV
# ---------------------------
def process_and_output(client_t, client_o, client_g, server_t, server_g, out_csv="result.csv"):
    table = PrettyTable()
    table.field_names = [
        "Idx",
        "Client Total", "obf", "Client gRPC",
        "Server Total", "Server gRPC",
        "comm",
        "total time",
        "%client_total", "%obf", "%server_total", "%comm"
    ]

    rows = []

    for i in range(len(client_t)):
        comm = client_g[i] - server_g[i]
        total_time = client_t[i] + client_o[i] + server_t[i] + comm

        pct_client = client_t[i] / total_time * 100
        pct_obf = client_o[i] / total_time * 100
        pct_server = server_t[i] / total_time * 100
        pct_comm = comm / total_time * 100

        row = [
            i + 1,
            client_t[i], client_o[i], client_g[i],
            server_t[i], server_g[i],
            comm,
            total_time,
            pct_client, pct_obf, pct_server, pct_comm
        ]
        rows.append(row)
        table.add_row(row)

    # 计算平均值
    import numpy as np
    avg_row = ["AVG"] + list(np.mean(np.array(rows)[:, 1:], axis=0))
    table.add_row(avg_row)

    print("\n========== 最终数据表格 ==========\n")
    print(table)

    # 保存 CSV
    with open(out_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(table.field_names)
        writer.writerows(rows)
        writer.writerow(avg_row)

    print(f"\nCSV 已保存为 {out_csv}\n")


# ---------------------------
# 程序入口
# ---------------------------
if __name__ == "__main__":
    client_t, client_o, client_g = parse_client_log("tee_client.log")
    print(f"解析 tee_client.log 得到 {len(client_t)} 组数据")

    server_t, server_g = parse_server_log("tee_server.log", len(client_t))
    print(f"解析 tee_server.log 得到 {len(server_t)} 组数据")

    process_and_output(client_t, client_o, client_g, server_t, server_g)
