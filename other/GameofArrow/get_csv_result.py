#!/usr/bin/env python3
# coding: utf-8
"""
merge_client_server_csv_table.py

将终端输出表格（含均值行）保存为 CSV，且占比列以纯数值形式保存（不带 '%'）。

用法:
    python merge_client_server_csv_table.py client.log server.log
选项:
    --no-csv           不保存 CSV（仍会在终端打印表格）
    --csv <filename>   指定 CSV 文件名（默认 merged_results.csv）
"""
import sys
import re
import csv
import argparse
from statistics import mean

NUM_RE = r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'

def clean_number_token(tok: str) -> str:
    tok = tok.strip()
    tok = tok.replace(',', '')  # 去掉千位分隔符
    return tok

def extract_grand_totals_from_lines(lines, need_count_numbers=3):
    results = []
    header_pat = re.compile(r'^\s*GRAND TOTAL\b', re.IGNORECASE)
    num_pat = re.compile(NUM_RE)
    for idx, line in enumerate(lines):
        if header_pat.search(line):
            raw_nums = num_pat.findall(line)
            cleaned = [clean_number_token(x) for x in raw_nums]
            if len(cleaned) >= need_count_numbers:
                try:
                    vals = tuple(float(x) for x in cleaned[:need_count_numbers])
                    results.append(vals)
                    continue
                except Exception:
                    pass
            # 合并后续最多两行再试
            combined = line
            for j in range(1, 3):
                if idx + j < len(lines):
                    combined += ' ' + lines[idx + j]
            raw_nums2 = num_pat.findall(combined)
            cleaned2 = [clean_number_token(x) for x in raw_nums2]
            if len(cleaned2) >= need_count_numbers:
                try:
                    vals = tuple(float(x) for x in cleaned2[:need_count_numbers])
                    results.append(vals)
                except Exception:
                    continue
            else:
                continue
    return results

def read_file_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_csv(out_path, header, rows):
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

def main():
    parser = argparse.ArgumentParser(description="合并 client/server 的 GRAND TOTAL，并将终端表格保存为 CSV（占比为纯数值）。")
    parser.add_argument('client_log', help='client 日志文件路径')
    parser.add_argument('server_log', help='server 日志文件路径')
    parser.add_argument('--no-csv', action='store_true', help='不保存 CSV 文件')
    parser.add_argument('--csv', type=str, default='merged_results.csv', help='CSV 输出文件名（默认 merged_results.csv）')
    args = parser.parse_args()

    # 读取文件
    try:
        client_lines = read_file_lines(args.client_log)
    except Exception as e:
        print(f"无法读取 client 文件 {args.client_log}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        server_lines = read_file_lines(args.server_log)
    except Exception as e:
        print(f"无法读取 server 文件 {args.server_log}: {e}", file=sys.stderr)
        sys.exit(2)

    # 提取 client (3 列)
    client_matches = extract_grand_totals_from_lines(client_lines, need_count_numbers=3)
    if not client_matches:
        print("错误：在 client_log 中未找到任何 'GRAND TOTAL' 行或每行数字不足 3 个。", file=sys.stderr)
        sys.exit(1)
    client_back_first = list(reversed(client_matches))
    n = len(client_back_first)
    print(f"在 client_log 中提取到 {n} 组 GRAND TOTAL（三列）。")

    # 提取 server (2 列)
    server_matches = extract_grand_totals_from_lines(server_lines, need_count_numbers=2)
    if len(server_matches) < n:
        print(f"错误：在 server_log 中仅找到 {len(server_matches)} 组 'GRAND TOTAL'，少于 client 中的 {n} 组。", file=sys.stderr)
        sys.exit(1)
    server_tail = server_matches[-n:]
    server_back_first = list(reversed(server_tail))

    # 合并并计算
    merged = []
    for i in range(n):
        c_pure, c_scaling, c_rpc = client_back_first[i]
        s_pure, s_total_call = server_back_first[i]
        communication = c_rpc - s_total_call
        total = c_pure + c_scaling + s_pure + communication
        if total == 0:
            pct_c_pure = pct_c_scaling = pct_s_pure = pct_comm = 0.0
        else:
            pct_c_pure = (c_pure / total) * 100.0
            pct_c_scaling = (c_scaling / total) * 100.0
            pct_s_pure = (s_pure / total) * 100.0
            pct_comm = (communication / total) * 100.0
        merged.append({
            'client_PureCompute': c_pure,
            'client_Scaling': c_scaling,
            'client_RPC_Total': c_rpc,
            'server_PureCompute': s_pure,
            'server_TotalServerCall': s_total_call,
            'communication': communication,
            'Total': total,
            'pct_client_PureCompute': pct_c_pure,
            'pct_client_Scaling': pct_c_scaling,
            'pct_server_PureCompute': pct_s_pure,
            'pct_communication': pct_comm
        })

    # 准备打印表格的列顺序
    headers = [
        "client_PureCompute",
        "client_Scaling",
        "client_RPC_Total",
        "server_PureCompute",
        "server_TotalServerCall",
        "communication",
        "Total",
        "pct_client_PureCompute",
        "pct_client_Scaling",
        "pct_server_PureCompute",
        "pct_communication"
    ]

    # 终端打印（文件最后的组先）
    print("\n提取并计算后的每组数据（文件中最后的组先列出），最后一行为均值：")
    # 打印表头
    print("组 | " + " | ".join(f"{h:>20}" for h in headers))
    for idx, row in enumerate(merged, start=1):
        numeric_vals = [row[h] for h in headers[:7]]
        pct_vals = [row[h] for h in headers[7:]]
        print(f"{idx:2d} | " + " | ".join(f"{v:20.6f}" for v in numeric_vals) + " | " + " | ".join(f"{p:17.2f}%" for p in pct_vals))

    # 均值计算（数值列与占比列分别平均）
    # 把每列提取为 list
    col_lists = {h: [row[h] for row in merged] for h in headers}
    averages = {h: (mean(col_lists[h]) if col_lists[h] else 0.0) for h in headers}

    # 打印均值行（终端显示占比带 %）
    numeric_avg_vals = [averages[h] for h in headers[:7]]
    pct_avg_vals = [averages[h] for h in headers[7:]]
    print("\n均值（最后一行）:")
    print("AVG| " + " | ".join(f"{v:20.6f}" for v in numeric_avg_vals) + " | " + " | ".join(f"{p:17.2f}%" for p in pct_avg_vals))

    print(f"\n共提取并配对到 {n} 组数据。")

    # 准备 CSV（与表格一致），占比列以纯数值保存（不带 %）
    csv_header = ['group_index'] + headers
    csv_rows = []
    for idx, row in enumerate(merged, start=1):
        csv_rows.append([idx] + [row[h] for h in headers])
    # 最后一行均值，group_index 用 'AVERAGE'
    csv_rows.append(['AVERAGE'] + [averages[h] for h in headers])

    if not args.no_csv:
        try:
            write_csv(args.csv, csv_header, csv_rows)
            print(f"\n已将合并结果（含均值行）保存为 CSV: {args.csv}")
        except Exception as e:
            print(f"保存 CSV 失败: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()
