#!/bin/bash

# 设置日志文件名
CLIENT_LOG="time_client.log"
SERVER_LOG="time_server.log"

# 1. 检查并清空/删除日志文件
echo "处理日志文件..."
for log in "$CLIENT_LOG" "$SERVER_LOG"; do
    if [ -f "$log" ]; then
        # 方法1：清空文件（推荐，保留文件）
        echo "  - 清空 $log"
        > "$log"
        
        # 方法2：删除文件（如需删除，请取消下面这行注释）
        # echo "  - 删除 $log"
        # rm -f "$log"
    else
        echo "  - $log 不存在，跳过"
    fi
done

python3 time_client.py
# 2. 运行 time_deal.py
echo "运行 time_deal.py..."
python3 time_calc/time_deal.py time_client.log time_server.log

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "✅ 脚本执行成功"
else
    echo "❌ 脚本执行失败"
    exit 1
fi