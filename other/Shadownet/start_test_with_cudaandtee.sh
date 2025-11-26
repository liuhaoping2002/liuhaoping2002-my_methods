#!/bin/zsh
#python3 -u time_server.py | tee tee_server.log
set -e  # 一出错了就立刻停

conda init zsh > /dev/null
source $HOME/.zshrc > /dev/null
conda activate tsqp313 > /dev/null
#ALL_DIR="/home/l/test/my_methods/other/"
#cd ${ALL_DIR}Soter/

# 脚本所在目录（保证能找到 python 文件）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/time_client_shadownet.py"

# 检查 Python 脚本是否存在
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "错误：找不到 Python 脚本！"
    echo "   期望路径：$PYTHON_SCRIPT"
    exit 1
fi

PORT=50051

# 查找占用端口的进程，并获取 PID
PIDS=$(lsof -t -i :$PORT) || true

echo $PIDS
if [ -n "$PIDS" ]; then
    # 如果 PIDS 非空，表示端口被占用，逐个终止占用该端口的进程
    echo "端口 $PORT 已被进程 $PIDS 占用，正在终止该进程..."

    # 使用 while 循环逐个终止进程
    echo "$PIDS" | while read PID; do
        echo "终止进程 $PID ..."
        kill -9 $PID
        # 检查 kill 命令是否成功执行
        if [ $? -eq 0 ]; then
            echo "进程 $PID 已被终止。"
        else
            echo "无法终止进程 $PID，可能需要手动处理。"
        fi
    done

    # 等待 5 秒钟，确保进程完全被终止
    echo "等待 5 秒钟，确保进程完全终止..."
    sleep 5

    # 再次检查是否还有进程占用该端口
    #echo "sleep 5 秒后，重新检查端口 $PORT 的占用情况..."
    PIDS=$(lsof -t -i :$PORT) || true
    #echo "再次检测"
    if [ -n "$PIDS" ]; then
        echo "无法终止进程，端口 $PORT 仍被占用，请检查是否有其他进程正在使用该端口。"
        exit 1
    else
        echo "端口 $PORT 已被成功释放。"
    fi
else
    echo "端口 $PORT 没有被占用。"
fi

echo "启动 gRPC 服务端..."
python3 -u time_server_shadownet.py --device cuda | tee tee_server.log &
SERVER_PID=$!

echo "等待服务端启动并监听端口 50051..."
until nc -z localhost 50051; do
  sleep 1
done

# 参数处理
RUN_TIMES=${1:-5}      # 第一个参数，没填默认 5
INPUT_LEN=${2:-10}     # 第二个参数，没填默认 10

if ! [[ "$RUN_TIMES" =~ ^[0-9]+$ ]] || ! [[ "$INPUT_LEN" =~ ^[0-9]+$ ]]; then
    echo "错误：参数必须是正整数！"
    echo "你输入的是：次数=$RUN_TIMES  长度=$INPUT_LEN"
    exit 1
fi

echo "开始运行 Transformer gRPC 客户端"
echo "   运行次数：$RUN_TIMES"
echo "   输入长度：$INPUT_LEN 词"
echo "=========================================="

# 真正执行
make clean
make SGX=1
gramine-sgx ./pytorch "$PYTHON_SCRIPT" --run-times "$RUN_TIMES" --input-len "$INPUT_LEN" | tee tee_client.log
#python3 -u "$PYTHON_SCRIPT" --run-times "$RUN_TIMES" --input-len "$INPUT_LEN" | tee tee_client.log
python3 result_get.py tee_client.log tee_server.log

echo "=========================================="
echo "全部运行完毕！"

kill $SERVER_PID
wait $SERVER_PID
echo "服务端已结束，客户端任务完成。"
