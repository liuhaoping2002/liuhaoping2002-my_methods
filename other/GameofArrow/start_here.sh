#!/bin/zsh

set -e  # 一出错了就立刻停

conda init zsh
source $HOME/.zshrc
conda activate tsqp313
ALL_DIR="/home/liu/tee/liuhaoping2002-my_methods/other/"
cd ${ALL_DIR}GameofArrow/

# 脚本所在目录（保证能找到 python 文件）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/time_client_goa.py"

# 检查 Python 脚本是否存在
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "错误：找不到 Python 脚本！"
    echo "   期望路径：$PYTHON_SCRIPT"
    exit 1
fi

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
echo "   输入长度：约 $INPUT_LEN 词"
echo "=========================================="

# 真正执行
python3 -u "$PYTHON_SCRIPT" --run-times "$RUN_TIMES" --input-len "$INPUT_LEN" | tee tee_client.log
python3 result_get.py tee_client.log tee_server.log

echo "=========================================="
echo "全部运行完毕！"