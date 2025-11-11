1. 如果没有demo_pb2.py  demo_pb2_grpc.py，可以由demo.proto生成：

   ```shell
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. demo.proto
   ```

2. 需要安装包括：

   ```
   pip install scipy transformers numpy grpcio tqdm protobuf torch
   ```

3. 直接分别运行server和client即可。

