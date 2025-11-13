1. 需要grpc-tools

   ```shell
   pip install grpcio-tools 
   ```

2. 生成demo_pb2*.py

   ```shell
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. demo.proto
   ```

3. 启动服务端：

   ```shell
   python3 full_layer_server.py 
   ```

4. 编译gramine配置

   ```shell
   make clean && make SGX=1
   ```

5. 运行客户端

   ```shell
   gramine-direct ./pytorch full_layer_client.py	#without sgx hardware support
   gramine-sgx ./pytorch full_layer_client.py	#with sgx hartware support
   ```

   

