# CS6111_FinalProject
Replication of SplitWise - distributed LLM inference on different worker nodes

# Generate protobufs
python -m grpc_tools.protoc -I ./protos --python_out=./src/protobufs --pyi_out=./src/protobufs --grpc_python_out=./src/protobufs ./protos/kv_transfer.proto