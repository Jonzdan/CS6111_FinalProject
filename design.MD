from vllm.distributed.kv_transfer.kv_transfer_state import _KV_CONNECTOR_AGENT
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1, KVConnectorRole

# scheduler_cls - target class for new scheduler " path to local class "
from vllm.config import VllmConfig


we need to use p2p nccl connector for peer2peer similar to actual paper
- cant use nccl, so customize to local grpc & simulate network delay from infiniband

TODO: add metrics tracking (e2e latency, throughput...)
