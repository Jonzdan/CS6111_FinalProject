import grpc
from src.protobufs.kv_transfer_pb2 import SendQueueItem, KVBatch, Ack
from src.protobufs.kv_transfer_pb2_grpc import KVTransferServicer, KVTransferStub
from src.inference.shared.engine import P2pEngine

import torch
import io

class KVTransferService(KVTransferServicer):
    def __init__(self, engine: P2pEngine):
        super().__init__()
        self.engine = engine

    def PutBatch(self, request: KVBatch, context) -> Ack:
        for item in request.items:
            tensor = torch.load(io.BytesIO(item.tensor), map_location="cpu")
            self.engine.recv_store[item.tensor_id] = tensor
        
        return Ack(success=True)
    
    def GetBatch(self, request: KVBatch, context) -> KVBatch:
        response = KVBatch()
        for req in request.items:
            tensor_id = req.tensor_id
            tensor = self.engine.send_store.get(tensor_id)

            if tensor is None:
                continue

            buffer = io.BytesIO()
            torch.save(tensor.cpu(), buf)

            response.items.append(
                SendQueueItem(
                    tensor_id=tensor_id,
                    remote_address=self.engine.grpc_address,
                    tensor=buffer.getvalue()
                )
            )

        return response
