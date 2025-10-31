# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import msgpack
import torch
import grpc

from vllm.config.kv_transfer import KVTransferConfig
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import current_stream
from src.protobufs.kv_transfer_pb2_grpc import KVTransferServicer, KVTransferStub, add_KVTransferServicer_to_server
from src.protobufs.kv_transfer_pb2 import KVBatch, SendQueueItem

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32
import zmq
import io
from concurrent.futures import Future, ThreadPoolExecutor
from src.inference.shared.servicer import KVTransferService

class P2pEngine:
    def __init__(
        self,
        local_rank: int,
        config: KVTransferConfig,
        hostname: str = "localhost",
        port_offset: int = 0,
        library_path: str | None = None,
    ) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank

        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        self.grpc_address = f"{self._hostname}:{self._port}"
        self.server = grpc.server(thread_pool=ThreadPoolExecutor(max_workers=8))
        add_KVTransferServicer_to_server(KVTransferService(self), self.server)

        self.server.add_insecure_port(self.grpc_address)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()

        self.send_stream = None
        self.recv_stream = None

        self.pool = {}

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC.
        self.send_type = self.config.get_from_extra_config("send_type", "PUT_ASYNC")
        if self.send_type == "GET":
            # tensor_id: torch.Tensor
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            # PUT or PUT_ASYNC
            # tensor_id: torch.Tensor
            self.send_queue: deque[SendQueueItem] = deque()
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(
                    target=self.send_async, daemon=True
                )
                self._send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)

        logger.info(
            "💯P2pEngine init, rank:%d, local_rank:%d, "
            "grpc_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f",
            self.rank,
            self.local_rank,
            self.grpc_address,
            self.send_type,
            self.buffer_size_threshold,
        )
        self.server.start()

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str | None = None,
    ) -> bool:
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True

        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        item = SendQueueItem(
            tensor_id=tensor_id, remote_address=remote_address, tensor=buffer.getvalue()
        )

        if self.send_type == "PUT":
            with grpc.insecure_channel(remote_address) as channel:
                stub = KVTransferStub(channel=channel)

                request_batch = KVBatch([item])
                ack = stub.PutBatch(request_batch)

                return ack.success

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                self.send_queue_cv.notify()
            return True

        # GET
        with self.send_store_cv:
            tensor_size = tensor.element_size() * tensor.numel()
            if tensor_size > self.buffer_size_threshold:
                logger.warning(
                    "❗[GET]tensor_id:%s, tensor_size:%d, is greater than"
                    "buffer size threshold :%d, skip send to %s, rank:%d",
                    tensor_id,
                    tensor_size,
                    self.buffer_size_threshold,
                    remote_address,
                    self.rank,
                )
                return False
            while self.buffer_size + tensor_size > self.buffer_size_threshold:
                assert len(self.send_store) > 0
                oldest_tensor_id = next(iter(self.send_store))
                oldest_tensor = self.send_store.pop(oldest_tensor_id)
                oldest_tensor_size = (
                    oldest_tensor.element_size() * oldest_tensor.numel()
                )
                self.buffer_size -= oldest_tensor_size
                logger.debug(
                    "⛔[GET]Send to %s, tensor_id:%s, tensor_size:%d,"
                    " buffer_size:%d, oldest_tensor_size:%d, rank:%d",
                    remote_address,
                    tensor_id,
                    tensor_size,
                    self.buffer_size,
                    oldest_tensor_size,
                    self.rank,
                )

            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
            logger.debug(
                "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                "shape:%s, rank:%d, buffer_size:%d(%.2f%%)",
                remote_address,
                tensor_id,
                tensor_size,
                tensor.shape,
                self.rank,
                self.buffer_size,
                self.buffer_size / self.buffer_size_threshold * 100,
            )
        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: str | None = None,
    ) -> torch.Tensor:
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    self.recv_store_cv.wait()
                tensor = self.recv_store.pop(tensor_id)
            return tensor

        if remote_address is None:
            raise ValueError("remote_address must be provided for GET mode")
        # GET
        with grpc.insecure_channel(remote_address) as channel:
            stub = KVTransferStub(channel)
            request = KVBatch(items=[SendQueueItem(tensor_id=tensor_id)])
            response = stub.GetBatch(request)

            if len(response.items) == 0:
                # Tensor not found
                return None

            # Only one tensor expected per request
            tensor_bytes = response.items[0].tensor
            tensor = torch.load(io.BytesIO(tensor_bytes), map_location="cpu")
        return tensor

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.send_queue_cv:
                while self.send_queue:
                    self.send_queue_cv.wait()
            duration = time.time() - start_time
            logger.debug(
                "🚧[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d",
                duration * 1000,
                self.rank,
            )

    def get_finished(
        self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        # Clear the buffer upon request completion.
        for request_id in finished_req_ids:
            for layer_name in no_compile_layers:
                tensor_id = request_id + "#" + layer_name
                if tensor_id in self.recv_store:
                    with self.recv_store_cv:
                        tensor = self.recv_store.pop(tensor_id, None)

        # TODO:Retrieve requests that have already sent the KV cache.
        finished_sending: set[str] = set()

        # TODO:Retrieve requests that have already received the KV cache.
        finished_recving: set[str] = set()

        return finished_sending or None, finished_recving or None

    def close(self) -> None:
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()