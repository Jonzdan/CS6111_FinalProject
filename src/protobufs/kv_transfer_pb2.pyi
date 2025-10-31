from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SendQueueItem(_message.Message):
    __slots__ = ("tensor_id", "remote_address", "tensor")
    TENSOR_ID_FIELD_NUMBER: _ClassVar[int]
    REMOTE_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    tensor_id: str
    remote_address: str
    tensor: bytes
    def __init__(self, tensor_id: _Optional[str] = ..., remote_address: _Optional[str] = ..., tensor: _Optional[bytes] = ...) -> None: ...

class KVBatch(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[SendQueueItem]
    def __init__(self, items: _Optional[_Iterable[_Union[SendQueueItem, _Mapping]]] = ...) -> None: ...

class Ack(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
