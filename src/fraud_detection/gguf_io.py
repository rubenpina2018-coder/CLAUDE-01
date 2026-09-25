"""Minimal GGUF v3 reader/writer.

Implements the container described in ggml's ``docs/gguf.md``: little-endian
header, typed key/value metadata, tensor-info table and an aligned data
section. Only the features this project needs (scalar/array metadata, plain
and Q4_0/Q8_0 tensors) are supported. Files written here are parsed by
llama.cpp's reference ``gguf-py`` reader (see ``tests/test_gguf_io.py``).
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO, Final

import numpy as np

from .quantization import BLOCK_DTYPES, PLAIN_DTYPES, TYPE_LAYOUT, GGMLType

GGUF_MAGIC: Final = b"GGUF"
GGUF_VERSION: Final = 3
DEFAULT_ALIGNMENT: Final = 32
MAX_TENSOR_NAME_BYTES: Final = 64


class ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_SCALAR_FORMATS: Final[dict[ValueType, str]] = {
    ValueType.UINT8: "<B",
    ValueType.INT8: "<b",
    ValueType.UINT16: "<H",
    ValueType.INT16: "<h",
    ValueType.UINT32: "<I",
    ValueType.INT32: "<i",
    ValueType.FLOAT32: "<f",
    ValueType.BOOL: "<?",
    ValueType.UINT64: "<Q",
    ValueType.INT64: "<q",
    ValueType.FLOAT64: "<d",
}


class U32(int):
    """Marks an int metadata value to be stored as UINT32 (e.g. ``general.alignment``)."""


class GGUFFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Tensor:
    """A tensor in GGUF storage form.

    ``data`` is the raw storage: a plain numpy array for F*/I* types, or an array
    of block records (see ``quantization.BLOCK_DTYPES``) for quantized types.
    ``shape`` is the logical shape in numpy (row-major) order.
    """

    data: np.ndarray
    ggml_type: GGMLType
    shape: tuple[int, ...]

    @classmethod
    def plain(cls, array: np.ndarray, ggml_type: GGMLType) -> Tensor:
        array = np.ascontiguousarray(array, dtype=PLAIN_DTYPES[ggml_type])
        return cls(array, ggml_type, tuple(array.shape))

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        block_size, type_size = TYPE_LAYOUT[self.ggml_type]
        return self.n_elements // block_size * type_size


@dataclass
class GGUFFile:
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, Tensor] = field(default_factory=dict)


# --------------------------------------------------------------------------- writer


def _value_type(value: Any) -> ValueType:
    if isinstance(value, bool):  # before int: bool is an int subclass
        return ValueType.BOOL
    if isinstance(value, U32):
        return ValueType.UINT32
    if isinstance(value, int):
        return ValueType.INT64
    if isinstance(value, float):
        return ValueType.FLOAT64
    if isinstance(value, str):
        return ValueType.STRING
    if isinstance(value, list | tuple):
        return ValueType.ARRAY
    raise TypeError(f"unsupported GGUF metadata value type: {type(value).__name__}")


def _pack_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _pack_value(value: Any, vtype: ValueType) -> bytes:
    if vtype is ValueType.STRING:
        return _pack_string(value)
    if vtype is ValueType.ARRAY:
        items = list(value)
        item_type = _value_type(items[0]) if items else ValueType.STRING
        if item_type is ValueType.ARRAY:
            raise TypeError("nested metadata arrays are not supported")
        if any(_value_type(item) is not item_type for item in items):
            raise TypeError("metadata arrays must be homogeneous")
        body = b"".join(_pack_value(item, item_type) for item in items)
        return struct.pack("<IQ", item_type, len(items)) + body
    return struct.pack(_SCALAR_FORMATS[vtype], value)


def _align(offset: int, alignment: int) -> int:
    return offset + (-offset % alignment)


def write_gguf(
    path: str | Path,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Tensor],
    alignment: int = DEFAULT_ALIGNMENT,
) -> int:
    """Write a GGUF v3 file and return its size in bytes."""
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a power of two")
    metadata = dict(metadata)
    if alignment != DEFAULT_ALIGNMENT:
        metadata["general.alignment"] = U32(alignment)
    if "general.architecture" not in metadata:
        raise ValueError("'general.architecture' is required by the GGUF spec")

    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata))
    for key, value in metadata.items():
        vtype = _value_type(value)
        header += _pack_string(key) + struct.pack("<I", vtype) + _pack_value(value, vtype)

    offset = 0
    payloads: list[tuple[int, bytes]] = []
    for name, tensor in tensors.items():
        if len(name.encode()) > MAX_TENSOR_NAME_BYTES:
            raise ValueError(f"tensor name longer than {MAX_TENSOR_NAME_BYTES} bytes: {name!r}")
        block_size, _ = TYPE_LAYOUT[tensor.ggml_type]
        if tensor.shape[-1] % block_size:
            raise ValueError(f"{name}: innermost dim must be a multiple of {block_size}")
        payload = np.ascontiguousarray(tensor.data).tobytes()
        if len(payload) != tensor.nbytes:
            raise ValueError(f"{name}: expected {tensor.nbytes} bytes of data, got {len(payload)}")
        dims = tensor.shape[::-1]  # GGUF stores ne[0] (innermost) first
        header += _pack_string(name) + struct.pack(f"<I{len(dims)}Q", len(dims), *dims)
        header += struct.pack("<IQ", tensor.ggml_type, offset)
        payloads.append((offset, payload))
        offset = _align(offset + len(payload), alignment)

    data_start = _align(len(header), alignment)
    buffer = bytearray(data_start + offset)
    buffer[: len(header)] = header
    for tensor_offset, payload in payloads:
        start = data_start + tensor_offset
        buffer[start : start + len(payload)] = payload
    Path(path).write_bytes(buffer)
    return len(buffer)


# --------------------------------------------------------------------------- reader


class _Cursor:
    def __init__(self, buffer: bytes) -> None:
        self.buffer = buffer
        self.pos = 0

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        try:
            values = struct.unpack_from(fmt, self.buffer, self.pos)
        except struct.error as exc:
            raise GGUFFormatError(f"truncated file at byte {self.pos}") from exc
        self.pos += struct.calcsize(fmt)
        return values

    def string(self) -> str:
        (length,) = self.unpack("<Q")
        raw = self.buffer[self.pos : self.pos + length]
        if len(raw) != length:
            raise GGUFFormatError("truncated string")
        self.pos += length
        return raw.decode("utf-8")

    def value(self, vtype: ValueType) -> Any:
        if vtype is ValueType.STRING:
            return self.string()
        if vtype is ValueType.ARRAY:
            item_type, count = self.unpack("<IQ")
            return [self.value(ValueType(item_type)) for _ in range(count)]
        return self.unpack(_SCALAR_FORMATS[vtype])[0]


def read_gguf(source: str | Path | bytes | BinaryIO) -> GGUFFile:
    """Parse a GGUF v3 file. Tensor arrays are read-only views over the file bytes."""
    if isinstance(source, bytes):
        buffer = source
    elif isinstance(source, str | Path):
        buffer = Path(source).read_bytes()
    else:
        buffer = source.read()
    cursor = _Cursor(buffer)
    if cursor.buffer[:4] != GGUF_MAGIC:
        raise GGUFFormatError("not a GGUF file (bad magic)")
    cursor.pos = 4
    version, n_tensors, n_kv = cursor.unpack("<IQQ")
    if version != GGUF_VERSION:
        raise GGUFFormatError(f"unsupported GGUF version {version}")

    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        key = cursor.string()
        (vtype,) = cursor.unpack("<I")
        if key in metadata:
            raise GGUFFormatError(f"duplicate metadata key {key!r}")
        metadata[key] = cursor.value(ValueType(vtype))
    alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))

    infos: list[tuple[str, tuple[int, ...], GGMLType, int]] = []
    for _ in range(n_tensors):
        name = cursor.string()
        (n_dims,) = cursor.unpack("<I")
        dims = cursor.unpack(f"<{n_dims}Q")
        ggml_type, offset = cursor.unpack("<IQ")
        if offset % alignment:
            raise GGUFFormatError(f"tensor {name!r} is not aligned")
        infos.append((name, tuple(dims[::-1]), GGMLType(ggml_type), offset))

    data_start = _align(cursor.pos, alignment)
    tensors: dict[str, Tensor] = {}
    for name, shape, ggml_type, offset in infos:
        if name in tensors:
            raise GGUFFormatError(f"duplicate tensor name {name!r}")
        n_elements = int(np.prod(shape, dtype=np.int64))
        if ggml_type in BLOCK_DTYPES:
            block_size, _ = TYPE_LAYOUT[ggml_type]
            dtype, count = BLOCK_DTYPES[ggml_type], n_elements // block_size
            array_shape = (*shape[:-1], shape[-1] // block_size)
        else:
            dtype, count, array_shape = PLAIN_DTYPES[ggml_type], n_elements, shape
        start = data_start + offset
        if start + count * dtype.itemsize > len(buffer):
            raise GGUFFormatError(f"tensor {name!r} overruns the file")
        array = np.frombuffer(buffer, dtype=dtype, count=count, offset=start).reshape(array_shape)
        tensors[name] = Tensor(array, ggml_type, shape)
    return GGUFFile(metadata=metadata, tensors=tensors)
