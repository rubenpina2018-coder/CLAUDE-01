"""Block quantization kernels with ggml semantics.

Q8_0 and Q4_0 follow ggml's reference implementation (``ggml-quants.c``): blocks
of 32 values sharing one float16 scale ``d``.

* Q8_0: ``d = max|x| / 127``, ``q = roundf(x / d)`` as int8      -> 34 bytes / 32 values
* Q4_0: ``d = x[argmax|x|] / -8``, ``q = min(15, trunc(x / d + 8.5))`` packed two
  per byte (first half of the block in the low nibbles)      -> 18 bytes / 32 values

Tests check bit-exactness against llama.cpp's ``gguf-py`` reference.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np


class GGMLType(IntEnum):
    """Tensor types (subset of ``enum ggml_type``) used by this project."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q8_0 = 8
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28


QK: Final = 32  # values per quantization block (QK4_0 == QK8_0 == 32)
BLOCK_Q8_0: Final = np.dtype([("d", "<f2"), ("qs", "i1", (QK,))])
BLOCK_Q4_0: Final = np.dtype([("d", "<f2"), ("qs", "u1", (QK // 2,))])

# (values per block, bytes per block) -- mirrors GGML_QUANT_SIZES.
TYPE_LAYOUT: Final[dict[GGMLType, tuple[int, int]]] = {
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.Q4_0: (QK, BLOCK_Q4_0.itemsize),
    GGMLType.Q8_0: (QK, BLOCK_Q8_0.itemsize),
    GGMLType.I8: (1, 1),
    GGMLType.I16: (1, 2),
    GGMLType.I32: (1, 4),
    GGMLType.I64: (1, 8),
    GGMLType.F64: (1, 8),
}

PLAIN_DTYPES: Final[dict[GGMLType, np.dtype]] = {
    GGMLType.F32: np.dtype("<f4"),
    GGMLType.F16: np.dtype("<f2"),
    GGMLType.I8: np.dtype("i1"),
    GGMLType.I16: np.dtype("<i2"),
    GGMLType.I32: np.dtype("<i4"),
    GGMLType.I64: np.dtype("<i8"),
    GGMLType.F64: np.dtype("<f8"),
}
BLOCK_DTYPES: Final[dict[GGMLType, np.dtype]] = {
    GGMLType.Q4_0: BLOCK_Q4_0,
    GGMLType.Q8_0: BLOCK_Q8_0,
}


def storage_nbytes(n_values: int, qtype: GGMLType) -> int:
    block_size, type_size = TYPE_LAYOUT[qtype]
    if n_values % block_size:
        raise ValueError(f"{qtype.name} needs a multiple of {block_size} values, got {n_values}")
    return n_values // block_size * type_size


def _blocks(x: np.ndarray) -> np.ndarray:
    flat = np.asarray(x, dtype=np.float32).reshape(-1)
    if flat.size % QK:
        raise ValueError(f"block quantization needs a multiple of {QK} values, got {flat.size}")
    return flat.reshape(-1, QK)


def _roundf(x: np.ndarray) -> np.ndarray:
    """C ``roundf``: round half away from zero (``np.round`` rounds half to even)."""
    a = np.abs(x)
    floored = np.floor(a)
    return np.sign(x) * (floored + np.floor(2 * (a - floored)))


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    blocks = _blocks(x)
    d = np.abs(blocks).max(axis=1, keepdims=True) / np.float32(127)
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, np.float32(0), np.float32(1) / d)
    out = np.empty(blocks.shape[0], dtype=BLOCK_Q8_0)
    out["d"] = d[:, 0].astype(np.float16)
    out["qs"] = _roundf(blocks * inv_d).astype(np.int8)
    return out


def dequantize_q8_0(blocks: np.ndarray) -> np.ndarray:
    d = blocks["d"].astype(np.float32)[:, None]
    return (blocks["qs"].astype(np.float32) * d).reshape(-1)


def quantize_q4_0(x: np.ndarray) -> np.ndarray:
    blocks = _blocks(x)
    imax = np.abs(blocks).argmax(axis=1)[:, None]
    signed_max = np.take_along_axis(blocks, imax, axis=1)
    d = signed_max / np.float32(-8)
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, np.float32(0), np.float32(1) / d)
    q = np.trunc(blocks * inv_d + np.float32(8.5)).astype(np.uint8).clip(0, 15)
    out = np.empty(blocks.shape[0], dtype=BLOCK_Q4_0)
    out["d"] = d[:, 0].astype(np.float16)
    out["qs"] = q[:, : QK // 2] | (q[:, QK // 2 :] << np.uint8(4))
    return out


def dequantize_q4_0(blocks: np.ndarray) -> np.ndarray:
    d = blocks["d"].astype(np.float32)[:, None]
    qs = blocks["qs"]
    q = np.concatenate([qs & np.uint8(0x0F), qs >> np.uint8(4)], axis=1).astype(np.int8)
    return ((q - np.int8(8)).astype(np.float32) * d).reshape(-1)


def quantize(x: np.ndarray, qtype: GGMLType) -> np.ndarray:
    """Encode a float vector into the storage representation of ``qtype``."""
    if qtype is GGMLType.Q8_0:
        return quantize_q8_0(x)
    if qtype is GGMLType.Q4_0:
        return quantize_q4_0(x)
    if qtype in (GGMLType.F64, GGMLType.F32, GGMLType.F16):
        return np.asarray(x, dtype=np.float64).reshape(-1).astype(PLAIN_DTYPES[qtype])
    raise ValueError(f"unsupported value quantization type: {qtype!r}")


def dequantize(storage: np.ndarray, qtype: GGMLType) -> np.ndarray:
    if qtype is GGMLType.Q8_0:
        return dequantize_q8_0(storage)
    if qtype is GGMLType.Q4_0:
        return dequantize_q4_0(storage)
    return (
        np.asarray(storage).reshape(-1).astype(np.float64 if qtype is GGMLType.F64 else np.float32)
    )


class QuantizedVector:
    """Read-only vector kept compressed in memory, with random-access decoding.

    ``gather`` dequantizes only the requested positions (like ggml kernels do
    on the fly), so the resident footprint is the quantized one.
    """

    def __init__(self, storage: np.ndarray, qtype: GGMLType, size: int) -> None:
        self.qtype = GGMLType(qtype)
        self.size = size
        if self.qtype in BLOCK_DTYPES:
            self._qs = np.ascontiguousarray(storage["qs"])
            self._d = storage["d"].astype(np.float32)  # 1/32 of the payload, pre-widened
            self.nbytes = self._qs.nbytes + storage["d"].nbytes
        else:
            self._values = np.ascontiguousarray(storage)
            self.nbytes = self._values.nbytes

    @classmethod
    def encode(cls, values: np.ndarray, qtype: GGMLType) -> QuantizedVector:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        return cls(quantize(values, qtype), qtype, values.size)

    def gather(self, index: np.ndarray) -> np.ndarray:
        if self.qtype is GGMLType.Q4_0:
            block, pos = index >> 5, index & (QK - 1)
            byte = self._qs[block, pos & (QK // 2 - 1)]
            nibble = np.where(pos < QK // 2, byte & np.uint8(0x0F), byte >> np.uint8(4))
            return (nibble.astype(np.float32) - np.float32(8)) * self._d[block]
        if self.qtype is GGMLType.Q8_0:
            return self._qs[index >> 5, index & (QK - 1)].astype(np.float32) * self._d[index >> 5]
        return self._values[index]

    def storage(self) -> np.ndarray:
        """Storage array in GGUF layout (structured blocks or plain floats)."""
        if self.qtype in BLOCK_DTYPES:
            out = np.empty(self._qs.shape[0], dtype=BLOCK_DTYPES[self.qtype])
            out["d"] = self._d.astype(np.float16)
            out["qs"] = self._qs
            return out
        return self._values

    def decode(self) -> np.ndarray:
        return dequantize(self.storage(), self.qtype)
