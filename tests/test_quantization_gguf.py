"""Q8_0/Q4_0 kernels and the GGUF container, checked against llama.cpp's gguf-py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fraud_detection.gguf_io import U32, GGUFFormatError, Tensor, read_gguf, write_gguf
from fraud_detection.quantization import (
    QK,
    GGMLType,
    QuantizedVector,
    dequantize_q4_0,
    dequantize_q8_0,
    quantize_q4_0,
    quantize_q8_0,
    storage_nbytes,
)

gguf = pytest.importorskip("gguf", reason="reference implementation (dev dependency)")
from gguf import quants  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402

RNG = np.random.default_rng(1234)
DISTRIBUTIONS = {
    "normal": RNG.normal(0, 0.3, 2048),
    "heavy_tail": RNG.standard_t(2, 2048) * 0.05,
    "sparse": np.where(RNG.random(2048) < 0.6, 0.0, RNG.normal(0, 1, 2048)),
    "rounding_ties": np.round(RNG.normal(0, 3, 2048)) / 2,
    "all_zero_blocks": np.concatenate([np.zeros(64), RNG.normal(0, 1, 64)]),
}
KERNELS = {
    GGMLType.Q8_0: (quantize_q8_0, dequantize_q8_0, GGMLQuantizationType.Q8_0),
    GGMLType.Q4_0: (quantize_q4_0, dequantize_q4_0, GGMLQuantizationType.Q4_0),
}


@pytest.mark.parametrize("qtype", list(KERNELS))
@pytest.mark.parametrize("name", list(DISTRIBUTIONS))
def test_kernels_are_bit_exact_with_ggml(qtype: GGMLType, name: str) -> None:
    quantize, dequantize, reference_type = KERNELS[qtype]
    x = DISTRIBUTIONS[name].astype(np.float32)
    ours = quantize(x)
    reference = quants.quantize(x.reshape(1, -1), reference_type).reshape(-1)
    np.testing.assert_array_equal(ours.view(np.uint8).reshape(-1), reference)
    np.testing.assert_array_equal(
        dequantize(ours), quants.dequantize(reference.reshape(1, -1), reference_type).reshape(-1)
    )


@pytest.mark.parametrize("qtype", list(KERNELS))
def test_reconstruction_error_is_bounded_by_the_block_scale(qtype: GGMLType) -> None:
    quantize, dequantize, _ = KERNELS[qtype]
    x = DISTRIBUTIONS["normal"].astype(np.float32)
    blocks = quantize(x)
    scale = np.abs(blocks["d"].astype(np.float32)).repeat(QK)
    error = np.abs(dequantize(blocks) - x)
    # Rounding error: half a step (Q8_0), or up to one step for Q4_0 values clipped at
    # code 15. The scale itself is stored as fp16 (relative error <= 2**-11), which adds
    # up to |max code| * 2**-11 steps.
    steps, max_code = (0.5, 127) if qtype is GGMLType.Q8_0 else (1.0, 8)
    bound = (steps + max_code * 2.0**-11) * scale * (1 + 2.0**-10)
    assert np.all(error <= bound + 1e-7)


@pytest.mark.parametrize("qtype", [GGMLType.F32, GGMLType.F16, GGMLType.Q8_0, GGMLType.Q4_0])
def test_random_access_gather_matches_full_decode(qtype: GGMLType) -> None:
    vector = QuantizedVector.encode(DISTRIBUTIONS["heavy_tail"], qtype)
    index = RNG.integers(0, vector.size, size=(64, 8))
    np.testing.assert_array_equal(vector.gather(index), vector.decode()[index])
    assert vector.nbytes == storage_nbytes(vector.size, qtype)


def test_storage_sizes() -> None:
    assert storage_nbytes(64, GGMLType.Q4_0) == 2 * 18
    assert storage_nbytes(64, GGMLType.Q8_0) == 2 * 34
    assert storage_nbytes(64, GGMLType.F16) == 128
    with pytest.raises(ValueError):
        storage_nbytes(33, GGMLType.Q4_0)


def _write_sample(path: Path) -> dict[str, Tensor]:
    leaves = QuantizedVector.encode(RNG.normal(0, 0.1, 3 * 64), GGMLType.Q4_0)
    tensors = {
        "blk.q4": Tensor(leaves.storage().reshape(3, -1), GGMLType.Q4_0, (3, 64)),
        "blk.q8": Tensor(quantize_q8_0(RNG.normal(size=32)), GGMLType.Q8_0, (32,)),
        "edges": Tensor.plain(np.array([0.5, 1.5, 2.25]), GGMLType.F64),
        "weights": Tensor.plain(RNG.normal(size=(4, 5)), GGMLType.F32),
        "codes": Tensor.plain(np.arange(-3, 4), GGMLType.I8),
        "offsets": Tensor.plain(np.array([0, 3, 7]), GGMLType.I32),
    }
    metadata = {
        "general.architecture": "gbdt",
        "general.file_type": U32(2),
        "test.name": "fraud",
        "test.count": -7,
        "test.ratio": 0.125,
        "test.flag": True,
        "test.features": ["amount", "hour_of_day"],
    }
    write_gguf(path, metadata, tensors)
    return tensors


def test_round_trip_preserves_metadata_and_tensors(tmp_path: Path) -> None:
    tensors = _write_sample(tmp_path / "m.gguf")
    parsed = read_gguf(tmp_path / "m.gguf")
    assert parsed.metadata == {
        "general.architecture": "gbdt",
        "general.file_type": 2,
        "test.name": "fraud",
        "test.count": -7,
        "test.ratio": 0.125,
        "test.flag": True,
        "test.features": ["amount", "hour_of_day"],
    }
    for name, tensor in tensors.items():
        assert parsed.tensors[name].ggml_type == tensor.ggml_type
        assert parsed.tensors[name].shape == tensor.shape
        assert parsed.tensors[name].data.tobytes() == np.ascontiguousarray(tensor.data).tobytes()


def test_reference_reader_parses_our_files(tmp_path: Path) -> None:
    tensors = _write_sample(tmp_path / "m.gguf")
    reader = gguf.GGUFReader(tmp_path / "m.gguf")
    parsed = {t.name: t for t in reader.tensors}
    assert set(parsed) == set(tensors)
    for name, tensor in tensors.items():
        assert parsed[name].tensor_type.name == tensor.ggml_type.name
        assert [int(d) for d in parsed[name].shape] == list(tensor.shape[::-1])
        assert int(parsed[name].n_bytes) == tensor.nbytes
    q4 = parsed["blk.q4"]
    np.testing.assert_array_equal(
        quants.dequantize(q4.data, GGMLQuantizationType.Q4_0).reshape(-1),
        dequantize_q4_0(tensors["blk.q4"].data.reshape(-1)),
    )
    architecture = reader.fields["general.architecture"]
    assert bytes(architecture.parts[architecture.data[0]]).decode() == "gbdt"


def test_corrupted_files_are_rejected(tmp_path: Path) -> None:
    _write_sample(tmp_path / "m.gguf")
    raw = (tmp_path / "m.gguf").read_bytes()
    with pytest.raises(GGUFFormatError, match="magic"):
        read_gguf(b"GGML" + raw[4:])
    with pytest.raises(GGUFFormatError):
        read_gguf(raw[:200])
    with pytest.raises(ValueError, match="architecture"):
        write_gguf(tmp_path / "x.gguf", {"general.name": "x"}, {})
    with pytest.raises(ValueError, match="longer than"):
        write_gguf(
            tmp_path / "x.gguf",
            {"general.architecture": "gbdt"},
            {"n" * 65: Tensor.plain(np.zeros(1), GGMLType.F32)},
        )
