"""Numpy-only inference engine for compiled, quantized gradient-boosted trees.

Model layout (built by ``train_and_optimize.compile_hgb``):

* Every tree is stored as a *perfect* binary tree of depth ``D`` in breadth-first
  order (children of node ``i`` are ``2i+1`` / ``2i+2``), so traversal needs no
  child pointers. Shallower branches are padded with "always go left" nodes.
* Inputs are quantized to ``uint8`` bins using the split points the trees
  actually use. Since ``x <= edge[b]  <=>  bin(x) <= b``, this is lossless with
  respect to the trained model; bin 255 encodes a missing value.
* Leaf values are the only lossy part: F32, F16, Q8_0 or Q4_0 (ggml blocks),
  kept compressed in memory and dequantized on the fly for the visited leaves.

All trees are evaluated at once with ``D`` vectorized steps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from .gguf_io import U32, Tensor, read_gguf, write_gguf
from .quantization import GGMLType, QuantizedVector

ARCHITECTURE: Final = "gbdt"
MISSING_BIN: Final = 255
# llama_ftype of the dominant (leaf) tensor type, for the GGUF `general.file_type` key.
_FILE_TYPE: Final[dict[GGMLType, int]] = {
    GGMLType.F32: 0,
    GGMLType.F16: 1,
    GGMLType.Q4_0: 2,
    GGMLType.Q8_0: 7,
}
_QUANTIZATION_VERSION: Final = 2  # GGML_QNT_VERSION
# GGUF has no unsigned 8-bit tensor type: these tensors hold uint8 payloads in I8 storage.
_UINT8_TENSORS: Final = ("gbdt.split_feature", "gbdt.split_bin", "gbdt.missing_left_bits")


class ModelFormatError(ValueError):
    pass


class QuantizedGBDT:
    """Binary classifier: ``p = sigmoid(baseline + sum_t leaf_t(x))``."""

    def __init__(
        self,
        *,
        split_feature: np.ndarray,
        split_bin: np.ndarray,
        missing_left: np.ndarray,
        leaves: QuantizedVector,
        bin_edges: Sequence[np.ndarray],
        baseline: float,
        feature_names: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
        chunk_size: int = 4096,
    ) -> None:
        n_trees, n_internal = split_feature.shape
        depth = int(np.log2(n_internal + 1))
        if n_internal != 2**depth - 1:
            raise ModelFormatError(f"{n_internal} internal nodes is not a perfect tree")
        if split_bin.shape != split_feature.shape or missing_left.shape != split_feature.shape:
            raise ModelFormatError("split tensors must share the same shape")
        if leaves.size != n_trees * 2**depth:
            raise ModelFormatError(f"expected {n_trees * 2**depth} leaves, got {leaves.size}")
        if len(bin_edges) != len(feature_names):
            raise ModelFormatError("one bin-edge vector per feature is required")
        if int(split_feature.max(initial=0)) >= len(feature_names):
            raise ModelFormatError("split on an unknown feature index")

        self.n_trees, self.depth = n_trees, depth
        self.n_internal, self.n_leaves = n_internal, 2**depth
        self.feature_names = tuple(feature_names)
        self.baseline = float(baseline)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.leaves = leaves
        self.chunk_size = chunk_size
        self._feature = np.ascontiguousarray(split_feature, dtype=np.uint8).reshape(-1)
        self._bin = np.ascontiguousarray(split_bin, dtype=np.uint8).reshape(-1)
        self._missing_left = np.ascontiguousarray(missing_left, dtype=bool).reshape(-1)
        self._edges = tuple(np.ascontiguousarray(e, dtype=np.float64) for e in bin_edges)
        if any(len(e) >= MISSING_BIN for e in self._edges):
            raise ModelFormatError("too many bins for a uint8 feature")
        self._tree_node_base = np.arange(n_trees, dtype=np.intp) * n_internal
        self._tree_leaf_base = np.arange(n_trees, dtype=np.intp) * self.n_leaves - n_internal

    # ------------------------------------------------------------------ inference

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def leaf_type(self) -> GGMLType:
        return self.leaves.qtype

    @property
    def nbytes(self) -> int:
        """Resident size of the model parameters (bytes)."""
        structure = self._feature.nbytes + self._bin.nbytes + self._missing_left.nbytes
        return structure + self.leaves.nbytes + sum(e.nbytes for e in self._edges)

    def bin_features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected an array of shape (n, {self.n_features}), got {X.shape}")
        binned = np.empty(X.shape, dtype=np.uint8)
        for j, edges in enumerate(self._edges):
            column = X[:, j]
            binned[:, j] = np.searchsorted(edges, column, side="left")
            missing = np.isnan(column)
            if missing.any():
                binned[missing, j] = MISSING_BIN
        return binned

    def _leaf_index(self, binned: np.ndarray) -> np.ndarray:
        rows = np.arange(binned.shape[0], dtype=np.intp)[:, None]
        node = np.zeros((binned.shape[0], self.n_trees), dtype=np.intp)
        for _ in range(self.depth):
            flat = self._tree_node_base + node
            value = binned[rows, self._feature[flat]]
            go_left = (value <= self._bin[flat]) | (
                (value == MISSING_BIN) & self._missing_left[flat]
            )
            node = 2 * node + 2 - go_left
        return self._tree_leaf_base + node  # flat index into the leaf vector

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw log-odds of fraud, shape (n,)."""
        binned = self.bin_features(X)
        out = np.empty(binned.shape[0], dtype=np.float64)
        for start in range(0, binned.shape[0], self.chunk_size):
            chunk = binned[start : start + self.chunk_size]
            leaf_values = self.leaves.gather(self._leaf_index(chunk))
            out[start : start + len(chunk)] = leaf_values.sum(axis=1, dtype=np.float64)
        return out + self.baseline

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive (fraud) class, shape (n,)."""
        return np.exp(-np.logaddexp(0.0, -self.decision_function(X)))

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path, extra_metadata: Mapping[str, Any] | None = None) -> int:
        """Serialize to a self-contained GGUF v3 file; returns the file size."""
        metadata: dict[str, Any] = {
            "general.architecture": ARCHITECTURE,
            **{k: v for k, v in self.metadata.items() if not k.startswith("gbdt.")},
            **dict(extra_metadata or {}),
        }
        if self.leaf_type in _FILE_TYPE:
            metadata["general.file_type"] = U32(_FILE_TYPE[self.leaf_type])
        if self.leaf_type in (GGMLType.Q4_0, GGMLType.Q8_0):
            metadata["general.quantization_version"] = U32(_QUANTIZATION_VERSION)
        metadata |= {
            "gbdt.tree_count": self.n_trees,
            "gbdt.tree_depth": self.depth,
            "gbdt.feature_count": self.n_features,
            "gbdt.feature_names": list(self.feature_names),
            "gbdt.baseline_logit": self.baseline,
            "gbdt.missing_bin": MISSING_BIN,
            "gbdt.leaf_type": self.leaf_type.name,
            "gbdt.uint8_tensors": list(_UINT8_TENSORS),
        }
        shape = (self.n_trees, self.n_internal)
        offsets = np.cumsum([0, *(len(e) for e in self._edges)], dtype=np.int64)
        edges = np.concatenate(self._edges) if offsets[-1] else np.zeros(1)
        tensors = {
            "gbdt.split_feature": Tensor.plain(
                self._feature.reshape(shape).view(np.int8), GGMLType.I8
            ),
            "gbdt.split_bin": Tensor.plain(self._bin.reshape(shape).view(np.int8), GGMLType.I8),
            "gbdt.missing_left_bits": Tensor.plain(
                np.packbits(self._missing_left).view(np.int8), GGMLType.I8
            ),
            "gbdt.leaf_value": Tensor(
                self.leaves.storage().reshape(self.n_trees, -1),
                self.leaf_type,
                (self.n_trees, self.n_leaves),
            ),
            "binning.edges": Tensor.plain(edges, GGMLType.F64),
            "binning.offsets": Tensor.plain(offsets.astype(np.int32), GGMLType.I32),
        }
        return write_gguf(path, metadata, tensors)

    @classmethod
    def load(cls, path: str | Path) -> QuantizedGBDT:
        gguf = read_gguf(path)
        md, tensors = gguf.metadata, gguf.tensors
        if md.get("general.architecture") != ARCHITECTURE:
            raise ModelFormatError(
                f"not a '{ARCHITECTURE}' model: {md.get('general.architecture')}"
            )
        try:
            n_trees, depth = int(md["gbdt.tree_count"]), int(md["gbdt.tree_depth"])
            feature = tensors["gbdt.split_feature"].data.view(np.uint8)
            split_bin = tensors["gbdt.split_bin"].data.view(np.uint8)
            bits = tensors["gbdt.missing_left_bits"].data.view(np.uint8)
            leaf = tensors["gbdt.leaf_value"]
            edges = tensors["binning.edges"].data
            offsets = tensors["binning.offsets"].data.astype(np.int64)
        except KeyError as exc:
            raise ModelFormatError(f"missing GGUF entry: {exc}") from exc
        n_internal = 2**depth - 1
        missing_left = np.unpackbits(bits, count=n_trees * n_internal).astype(bool)
        return cls(
            split_feature=feature.reshape(n_trees, n_internal),
            split_bin=split_bin.reshape(n_trees, n_internal),
            missing_left=missing_left.reshape(n_trees, n_internal),
            leaves=QuantizedVector(leaf.data.reshape(-1), leaf.ggml_type, leaf.n_elements),
            bin_edges=[edges[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)],
            baseline=float(md["gbdt.baseline_logit"]),
            feature_names=md["gbdt.feature_names"],
            metadata=md,
        )
