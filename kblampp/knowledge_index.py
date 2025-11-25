"""Approximate nearest neighbour index wrapper.

This module defines a small wrapper around FAISS to perform approximate
nearest neighbour (ANN) searches over an array of key vectors.  It
supports HNSW and IVF‑PQ indices.  The interface is designed to be
easily swapped out for exact search or other ANN libraries.

For Plan B we keep the implementation deliberately simple.  The
``fit`` method takes a NumPy array of shape (N, d_k) and builds the
index.  ``query`` takes a float32 array of shape (M, d_k) and
returns the top k nearest neighbours and their scores.

You must have ``faiss`` installed for this module to work.  To keep
Plan B lightweight, you can install ``faiss-cpu`` (it doesn’t require
a GPU).  If ``faiss`` is unavailable, the module will raise an
ImportError.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import os
import numpy as np


def _force_brute() -> bool:
    return os.environ.get("KBLAMPP_FORCE_BRUTE_INDEX", "0").lower() in {"1", "true", "yes"}


try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - faiss missing or unusable
    faiss = None  # type: ignore


def _faiss_available() -> bool:
    return faiss is not None and not _force_brute()


@dataclass
class KnowledgeIndex:
    """Wrapper around FAISS index for key vectors.

    Attributes
    ----------
    dim : int
        The dimensionality of the key vectors.
    method : str
        Which ANN method to use: ``"hnsw"`` or ``"ivfpq"``.
    similarity : str
        Which similarity to optimise: currently only ``"cosine"`` is
        supported.  Cosine similarity is implemented via L2 normalisation
        and inner products.
    index : Optional[faiss.Index]
        The underlying FAISS index.  Created in ``fit``.
    """

    dim: int
    method: str = "hnsw"
    similarity: str = "cosine"
    index: Optional["faiss.Index"] = None  # type: ignore[name-defined]
    _bruteforce_keys: Optional[np.ndarray] = None

    def _l2_normalise(self, X: np.ndarray) -> np.ndarray:
        """Optionally L2 normalise key/query vectors for cosine similarity.

        The FAISS inner-product indices operate on raw dot products, so to
        approximate cosine similarity we normalise each vector to unit length
        before inserting/querying.  The tiny epsilon avoids division by zero
        for degenerate vectors that occasionally show up during early
        experiments.
        """
        if self.similarity != "cosine":
            return X
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / norms

    def fit(self, keys: np.ndarray) -> None:
        """Build the ANN index from an array of key vectors.

        Parameters
        ----------
        keys : np.ndarray of shape (N, dim)
            Key vectors.  They should be float32.  If using cosine
            similarity, they will be L2‑normalised.
        """
        assert keys.dtype == np.float32, "Keys must be float32"
        keys_norm = self._l2_normalise(keys)

        if not _faiss_available():
            self._bruteforce_keys = keys_norm.copy()
            self.index = None
            return

        if self.method == "hnsw":
            # HNSW works well for medium-sized stores (<1M entries) and keeps
            # build/query times predictable for Plan B scale experiments.
            index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.add(keys_norm)
            index.hnsw.efSearch = 64
        elif self.method == "ivfpq":
            # IVFPQ builder for larger datasets where memory becomes an issue.
            # The parameters below are conservative defaults; feel free to tune
            # them for your KB density/profile.
            nlist = 4096
            m = 16  # PQ subdivisions
            nbits = 8
            quantiser = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFPQ(quantiser, self.dim, nlist, m, nbits)
            index.train(keys_norm)
            index.add(keys_norm)
            index.nprobe = 32
        else:
            raise ValueError(f"Unknown ANN method: {self.method}")

        self.index = index

    def query(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Query the index for nearest neighbours.

        Parameters
        ----------
        queries : np.ndarray of shape (M, dim)
            Query vectors.  If similarity is cosine, they will be normalised.
        k : int
            Number of neighbours to return.

        Returns
        -------
        distances : np.ndarray of shape (M, k)
            The similarity scores (inner products).
        indices : np.ndarray of shape (M, k)
            The indices of the nearest neighbours in the original key array.
        """
        if self.index is None and self._bruteforce_keys is None:
            raise RuntimeError("Index has not been built.  Call fit() first.")
        assert queries.shape[1] == self.dim, "Query dimension mismatch"
        # Convert to float32 because FAISS bindings expect contiguous float32
        # buffers; float16/64 inputs would otherwise trigger subtle segfaults.
        queries_norm = self._l2_normalise(queries.astype(np.float32))
        if self.index is not None:
            D, I = self.index.search(queries_norm, k)
            return D, I
        assert self._bruteforce_keys is not None
        keys = self._l2_normalise(self._bruteforce_keys)
        scores = queries_norm @ keys.T  # [M, N]
        top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        # argpartition gives unordered set; sort within top-k for consistency
        top_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        I = np.take_along_axis(top_idx, order, axis=1)
        D = np.take_along_axis(top_scores, order, axis=1)
        return D, I

    def save(self, path: str) -> None:
        """Save the index to disk along with meta information."""
        meta = {
            "dim": self.dim,
            "method": self.method,
            "similarity": self.similarity,
        }
        np.save(f"{path}/meta.npy", meta, allow_pickle=True)
        if self.index is not None:
            faiss.write_index(self.index, f"{path}/index.faiss")
        else:  # pragma: no cover - brute force fallback
            # Store the dense keys to allow loading without FAISS
            assert self._bruteforce_keys is not None
            np.save(f"{path}/keys.npy", self._bruteforce_keys)

    @classmethod
    def load(cls, path: str) -> "KnowledgeIndex":
        """Load an index from disk."""
        meta = np.load(f"{path}/meta.npy", allow_pickle=True).item()
        obj = cls(dim=meta["dim"], method=meta["method"], similarity=meta["similarity"])
        index_path = Path(path) / "index.faiss"
        if _faiss_available() and index_path.exists():
            try:
                obj.index = faiss.read_index(str(index_path))
                return obj
            except Exception as exc:  # pragma: no cover - runtime CPU mismatch
                print(
                    f"[knowledge_index] Failed to load FAISS index ({exc}); falling back to brute-force",
                    flush=True,
                )
        # Brute-force fallback using stored keys (expect parent dir holds K.npy)
        store_root = Path(path).parent
        keys_path = store_root / "K.npy"
        if not keys_path.exists():
            raise FileNotFoundError(
                f"Brute-force index fallback requires {keys_path}, but it was not found"
            )
        obj._bruteforce_keys = np.load(keys_path).astype(np.float32)
        return obj