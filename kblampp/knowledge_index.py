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
from typing import Tuple, Optional
import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FAISS is required for KBLaM++ knowledge index.  Install faiss-cpu or faiss-gpu."
    ) from exc


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
    index: Optional[faiss.Index] = None

    def _l2_normalise(self, X: np.ndarray) -> np.ndarray:
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

        if self.method == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.add(keys_norm)
            index.hnsw.efSearch = 64
        elif self.method == "ivfpq":
            # IVFPQ builder for larger datasets
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
        if self.index is None:
            raise RuntimeError("Index has not been built.  Call fit() first.")
        assert queries.shape[1] == self.dim, "Query dimension mismatch"
        queries_norm = self._l2_normalise(queries.astype(np.float32))
        D, I = self.index.search(queries_norm, k)
        return D, I

    def save(self, path: str) -> None:
        """Save the index to disk along with meta information."""
        meta = {
            "dim": self.dim,
            "method": self.method,
            "similarity": self.similarity,
        }
        np.save(f"{path}/meta.npy", meta, allow_pickle=True)
        assert self.index is not None, "Index is not built"
        faiss.write_index(self.index, f"{path}/index.faiss")

    @classmethod
    def load(cls, path: str) -> "KnowledgeIndex":
        """Load an index from disk."""
        meta = np.load(f"{path}/meta.npy", allow_pickle=True).item()
        obj = cls(dim=meta["dim"], method=meta["method"], similarity=meta["similarity"])
        obj.index = faiss.read_index(f"{path}/index.faiss")
        return obj