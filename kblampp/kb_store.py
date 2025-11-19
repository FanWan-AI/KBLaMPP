"""Key/Value store for KBLaM++.

The ``KBValueStore`` loads key and value vectors and associated
metadata from disk and provides a method to fetch batches of them
efficiently given an array of indices.  This allows the
``KBSelector`` to retrieve candidate keys/values from an ANN index
without loading the entire matrix into GPU memory.

In a typical pipeline you will:

* Build five‑tuples from your data and encode them into key and value
  arrays saved as NumPy files.
* Load those arrays into a ``KBValueStore`` instance at runtime.
* Given indices returned by the ANN index (shape [B*T, K]), call
  ``fetch()`` to obtain the keys, values and metadata as PyTorch
  tensors on the appropriate device.

This class is deliberately lightweight; if your dataset is very
large, consider implementing memory mapping or sharding here.
"""

from __future__ import annotations

from typing import Tuple, Any
import numpy as np
import torch


class KBValueStore:
    """Stores key/value vectors and meta data and returns them by index.

    Parameters
    ----------
    root : str
        The directory containing ``K.npy``, ``V.npy`` and ``meta``.
    device : torch.device
        The device on which to return the tensors.  Typically this
        will be your GPU.
    """

    def __init__(self, root: str, device: torch.device) -> None:
        self.device = device
        self.K = np.load(f"{root}/K.npy", mmap_mode="r").astype(np.float32)
        self.V = np.load(f"{root}/V.npy", mmap_mode="r").astype(np.float32)
        # context vectors, time minima/maxima and IDs
        self.ctx_vec = np.load(f"{root}/meta/ctx_vec.npy", mmap_mode="r").astype(np.float32)
        self.tau_min = np.load(f"{root}/meta/tau_min.npy", mmap_mode="r").astype(np.float32)
        self.tau_max = np.load(f"{root}/meta/tau_max.npy", mmap_mode="r").astype(np.float32)
        # These may not exist yet in Plan B; fill with zeros
        try:
            self.rel_ids = np.load(f"{root}/meta/rel_ids.npy", mmap_mode="r")
        except FileNotFoundError:
            self.rel_ids = np.zeros((self.K.shape[0],), dtype=np.int64)
        try:
            self.ent_ids = np.load(f"{root}/meta/entity_ids.npy", mmap_mode="r")
        except FileNotFoundError:
            self.ent_ids = np.zeros((self.K.shape[0],), dtype=np.int64)

        assert self.K.shape[0] == self.V.shape[0] == self.ctx_vec.shape[0] == self.tau_min.shape[0] == self.tau_max.shape[0]

    def fetch(self, idx: np.ndarray) -> Tuple[torch.Tensor, ...]:
        """Fetch key, value and meta entries by indices.

        Parameters
        ----------
        idx : np.ndarray of shape (B*T, K)
            The indices of the candidates returned by the ANN index.

        Returns
        -------
        K_batch : torch.Tensor of shape (B*T, K, d_k)
        V_batch : torch.Tensor of shape (B*T, K, d_v)
        ctx_vec : torch.Tensor of shape (B*T, K, d_ctx)
        rel_id : torch.Tensor of shape (B*T, K)
        ent_id : torch.Tensor of shape (B*T, K)
        tau_min : torch.Tensor of shape (B*T, K)
        tau_max : torch.Tensor of shape (B*T, K)
        """
        assert idx.ndim == 2
        # Use advanced indexing into the numpy memmaps
        K_batch = torch.from_numpy(self.K[idx]).to(self.device)
        V_batch = torch.from_numpy(self.V[idx]).to(self.device)
        ctx_batch = torch.from_numpy(self.ctx_vec[idx]).to(self.device)
        rel_batch = torch.from_numpy(self.rel_ids[idx]).to(self.device)
        ent_batch = torch.from_numpy(self.ent_ids[idx]).to(self.device)
        tau_min_batch = torch.from_numpy(self.tau_min[idx]).to(self.device)
        tau_max_batch = torch.from_numpy(self.tau_max[idx]).to(self.device)
        return (K_batch, V_batch, ctx_batch, rel_batch, ent_batch, tau_min_batch, tau_max_batch)