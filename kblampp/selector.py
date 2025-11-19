"""Candidate re‑scoring and value aggregation.

The ``KBSelector`` takes the query vectors ``Q[b,t]`` and a batch
of candidate keys/values/metadata, computes a score for each
candidate using the semantic similarity, context matching and time
matching modules, normalises these scores with a softmax and
returns the weighted sum of the candidate value vectors.

This module contains no references to FAISS or to the backbone
model.  It purely operates on PyTorch tensors.
"""

from __future__ import annotations

from typing import Tuple
import math
import torch
import torch.nn as nn
from .scorers import ContextScorer, TimeScorer


class KBSelector(nn.Module):
    """Combine semantic, context and time scores to select knowledge.

    Parameters
    ----------
    d_k : int
        Dimensionality of key and query vectors.
    d_ctx : int
        Dimensionality of the context sentence embeddings.
    num_rel : int
        Size of relation ID vocabulary (optional).
    num_ent : int
        Size of entity type vocabulary (optional).
    gamma : float
        Weight of context score.
    eta : float
        Weight of time score.
    temperature : float
        Temperature for softmax.  Lower values produce sharper
        distributions.
    """

    def __init__(
        self,
        d_k: int,
        d_ctx: int,
        num_rel: int = 0,
        num_ent: int = 0,
        gamma: float = 1.0,
        eta: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.eta = eta
        self.temperature = temperature
        self.ctx_scorer = ContextScorer(d_k, d_ctx, num_rel, num_ent)
        self.time_scorer = TimeScorer()

    def forward(
        self,
        Q: torch.Tensor,         # [B, T, d_k]
        K_kb: torch.Tensor,      # [B, T, K, d_k]
        V_kb: torch.Tensor,      # [B, T, K, d_v]
        ctx_vec: torch.Tensor,   # [B, T, K, d_ctx]
        rel_id: torch.Tensor,    # [B, T, K]
        ent_id: torch.Tensor,    # [B, T, K]
        q_min: torch.Tensor,     # [B, T]
        q_max: torch.Tensor,     # [B, T]
        tau_min: torch.Tensor,   # [B, T, K]
        tau_max: torch.Tensor    # [B, T, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return attention weights and value mixture.

        Returns
        -------
        alpha : torch.Tensor of shape [B, T, K]
            The normalised attention weights over candidates.
        V_tilde : torch.Tensor of shape [B, T, d_v]
            The weighted sum of Value vectors for each token.
        """
        # Semantic similarity (Q·K) / sqrt(d_k)
        sem = torch.einsum("btd,btkd->btk", Q, K_kb) / math.sqrt(Q.size(-1))
        # Context matching
        ctx_score = self.ctx_scorer(Q, ctx_vec, rel_id, ent_id)
        # Time matching
        time_score = self.time_scorer(q_min, q_max, tau_min, tau_max)
        # Combine scores
        s = sem + self.gamma * ctx_score + self.eta * time_score
        s = s - s.max(dim=-1, keepdim=True).values  # numerical stability
        alpha = torch.softmax(s / self.temperature, dim=-1)  # [B,T,K]
        # Weighted sum of values
        B, T, K, d_v = V_kb.shape
        V_tilde = torch.bmm(
            alpha.view(B*T, 1, K),
            V_kb.view(B*T, K, d_v)
        ).view(B, T, d_v)
        return alpha, V_tilde