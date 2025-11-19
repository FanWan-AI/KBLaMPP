"""Scoring modules for KBLaM++.

This file defines two small neural modules used to compute additional
scores for each candidate knowledge entry returned by the ANN:

* :class:`ContextScorer` – measures whether the candidate's context
  sentence matches the current token's semantic context (the query
  vector).
* :class:`TimeScorer` – measures whether the candidate's time window
  overlaps well with the query's time window.

Both modules return a tensor of shape (B, T, K), one score per
candidate for each token in the batch.

In Plan B these modules are intentionally simple.  You may want to
replace them with more sophisticated models as you scale up.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class ContextScorer(nn.Module):
    """Compute a context matching score for each candidate.

    Given a query vector ``Q[b,t]`` for each token and the context
    sentence embeddings of each candidate ``ctx_vec[b,t,k]``, this
    module concatenates them with optional relation and entity ID
    embeddings and feeds them through a small MLP to produce a
    single score per candidate.

    Parameters
    ----------
    d_q : int
        Dimensionality of the query vector.
    d_ctx : int
        Dimensionality of the context sentence embedding.
    num_rel : int
        Size of the relation ID embedding vocabulary.  If zero
        (default), relation embeddings are not used.
    num_ent : int
        Size of the entity ID embedding vocabulary.  If zero
        (default), entity embeddings are not used.
    d_hid : int
        Hidden dimension of the MLP.
    """

    def __init__(self, d_q: int, d_ctx: int, num_rel: int = 0, num_ent: int = 0, d_hid: int = 128) -> None:
        super().__init__()
        self.use_rel = num_rel > 0
        self.use_ent = num_ent > 0
        self.rel_emb = nn.Embedding(num_rel, 16) if self.use_rel else None
        self.ent_emb = nn.Embedding(num_ent, 16) if self.use_ent else None
        in_dim = d_q + d_ctx
        if self.use_rel:
            in_dim += 16
        if self.use_ent:
            in_dim += 16
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, 1)
        )

    def forward(
        self,
        Q: torch.Tensor,         # [B, T, d_q]
        ctx_vec: torch.Tensor,    # [B, T, K, d_ctx]
        rel_id: torch.Tensor,     # [B, T, K]
        ent_id: torch.Tensor      # [B, T, K]
    ) -> torch.Tensor:
        B, T, K, d_ctx = ctx_vec.shape
        d_q = Q.size(-1)
        # Expand Q to [B, T, K, d_q]
        Q_expand = Q.unsqueeze(2).expand(-1, -1, K, -1)
        feats = [Q_expand, ctx_vec]
        if self.use_rel:
            feats.append(self.rel_emb(rel_id))  # [B,T,K,16]
        if self.use_ent:
            feats.append(self.ent_emb(ent_id))  # [B,T,K,16]
        x = torch.cat(feats, dim=-1)  # [B,T,K,in_dim]
        scores = self.mlp(x).squeeze(-1)  # [B,T,K]
        return scores


class TimeScorer(nn.Module):
    """Compute a time compatibility score for each candidate.

    This module measures how well a candidate fact's time window
    overlaps with a query's time window.  It implements a simple
    overlap + Gaussian penalty heuristic:

    .. math::

        g_{ij} = \lambda_{\mathrm{IoU}} \cdot \mathrm{IoU}(\tau_q, \tau_i)
                  + \lambda_\Delta \cdot \exp\bigl(-\Delta_{ij}^2 / (2\sigma^2)\bigr)

    where ``IoU`` is the interval overlap ratio and ``Delta`` is the
    distance between the centres of the two intervals.  The output
    shape is [B, T, K].  The parameters ``lambda_iou`` and
    ``lambda_delta`` control the relative weight of the two
    components.
    """

    def __init__(self, lambda_iou: float = 1.0, lambda_delta: float = 1.0, sigma: float = 365.0) -> None:
        super().__init__()
        self.lambda_iou = lambda_iou
        self.lambda_delta = lambda_delta
        self.sigma = sigma

    def forward(
        self,
        q_min: torch.Tensor,      # [B, T] or [B, T, 1]
        q_max: torch.Tensor,      # [B, T] or [B, T, 1]
        tau_min: torch.Tensor,    # [B, T, K]
        tau_max: torch.Tensor     # [B, T, K]
    ) -> torch.Tensor:
        # broadcast q_min/q_max to [B,T,K]
        qmin = q_min.unsqueeze(-1).expand_as(tau_min)
        qmax = q_max.unsqueeze(-1).expand_as(tau_max)
        inter = (torch.min(qmax, tau_max) - torch.max(qmin, tau_min)).clamp(min=0)
        union = (torch.max(qmax, tau_max) - torch.min(qmin, tau_min)).clamp(min=1e-8)
        iou = inter / union  # [B,T,K]
        q_cent = (qmin + qmax) * 0.5
        t_cent = (tau_min + tau_max) * 0.5
        delta = (q_cent - t_cent).abs()  # [B,T,K]
        gaussian = torch.exp(-(delta ** 2) / (2 * (self.sigma ** 2)))
        score = self.lambda_iou * iou + self.lambda_delta * gaussian
        return score