"""Knowledge/text attention fusion layer.

The :class:`KBFusionLayer` is responsible for merging the
knowledge branch with the standard attention branch in a
transformer.  It maintains two multi‑head attention modules:

* ``text_mha`` – the original self‑attention from the backbone.  It
  produces the baseline output for the token.
* ``kb_mha`` – a separate attention that uses the same query and
  key but receives its values from the knowledge injection.  This
  allows the model to attend to the KB values independently.

The fusion is controlled by a per‑token gate ``beta_j ∈ [0,1]``
learned via a small linear head.  The final output is:

``Y_j = Y_text,j + beta_j * Y_kb,j``.

Note: In this Plan B skeleton we instantiate new MHA modules
instead of sharing parameters with the backbone.  This keeps the
module lightweight and makes it independent from the backbone’s
implementation.  In a production system you may choose to share
weights or to place the fusion inside an existing block.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KBFusionLayer(nn.Module):
    """Fuse text and knowledge attention outputs via a gate."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.text_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.kb_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.beta_head = nn.Linear(d_model, 1)
        # Start with a negative bias so the knowledge gate is closed by default
        nn.init.constant_(self.beta_head.bias, -2.0)

    def forward(
        self,
        h: torch.Tensor,
        V_kb_tilde: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fused output.

        Parameters
        ----------
        h : torch.Tensor of shape [B, T, d_model]
            Hidden states from the backbone at the injection layer.
        V_kb_tilde : torch.Tensor of shape [B, T, d_model]
            Projected knowledge values.
        key_padding_mask : torch.Tensor or None
            Boolean mask with shape [B, T]; True positions are ignored.

        Returns
        -------
        Y : torch.Tensor [B, T, d_model]
            The fused token representations.
        beta : torch.Tensor [B, T, 1]
            The gate values applied to knowledge attention.
        """
        # text branch: behaves like the original self-attention to preserve the
        # backbone's behaviour when no knowledge is injected.
        Y_txt, _ = self.text_mha(h, h, h, key_padding_mask=key_padding_mask)
        # knowledge branch (Q,K use backbone states, V comes from KB) so the
        # model can treat injected information as an auxiliary attention stream.
        Y_kb, _ = self.kb_mha(h, h, V_kb_tilde, key_padding_mask=key_padding_mask)
        # gate
        beta = torch.sigmoid(self.beta_head(h))  # [B,T,1]
        # fuse the two streams; gating keeps gradients well-behaved when the KB
        # content is noisy because the model can down-weight it token-wise.
        Y = Y_txt + beta * Y_kb
        return Y, beta