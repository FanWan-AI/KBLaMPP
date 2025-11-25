"""Wrapper to inject KBLaM++ into a transformer model.

This module defines a convenience class that wraps a HuggingFace
decoder‑only transformer (e.g. LLaMA or Qwen) and inserts the
KBLaM++ components at specified layers.  During the forward pass
it performs the following steps:

1.  Run the backbone through the first ``L`` layers until the
    specified injection layer.
2.  Compute query vectors ``Q`` from the hidden states of the
    injection layer via a linear projection.
3.  For each token/time step, perform an ANN lookup on the
    knowledge index to obtain candidate keys and values.
4.  Use the ``KBSelector`` to compute attention weights and the
    aggregated knowledge vector.
5.  Project the knowledge vector to the model dimension and fuse
    with the hidden state using ``KBFusionLayer``.
6.  Continue the backbone through the remaining layers.

Only a single injection point is implemented in this Plan B version.
Multiple injection points can be added by repeating this pattern.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .knowledge_index import KnowledgeIndex
from .kb_store import KBValueStore
from .selector import KBSelector
from .fusion import KBFusionLayer


class KBInjectedModel(nn.Module):
    """Wrap a transformer with KBLaM++ injection.

    Parameters
    ----------
    backbone : PreTrainedModel
        A decoder‑only transformer (e.g. Llama, Qwen).
    inject_layer : int
        Index of the layer at which to inject knowledge.
    selector : KBSelector
        Module to compute attention weights over KB candidates.
    fusion : KBFusionLayer
        Module to fuse knowledge output with transformer output.
    index : KnowledgeIndex
        ANN index containing key vectors.
    kb_store : KBValueStore
        Data store for key/value vectors and metadata.
    """
    def __init__(
        self,
        backbone: PreTrainedModel,
        inject_layer: int,
        selector: KBSelector,
        fusion: KBFusionLayer,
        index: KnowledgeIndex,
        kb_store: KBValueStore,
        d_k: int,
        d_v: int,
        d_ctx: int,
        k_top: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.inject_layer = inject_layer
        self.selector = selector
        self.fusion = fusion
        self.index = index
        self.kb_store = kb_store
        self.d_ctx = d_ctx
        self.k_top = k_top
        self.linear_q = nn.Linear(backbone.config.hidden_size, d_k, bias=False)
        self.linear_v = nn.Linear(d_v, backbone.config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        question_time: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with knowledge injection.

        Note: In this Plan B stub, we assume a single injection layer.
        You must implement splitting the forward pass around the layer.
        See the comments below for guidance.
        """
        # 1. Run embedding and first few layers
        embed = self.backbone.get_input_embeddings()
        hidden_states = embed(input_ids)

        # Prepare decoder masks/positional encodings just like the original model
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        decoder_attention_mask = attention_mask
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        decoder_attention_mask = self.backbone.model._prepare_decoder_attention_mask(  # type: ignore[attr-defined]
            decoder_attention_mask,
            (batch_size, seq_len),
            hidden_states,
            past_key_values_length=0,
        )
        if hasattr(self.backbone.model, "rotary_emb"):
            position_embeddings = self.backbone.model.rotary_emb(hidden_states, seq_len=seq_len)  # type: ignore[attr-defined]
        else:
            position_embeddings = None

        # We run through each block up to inject_layer - 1
        blocks: List[nn.Module] = list(self.backbone.model.layers)
        for i, block in enumerate(blocks):
            if i == self.inject_layer:
                break
            hidden_states = block(
                hidden_states,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # 2. Compute query vectors at injection layer
        Q = self.linear_q(hidden_states)  # [B,T,d_k]
        B, T, d_k = Q.shape
        # 3. Flatten Q and perform ANN search.  Detaching before the FAISS call
        # avoids unnecessary autograd tracking whilst keeping the GPU<->CPU
        # transfer explicit.
        Q_flat = Q.detach().cpu().numpy().reshape(B*T, d_k)
        _, idx = self.index.query(Q_flat.astype('float32'), self.k_top)
        # idx: [B*T,K]
        # 4. Fetch KB entries from store
        K_kb, V_kb, ctx_vec, rel_id, ent_id, tau_min, tau_max = self.kb_store.fetch(idx)
        # Reshape to [B,T,K,?]
        K_kb = K_kb.view(B, T, -1, d_k)
        V_kb = V_kb.view(B, T, -1, self.linear_v.in_features)
        ctx_vec = ctx_vec.view(B, T, self.k_top, self.d_ctx)
        rel_id = rel_id.view(B, T, -1)
        ent_id = ent_id.view(B, T, -1)
        tau_min = tau_min.view(B, T, -1)
        tau_max = tau_max.view(B, T, -1)

        # 5. Use selector to compute alpha and V_tilde
        if question_time is None:
            q_min = q_max = torch.zeros((B, T), device=hidden_states.device)
        else:
            question_time = question_time.to(hidden_states.device)
            q_min = question_time[:, 0].unsqueeze(1).expand(-1, T)
            q_max = question_time[:, 1].unsqueeze(1).expand(-1, T)
        alpha, V_tilde = self.selector(Q, K_kb, V_kb, ctx_vec, rel_id, ent_id, q_min, q_max, tau_min, tau_max)
        # 6. Project V_tilde to model dimension
        V_proj = self.linear_v(V_tilde)  # [B,T,d_model]
        # 7. Fuse with hidden_states using KBFusionLayer
        hidden_states, beta = self.fusion(hidden_states, V_proj, attn_mask=attention_mask)
        # 8. Continue running the remaining layers
        for j in range(self.inject_layer, len(blocks)):
            hidden_states = blocks[j](
                hidden_states,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        # 9. Final LM head
        logits = self.backbone.lm_head(hidden_states)
        return logits