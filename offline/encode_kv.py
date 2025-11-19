"""Encode KBLaM++ five‑tuples into key/value arrays and meta data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TauEncoder(nn.Module):
    """Simple MLP that maps raw timestamps into a continuous embedding."""

    def __init__(self, d_tau: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, d_tau),
            nn.ReLU(),
            nn.Linear(d_tau, d_tau),
        )

    def forward(self, tau_min: torch.Tensor, tau_max: torch.Tensor) -> torch.Tensor:
        duration = (tau_max - tau_min).clamp(min=0.0)
        centre = (tau_min + tau_max) * 0.5
        features = torch.stack([tau_min, tau_max, duration, centre], dim=-1)
        return self.mlp(features)


def parse_timestamp(value: Optional[str]) -> float:
    """Convert an ISO date/time string into a UNIX timestamp."""
    if not value:
        return 0.0
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return 0.0
    epoch = datetime(1970, 1, 1)
    return (dt - epoch).total_seconds()


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    model.eval()
    outputs: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        outputs.append(pooled.cpu())
    return torch.cat(outputs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode five‑tuples into KBLaM++ key/value store")
    parser.add_argument("input", type=str, help="Path to five‑tuple JSONL")
    parser.add_argument("output_dir", type=str, help="Directory to write K/V and metadata")
    parser.add_argument("--d_k", type=int, default=384, help="Dimensionality of key vectors")
    parser.add_argument("--d_v", type=int, default=384, help="Dimensionality of value vectors")
    parser.add_argument("--d_ctx", type=int, default=384, help="Dimensionality of context embeddings")
    parser.add_argument("--d_tau", type=int, default=32, help="Dimensionality of time embeddings")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Sentence embedding model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the embedding model")
    parser.add_argument("--max_length", type=int, default=128, help="Max tokens for sentence encoder")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    heads: List[str] = []
    rels: List[str] = []
    tails: List[str] = []
    contexts: List[str] = []
    tau_min_list: List[float] = []
    tau_max_list: List[float] = []
    with input_path.open() as f:
        for line in f:
            rec = json.loads(line)
            heads.append(rec["head"]["name"])
            rels.append(rec["relation"]["name"])
            tails.append(rec["tail"]["name"])
            contexts.append(rec["context"]["sent_span"])
            start = rec["time_window"].get("start")
            end = rec["time_window"].get("end")
            start_ts = parse_timestamp(start)
            end_ts = parse_timestamp(end) if end else start_ts
            tau_min_list.append(start_ts)
            tau_max_list.append(end_ts)

    d_k, d_v, d_ctx, d_tau = args.d_k, args.d_v, args.d_ctx, args.d_tau
    e_h = encode_texts(heads, tokenizer, model, device, args.batch_size, args.max_length)
    e_r = encode_texts(rels, tokenizer, model, device, args.batch_size, args.max_length)
    e_t = encode_texts(tails, tokenizer, model, device, args.batch_size, args.max_length)
    e_c = encode_texts(contexts, tokenizer, model, device, args.batch_size, args.max_length)

    mlp_k = nn.Sequential(
        nn.Linear(d_ctx * 2, max(d_k * 2, d_ctx * 2)),
        nn.ReLU(),
        nn.Linear(max(d_k * 2, d_ctx * 2), d_k),
    ).to(device)
    mlp_v = nn.Sequential(
        nn.Linear(d_ctx * 2 + d_tau, max(d_v * 2, d_ctx * 2 + d_tau)),
        nn.ReLU(),
        nn.Linear(max(d_v * 2, d_ctx * 2 + d_tau), d_v),
    ).to(device)
    tau_encoder = TauEncoder(d_tau).to(device)

    whole_h = torch.cat([e_h, e_r], dim=-1).to(device)
    whole_t = torch.cat([e_t, e_c], dim=-1).to(device)
    tau_min_tensor = torch.tensor(tau_min_list, dtype=torch.float32, device=device)
    tau_max_tensor = torch.tensor(tau_max_list, dtype=torch.float32, device=device)
    e_tau = tau_encoder(tau_min_tensor, tau_max_tensor)

    with torch.no_grad():
        K = mlp_k(whole_h).cpu().numpy().astype("float32")
        V = mlp_v(torch.cat([whole_t, e_tau], dim=-1)).cpu().numpy().astype("float32")

    ctx_vec = e_c.cpu().numpy().astype("float32")
    tau_min_arr = tau_min_tensor.cpu().numpy().astype("float32")
    tau_max_arr = tau_max_tensor.cpu().numpy().astype("float32")

    np.save(output_dir / "K.npy", K)
    np.save(output_dir / "V.npy", V)
    np.save(output_dir / "meta" / "ctx_vec.npy", ctx_vec)
    np.save(output_dir / "meta" / "tau_min.npy", tau_min_arr)
    np.save(output_dir / "meta" / "tau_max.npy", tau_max_arr)
    np.save(output_dir / "meta" / "entity_ids.npy", np.zeros(K.shape[0], dtype=np.int64))
    np.save(output_dir / "meta" / "rel_ids.npy", np.zeros(K.shape[0], dtype=np.int64))
    print(f"Encoded {K.shape[0]} five‑tuples into {output_dir}")


if __name__ == "__main__":
    main()