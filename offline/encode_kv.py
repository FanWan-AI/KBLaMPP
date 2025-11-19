"""Encode KBLaM++ five-tuples into key/value arrays and metadata."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

HAS_TORCH = torch is not None and nn is not None and AutoModel is not None and AutoTokenizer is not None


def parse_timestamp(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return 0.0
    epoch = datetime(1970, 1, 1)
    return (dt - epoch).total_seconds()


if HAS_TORCH:
    class TauEncoder(nn.Module):
        def __init__(self, d_tau: int) -> None:
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(4, d_tau),
                nn.ReLU(),
                nn.Linear(d_tau, d_tau),
            )

        def forward(self, tau_min, tau_max):
            duration = (tau_max - tau_min).clamp(min=0.0)
            centre = (tau_min + tau_max) * 0.5
            features = torch.stack([tau_min, tau_max, duration, centre], dim=-1)
            return self.mlp(features)
else:  # pragma: no cover
    class TauEncoder:
        def __init__(self, d_tau: int) -> None:
            rng = np.random.default_rng(0)
            self.weights = rng.standard_normal((4, d_tau)).astype("float32")
            self.bias = np.zeros(d_tau, dtype="float32")

        def __call__(self, tau_min: np.ndarray, tau_max: np.ndarray) -> np.ndarray:
            duration = np.clip(tau_max - tau_min, 0.0, None)
            centre = (tau_min + tau_max) * 0.5
            features = np.stack([tau_min, tau_max, duration, centre], axis=-1)
            return features.astype("float32") @ self.weights + self.bias


def encode_texts(
    texts: List[str],
    *,
    tokenizer=None,
    model=None,
    device=None,
    batch_size: int = 64,
    max_length: int = 128,
    fallback_dim: Optional[int] = None,
    fallback_seed: int = 0,
) -> np.ndarray:
    if HAS_TORCH and tokenizer is not None and model is not None and device is not None:
        model.eval()
        outputs: List[Any] = []
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
        return torch.cat(outputs, dim=0).numpy().astype("float32")
    if fallback_dim is None:
        raise ValueError("fallback_dim must be set when torch is not available")
    rng = np.random.default_rng(fallback_seed)
    return rng.standard_normal((len(texts), fallback_dim)).astype("float32")


def fallback_random_embeddings(size: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size).astype("float32")


def load_five_tuples(path: Path) -> tuple[List[str], List[str], List[str], List[str], List[float], List[float]]:
    heads: List[str] = []
    rels: List[str] = []
    tails: List[str] = []
    contexts: List[str] = []
    tau_min_list: List[float] = []
    tau_max_list: List[float] = []
    with path.open(encoding="utf-8") as f:
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
    return heads, rels, tails, contexts, tau_min_list, tau_max_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode KBLaM++ five-tuples into key/value arrays")
    parser.add_argument("input", type=str, help="Path to five-tuple JSONL")
    parser.add_argument("output_dir", type=str, help="Directory to write K/V and meta")
    parser.add_argument("--d_k", type=int, default=384, help="Key vector dimension")
    parser.add_argument("--d_v", type=int, default=384, help="Value vector dimension")
    parser.add_argument("--d_ctx", type=int, default=384, help="Context embedding dimension")
    parser.add_argument("--d_tau", type=int, default=32, help="Time embedding dimension")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Sentence encoder name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for sentence encoder")
    parser.add_argument("--max_length", type=int, default=128, help="Max tokens for sentence encoder")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for fallbacks")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)

    heads, rels, tails, contexts, tau_min_list, tau_max_list = load_five_tuples(input_path)
    N = len(heads)
    d_k, d_v, d_ctx, d_tau = args.d_k, args.d_v, args.d_ctx, args.d_tau

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(device)
        e_h = encode_texts(
            heads,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        e_r = encode_texts(
            rels,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        e_t = encode_texts(
            tails,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        e_c = encode_texts(
            contexts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        mlp_k = nn.Sequential(
            nn.Linear(e_h.shape[1] + e_r.shape[1], max(d_k * 2, e_h.shape[1] + e_r.shape[1])),
            nn.ReLU(),
            nn.Linear(max(d_k * 2, e_h.shape[1] + e_r.shape[1]), d_k),
        ).to(device)
        mlp_v = nn.Sequential(
            nn.Linear(e_t.shape[1] + e_c.shape[1] + d_tau, max(d_v * 2, e_t.shape[1] + e_c.shape[1] + d_tau)),
            nn.ReLU(),
            nn.Linear(max(d_v * 2, e_t.shape[1] + e_c.shape[1] + d_tau), d_v),
        ).to(device)
        tau_encoder = TauEncoder(d_tau).to(device)
        ctx_projector = nn.Sequential(
            nn.Linear(e_c.shape[1], max(d_ctx * 2, e_c.shape[1])),
            nn.ReLU(),
            nn.Linear(max(d_ctx * 2, e_c.shape[1]), d_ctx),
        ).to(device)

        whole_h = torch.cat([torch.from_numpy(e_h).to(device), torch.from_numpy(e_r).to(device)], dim=-1)
        whole_t = torch.cat([torch.from_numpy(e_t).to(device), torch.from_numpy(e_c).to(device)], dim=-1)
        tau_min_tensor = torch.tensor(tau_min_list, dtype=torch.float32, device=device)
        tau_max_tensor = torch.tensor(tau_max_list, dtype=torch.float32, device=device)
        e_tau = tau_encoder(tau_min_tensor, tau_max_tensor)

        with torch.no_grad():
            K = mlp_k(whole_h).cpu().numpy().astype("float32")
            V = mlp_v(torch.cat([whole_t, e_tau], dim=-1)).cpu().numpy().astype("float32")
            ctx_vec = ctx_projector(torch.from_numpy(e_c).to(device)).cpu().numpy().astype("float32")
        tau_min_arr = tau_min_tensor.cpu().numpy().astype("float32")
        tau_max_arr = tau_max_tensor.cpu().numpy().astype("float32")
    else:
        print("[warning] torch not available; falling back to random embeddings", flush=True)
        K = fallback_random_embeddings((N, d_k), args.seed)
        V = fallback_random_embeddings((N, d_v), args.seed + 1)
        ctx_vec = fallback_random_embeddings((N, d_ctx), args.seed + 2)
        tau_min_arr = np.array(tau_min_list, dtype="float32")
        tau_max_arr = np.array(tau_max_list, dtype="float32")

    np.save(output_dir / "K.npy", K)
    np.save(output_dir / "V.npy", V)
    np.save(output_dir / "meta" / "ctx_vec.npy", ctx_vec)
    np.save(output_dir / "meta" / "tau_min.npy", tau_min_arr)
    np.save(output_dir / "meta" / "tau_max.npy", tau_max_arr)
    np.save(output_dir / "meta" / "entity_ids.npy", np.zeros(N, dtype=np.int64))
    np.save(output_dir / "meta" / "rel_ids.npy", np.zeros(N, dtype=np.int64))
    print(f"Encoded {N} five-tuples into {output_dir}")


if __name__ == "__main__":
    main()
