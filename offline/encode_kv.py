"""Encode five‑tuple facts into key/value vectors for KBLaM++.

This script reads a JSONL file containing five‑tuples and converts
each record into a pair of numeric vectors: a key vector ``K_i``
and a value vector ``V_i``.  It also stores context sentence
embeddings, time window boundaries and optional IDs in a meta
directory.  The key/value arrays are saved as NumPy files in
``store/`` and can subsequently be fed into FAISS.

The exact embedding model (e.g. MiniLM, BGE‑small) and the
dimension sizes for keys and values are controlled by the YAML
configuration file.  See ``configs/backbone_llama1b.yaml`` for
examples.

This script is currently a stub: it outlines how you should
structure the processing.  Replace the TODO blocks with calls to
HuggingFace models or your chosen sentence encoder.  You may
optionally implement batching for efficiency.
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel


def encode_sentences(texts: List[str], model_name: str) -> np.ndarray:
    """Encode a list of sentences into embeddings.

    TODO: Load the HuggingFace model once outside of this function,
    feed the tokenised sentences into it and extract the sentence
    embeddings (e.g. the pooler output or mean pooling).  For Plan B
    we use a dummy implementation that returns random vectors.
    """
    d = 384
    return np.random.randn(len(texts), d).astype("float32")


def main():
    parser = argparse.ArgumentParser(description="Encode five‑tuples into K/V and meta arrays")
    parser.add_argument("input", type=str, help="Path to five‑tuple JSONL")
    parser.add_argument("output_dir", type=str, help="Directory to write K.npy, V.npy and meta/")
    parser.add_argument("--d_k", type=int, default=384, help="Dimensionality of key vectors")
    parser.add_argument("--d_v", type=int, default=384, help="Dimensionality of value vectors")
    parser.add_argument("--d_ctx", type=int, default=384, help="Dimensionality of context embeddings")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Sentence embedding model")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(exist_ok=True)

    # Read five‑tuples
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
            # For time windows, parse YYYY-MM-DD into an integer day offset
            start = rec["time_window"].get("start")
            end = rec["time_window"].get("end")
            # TODO: Convert date strings to numeric values (e.g. days since epoch)
            tau_min_list.append(0.0 if start is None else 0.0)
            tau_max_list.append(0.0 if end is None else 0.0)

    N = len(heads)
    d_k, d_v, d_ctx = args.d_k, args.d_v, args.d_ctx
    # TODO: Use encode_sentences() to obtain actual embeddings for each field
    e_h = encode_sentences(heads, args.model)
    e_r = encode_sentences(rels, args.model)
    e_t = encode_sentences(tails, args.model)
    e_c = encode_sentences(contexts, args.model)
    # TODO: Define MLP_k and MLP_v.  For now, use random projections
    K = np.random.randn(N, d_k).astype("float32")
    V = np.random.randn(N, d_v).astype("float32")
    # Save arrays
    np.save(output_dir / "K.npy", K)
    np.save(output_dir / "V.npy", V)
    np.save(output_dir / "meta" / "ctx_vec.npy", e_c)
    np.save(output_dir / "meta" / "tau_min.npy", np.array(tau_min_list, dtype="float32"))
    np.save(output_dir / "meta" / "tau_max.npy", np.array(tau_max_list, dtype="float32"))
    # IDs are optional; leave empty
    np.save(output_dir / "meta" / "entity_ids.npy", np.zeros(N, dtype=np.int64))
    np.save(output_dir / "meta" / "rel_ids.npy", np.zeros(N, dtype=np.int64))
    print(f"Encoded {N} five‑tuples")


if __name__ == "__main__":
    main()