"""Build a FAISS index from key vectors.

This script loads ``K.npy`` from the store directory, initialises a
``KnowledgeIndex`` and writes the index back to disk.  It is a
thin wrapper around ``kblampp.knowledge_index.KnowledgeIndex``.  In
Plan B, we typically run this once after encoding the keys and
before training.

Example usage:

```
python offline/build_index.py --store_dir store --method hnsw
```
"""

import argparse
from pathlib import Path
import numpy as np
from kblampp.knowledge_index import KnowledgeIndex


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for KBLaM++")
    parser.add_argument("--store_dir", type=str, default="store", help="Directory containing K.npy")
    parser.add_argument("--method", type=str, default="hnsw", choices=["hnsw", "ivfpq"], help="ANN method")
    args = parser.parse_args()
    store = Path(args.store_dir)
    K_path = store / "K.npy"
    K = np.load(K_path).astype("float32")
    print(f"Loaded keys of shape {K.shape}")
    ki = KnowledgeIndex(dim=K.shape[1], method=args.method, similarity="cosine")
    ki.fit(K)
    out_dir = store / "index_hnsw"
    out_dir.mkdir(exist_ok=True)
    ki.save(str(out_dir))
    print(f"Index saved to {out_dir}")


if __name__ == "__main__":
    main()