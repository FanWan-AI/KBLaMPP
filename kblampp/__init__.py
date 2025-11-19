# Initialize the kblampp package.

"""
This package contains the core building blocks of KBLaM++.

* ``knowledge_index.py`` – wraps a FAISS index for approximate
  nearest neighbour search over key vectors.
* ``kb_store.py`` – loads key, value and metadata arrays from disk
  and provides a batch indexing API.
* ``scorers.py`` – defines ContextScorer and TimeScorer used for
  re‑ranking the ANN candidates.
* ``selector.py`` – combines semantic, context and time scores to
  produce a probability distribution over candidates and returns
  the weighted sum of Value vectors.
* ``fusion.py`` – defines KBFusionLayer, which merges the
  knowledge branch with the standard attention branch.
* ``injection_wrapper.py`` – wraps a transformer model to insert
  KBLaM++ injection at specified layers.

Note: These modules provide **stubs** and interface definitions.
They do not implement full functionality.  You must fill in the
algorithms (e.g. scoring heuristics, ANN queries, tensor
operations) to suit your needs.
"""
