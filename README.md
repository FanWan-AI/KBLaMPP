# KBLaM++ Plan B Skeleton

This repository provides a **minimal but complete skeleton** to implement a
small‑scale version of **KBLaM++**, suitable for machines with limited
GPU memory (e.g. one 2080 Ti or RTX 3060).  The goal of Plan B is
to *demonstrate* the end‑to‑end workflow of KBLaM++—from
knowledge extraction and key/value encoding through ANN lookup and
knowledge injection into a transformer—without requiring
multi‑billion‑parameter models or large clusters.  Once you verify that the
pipeline works on this small setup, you can easily switch to larger models
by editing the YAML configuration files.

## What’s Included?

The folder structure mirrors the design described in the implementation
specification.  At a high level, you will find:

* `configs/` – YAML files specifying the backbone model,
  embedding model and other hyperparameters for each experiment.
* `data/` – directories for raw source data, extracted five‑tuple
  facts and question/answer datasets.
* `store/` – binary files created during the offline stage:
  key and value matrices, FAISS indices and auxiliary metadata.
* `kblampp/` – Python modules implementing the core building blocks of
  KBLaM++: ANN wrappers, key/value stores, scoring functions,
  selection and fusion layers and a wrapper to insert knowledge
  injection into a small LLM.
* `offline/` – scripts to convert raw datasets or LLM‑generated
  worlds into the five‑tuple format, encode key/value vectors,
  build ANN indices and perform sanity checks.
* `train/` – training scripts and a basic dataloader that read
  question/answer files and execute stage A and stage B training loops.
* `eval/` – evaluation utilities to measure EM/F1 on QA tasks,
  export evidence chains and visualise attention weights.
* `infer/` – a stub showing how to expose the model via a simple
  inference API or CLI.

This skeleton contains **place‑holders and scaffolding**—you will need to
implement the actual logic (e.g. dataset parsing, model fine tuning)
according to your own requirements.  Comments and docstrings throughout
the code highlight where your code should go.

## Getting Started

1.  Choose a backbone model and embedding model in `configs/`.  Three
    ready‑made templates are provided:

    - `backbone_llama1b.yaml` – uses
      `meta-llama/Llama-3.2-1B-Instruct` as the small LLM backbone.
    - `backbone_qwen3b.yaml` – uses
      `Qwen/Qwen2.5-3B-Instruct` as a larger but still feasible model.
    - Dataset configs such as `synth_world.yaml`, `trex.yaml`,
      `hotpot.yaml` and `timeqa.yaml` show how to hook in your data
      sources.

2.  Populate `data/raw/` with your source data **or** run the
  automated pipeline:

  ```bash
  python offline/run_pipeline.py --config configs/synth_world.yaml
  ```

  The pipeline generates/refreshes five‑tuples, QA files,
  encoded key/value arrays and the FAISS index under the `store/`
  directory referenced by the config.  Use `--steps` to run a
  subset (e.g. only `encode` + `index`) and `--force` to
  overwrite existing artefacts.

3.  (Optional manual flow.) If you prefer step-by-step control, run
  `offline/encode_kv.py` to encode your five-tuples into key and
  value matrices, then follow up with `offline/build_index.py`
  to create the ANN index.  Adjust the `d_k`/`d_v`/`d_ctx`
  dimensions in the config to suit your hardware.  The encoder
  script now mirrors the training pipeline’s provider flags, so
  `--model_source auto` (default) pulls `BAAI/bge-small-en-v1.5`
  from ModelScope first and only falls back to HuggingFace if
  needed; pass `--model_source huggingface` if you want to skip
  ModelScope entirely.

5.  Train the KBLaM++ model using `train/train_stageA.py`.  Make
  sure your YAML config specifies the correct backbone, embedding
  model, dimension sizes and dataset locations.  Install
  `modelscope` (`pip install modelscope`) so the default
  `--model_source auto` path can fetch
  `LLM-Research/Llama-3.2-1B-Instruct` from ModelScope first and
  silently fall back to HuggingFace if required.  Use
  `--model_source huggingface` only when you deliberately want to
  skip ModelScope.

6.  Evaluate your model on held‑out questions using the utilities in
    `eval/`.

7.  Once comfortable, switch to larger backbones (e.g. 8B models) by
    editing the YAML config; everything else stays the same.

If you see a `TODO:` in the code, it means you need to provide an
implementation specific to your data or models.  For example, the
dataloader currently yields dummy tensors—replace this with logic to
tokenise your questions and feed them to the backbone.

## License

This skeleton is provided as is, without warranties.  It is
expected that you will adapt and modify it for your own research or
applications.