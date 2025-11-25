"""Stage A training loop for KBLaM++.

This script composes the backbone model, knowledge modules and
tokenized QA data for stage A, where the backbone is frozen and
only the KB selector/fusion components are optimised.

Dataset paths, store directories and offline artefacts are all
read from the dataset config (e.g. ``configs/synth_world.yaml``),
so running ``offline/run_pipeline.py`` beforehand automatically
prepares everything required for training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Tuple, cast

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Optional dependency for mainland-friendly downloads
    from modelscope import snapshot_download as ms_snapshot_download  # type: ignore
except ImportError:  # pragma: no cover
    ms_snapshot_download = None

from kblampp.knowledge_index import KnowledgeIndex
from kblampp.kb_store import KBValueStore
from kblampp.selector import KBSelector
from kblampp.fusion import KBFusionLayer
from kblampp.injection_wrapper import KBInjectedModel
from .dataloader import get_dataloader

ROOT = Path(__file__).resolve().parents[1]


MODEL_SCOPE_CACHE = Path(os.environ.get("MODEL_SCOPE_CACHE_DIR", Path.home() / ".cache" / "modelscope" / "snapshots"))


def as_float(value: Any, default: float) -> float:
    """Coerce YAML-loaded numbers (including scientific-notation strings) to float."""

    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - config error path
            raise ValueError(f"Expected numeric value but got '{value}'") from exc
    raise TypeError(f"Unsupported type for numeric field: {type(value)!r}")


def load_backbone_with_fallback(
    model_name: str,
    device: torch.device,
    *,
    source: str = "auto",
    revision: str | None = None,
) -> Tuple[Any, Any, str]:
    """Load backbone preferring ModelScope, then HuggingFace as a fallback."""

    hf_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision:
        hf_kwargs["revision"] = revision

    last_error: Exception | None = None
    if source in {"auto", "modelscope"}:
        if ms_snapshot_download is None:
            msg = "ModelScope is not installed (pip install modelscope)"
            if source == "modelscope":
                raise RuntimeError(msg) from None
            print(f"[train_stageA] {msg}; falling back to HuggingFace", flush=True)
        else:
            try:
                MODEL_SCOPE_CACHE.mkdir(parents=True, exist_ok=True)
                cache_dir = ms_snapshot_download(
                    model_name,
                    revision=revision or "master",
                    cache_dir=str(MODEL_SCOPE_CACHE),
                )
                model = cast(torch.nn.Module, AutoModelForCausalLM.from_pretrained(cache_dir, **hf_kwargs))
                model.to(device)
                model = cast(AutoModelForCausalLM, model)
                tokenizer = AutoTokenizer.from_pretrained(cache_dir, **hf_kwargs)
                return model, tokenizer, "modelscope"
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if source == "modelscope":
                    raise
                print(
                    f"[train_stageA] ModelScope load failed for '{model_name}' ({exc}); trying HuggingFace",
                    flush=True,
                )

    if source in {"auto", "huggingface"}:
        try:
            model = cast(torch.nn.Module, AutoModelForCausalLM.from_pretrained(model_name, **hf_kwargs))
            model.to(device)
            model = cast(AutoModelForCausalLM, model)
            tokenizer = AutoTokenizer.from_pretrained(model_name, **hf_kwargs)
            return model, tokenizer, "huggingface"
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if source == "huggingface":
                raise
            raise RuntimeError(
                f"Failed to load '{model_name}' from both ModelScope and HuggingFace"
            ) from exc

    raise last_error or RuntimeError(f"Unknown model source '{source}'")


def main():
    parser = argparse.ArgumentParser(description="Train KBLaM++ Stage A")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (default: auto)")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional override for optimiser updates")
    parser.add_argument(
        "--model_source",
        choices=["auto", "huggingface", "modelscope"],
        default="auto",
        help="Where to pull backbone weights from",
    )
    parser.add_argument("--model_revision", type=str, default=None, help="Optional revision/tag for the backbone")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Load backbone config
    with open(cfg["backbone_config"]) as f:
        bcfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    store_cfg = cfg.get("store", {})
    dataset_path = ROOT / data_cfg.get("qa_train_path", "")
    if not dataset_path.exists():
        raise FileNotFoundError(f"QA dataset not found: {dataset_path}")

    store_root = ROOT / store_cfg.get("root", "store")
    index_dir = store_root / store_cfg.get("index_subdir", "index_hnsw")
    required_files = [
        store_root / "K.npy",
        store_root / "V.npy",
        store_root / "meta" / "ctx_vec.npy",
        store_root / "meta" / "tau_min.npy",
        store_root / "meta" / "tau_max.npy",
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing encoded artefact: {path}")
    if not index_dir.exists():
        raise FileNotFoundError(f"FAISS index directory missing: {index_dir}")

    # Load backbone model and tokenizer
    model_name = bcfg["backbone"]
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, tokenizer, provider = load_backbone_with_fallback(
        model_name,
        device,
        source=args.model_source,
        revision=args.model_revision,
    )
    print(f"[train_stageA] Loaded backbone via {provider}")
    # Stage A freezes every backbone parameter so that we only train the KB
    # modules.  This keeps memory usage predictable on small GPUs and matches
    # the Plan B spec in the docs.
    backbone.requires_grad_(False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Knowledge components
    # The index + store encapsulate all offline artefacts, keeping the training
    # loop agnostic to how the KB was produced.
    os.environ.setdefault("KBLAMPP_FORCE_BRUTE_INDEX", "1")
    index = KnowledgeIndex.load(str(index_dir))
    kb_store = KBValueStore(str(store_root), device)
    d_k, d_v, d_ctx = bcfg["d_k"], bcfg["d_v"], bcfg["d_ctx"]
    selector = KBSelector(d_k, d_ctx, gamma=bcfg["gamma"], eta=bcfg["eta"], temperature=bcfg["temperature"])
    fusion = KBFusionLayer(bcfg["d_model"], n_heads=8)

    # Wrap model
    inj_model = KBInjectedModel(
        backbone=backbone,
        inject_layer=bcfg["inject_layers"][0],
        selector=selector,
        fusion=fusion,
        index=index,
        kb_store=kb_store,
        d_k=d_k,
        d_v=d_v,
        d_ctx=d_ctx,
        k_top=bcfg["K_top"],
    ).to(device)

    train_cfg = bcfg.get("train", {})
    lr = as_float(train_cfg.get("lr"), 5e-4)
    weight_decay = as_float(train_cfg.get("weight_decay"), 0.01)
    optimiser = AdamW(
        inj_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    dataloader = get_dataloader(
        str(dataset_path),
        tokenizer,
        train_cfg.get("max_seq_len", 512),
        train_cfg.get("batch_size", 1),
    )

    grad_accum = max(1, train_cfg.get("grad_accum", 1))
    max_steps = args.max_steps or train_cfg.get("max_steps", 1000)
    log_interval = train_cfg.get("log_interval", 50)

    inj_model.train()
    optimiser.zero_grad()
    update_step = 0
    for step, batch in enumerate(dataloader, start=1):
        if update_step >= max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        question_time = batch["question_time"].to(device)
        logits = inj_model(input_ids, attention_mask=attention_mask, question_time=question_time)
        # Standard causal LM loss; dividing by grad_accum lets us call backward
        # every step without scaling issues when we only step the optimiser
        # every ``grad_accum`` mini-batches.
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        ) / grad_accum
        loss.backward()
        if step % grad_accum == 0:
            optimiser.step()
            optimiser.zero_grad()
            update_step += 1
            if update_step % log_interval == 0:
                print(f"Update {update_step}: loss = {loss.item() * grad_accum:.4f}")
    print(f"Training complete (Stage A) – updates run: {update_step}")


if __name__ == "__main__":
    main()