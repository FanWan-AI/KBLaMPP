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
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Tuple, TextIO, cast

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


MODEL_SCOPE_CACHE = Path(
    os.environ.get("MODEL_SCOPE_CACHE_DIR", Path.home() / ".cache" / "modelscope" / "snapshots")
)

LOGGER = logging.getLogger(__name__)

_WARNING_FILTERS: tuple[tuple[type[Warning], str], ...] = (
    (FutureWarning, r".*past_key_value.*"),
    (UserWarning, r".*Torch was not compiled with flash attention.*"),
)


def apply_warning_filters(enabled: bool = True) -> None:
    """Suppress noisy framework warnings unless explicitly disabled."""

    if not enabled:
        return
    for category, pattern in _WARNING_FILTERS:
        warnings.filterwarnings("ignore", message=pattern, category=category)


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Set up root logging with optional file duplication."""

    lvl = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(lvl)
    for handler in handlers:
        root_logger.addHandler(handler)


def format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds == float("inf"):
        return "?"
    minutes, sec = divmod(max(0.0, seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{int(sec):02d}"
    return f"{int(minutes):02d}:{int(sec):02d}"


def emit_progress(metrics: dict[str, Any], jsonl_handle: TextIO | None = None) -> None:
    eta_str = format_seconds(metrics.get("eta_seconds"))
    max_updates = metrics.get("max_updates")
    header = f"Update {metrics['update_step']}/" + (str(max_updates) if max_updates else "?")
    epoch = metrics.get("epoch")
    parts = [
        header,
        f"Epoch {epoch}" if epoch is not None else None,
        f"Loss {metrics['loss']:.4f}",
    ]
    ema_loss = metrics.get("ema_loss")
    if ema_loss is not None:
        parts.append(f"EMA {ema_loss:.4f}")
    parts.append(f"LR {metrics['lr']:.2e}")
    parts.append(f"{metrics['tokens_per_sec']:.1f} tok/s")
    parts.append(f"{metrics['samples_per_sec']:.2f} samples/s")
    parts.append(f"ETA {eta_str}")
    gpu_mem = metrics.get("gpu_mem_gb")
    if gpu_mem is not None:
        parts.append(f"GPU {gpu_mem:.2f} GB")
    LOGGER.info(" | ".join(part for part in parts if part))

    if jsonl_handle:
        serialisable = metrics.copy()
        serialisable["eta"] = eta_str
        serialisable["timestamp"] = time.time()
        jsonl_handle.write(json.dumps(serialisable) + "\n")
        jsonl_handle.flush()

def _prepare_modelscope_snapshot(cache_dir: Path) -> None:
    """Patch ModelScope snapshots so HuggingFace loaders can parse them."""

    config_path = cache_dir / "config.json"
    alt_config = cache_dir / "configuration.json"
    if not config_path.exists() and alt_config.exists():
        config_path = alt_config

    if not config_path.exists():
        return
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return

    if not isinstance(cfg, dict):
        return

    if "model_type" not in cfg:
        model_type = None
        for arch in cfg.get("architectures", []):
            if not isinstance(arch, str):
                continue
            name = arch.lower()
            if "llama" in name:
                model_type = "llama"
                break
            if "qwen" in name:
                model_type = "qwen"
                break
        if model_type is None:
            model_type = "llama"
        cfg["model_type"] = model_type
    # Transformers < 4.58 applies a local mistral regex workaround when vocab_size>100k AND
    # transformers_version <= 4.57.2.  Setting this field to a newer version bypasses the buggy
    # branch that expects ``config`` to be an object instead of a raw dict.
    cfg["transformers_version"] = "4.58.0"
    target_path = cache_dir / "config.json"
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


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
    hf_token: str | None = None,
) -> Tuple[Any, Any, str]:
    """Load backbone preferring ModelScope, then HuggingFace as a fallback."""

    hf_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if revision:
        hf_kwargs["revision"] = revision
    if hf_token:
        hf_kwargs["token"] = hf_token

    last_error: Exception | None = None
    if source in {"auto", "modelscope"}:
        if ms_snapshot_download is None:
            msg = "ModelScope is not installed (pip install modelscope)"
            if source == "modelscope":
                raise RuntimeError(msg) from None
            LOGGER.warning("%s; falling back to HuggingFace", msg)
        else:
            try:
                MODEL_SCOPE_CACHE.mkdir(parents=True, exist_ok=True)
                download_kwargs = {
                    "revision": revision or "master",
                    "cache_dir": str(MODEL_SCOPE_CACHE),
                }
                try:
                    cache_dir = Path(ms_snapshot_download(model_name, use_symlinks=False, **download_kwargs))
                except TypeError:
                    cache_dir = Path(ms_snapshot_download(model_name, **download_kwargs))
                _prepare_modelscope_snapshot(cache_dir)
                config = AutoConfig.from_pretrained(str(cache_dir), **hf_kwargs)
                model = cast(
                    torch.nn.Module,
                    AutoModelForCausalLM.from_pretrained(str(cache_dir), config=config, **hf_kwargs),
                )
                model.to(device)
                model = cast(AutoModelForCausalLM, model)
                tokenizer = AutoTokenizer.from_pretrained(
                    str(cache_dir), config=config, fix_mistral_regex=True, **hf_kwargs
                )
                return model, tokenizer, "modelscope"
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if source == "modelscope":
                    raise
                LOGGER.warning(
                    "ModelScope load failed for '%s' (%s); trying HuggingFace",
                    model_name,
                    exc,
                )

    if source in {"auto", "huggingface"}:
        try:
            model = cast(torch.nn.Module, AutoModelForCausalLM.from_pretrained(model_name, **hf_kwargs))
            model.to(device)
            model = cast(AutoModelForCausalLM, model)
            tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True, **hf_kwargs)
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
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace access token for gated/private backbones",
    )
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Allow FAISS ANN (sets KBLAMPP_FORCE_BRUTE_INDEX=0)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=None,
        help="Override optimiser update interval for progress logs",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default=os.environ.get("KBLAMPP_LOG_LEVEL", "INFO"),
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to tee logs to a file",
    )
    parser.add_argument(
        "--log_jsonl",
        type=str,
        default=None,
        help="Write structured training metrics to this JSONL file",
    )
    parser.add_argument(
        "--loss_ema_beta",
        type=float,
        default=None,
        help="Override EMA smoothing factor used for displayed loss (default 0.9)",
    )
    parser.add_argument(
        "--show_warnings",
        dest="suppress_warnings",
        action="store_false",
        help="Disable warning suppression for debugging",
    )
    parser.set_defaults(suppress_warnings=True)
    args = parser.parse_args()

    configure_logging(args.log_level, args.log_file)
    apply_warning_filters(args.suppress_warnings)

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
        hf_token=args.hf_token,
    )
    LOGGER.info("Loaded backbone via %s", provider)
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
    if args.use_faiss:
        os.environ["KBLAMPP_FORCE_BRUTE_INDEX"] = "0"
    else:
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

    dataset_size = len(dataloader.dataset)  # type: ignore[arg-type]
    if dataset_size == 0:
        raise ValueError(
            f"QA dataset '{dataset_path}' is empty; run offline/run_pipeline.py or provide training data"
        )

    jsonl_handle: TextIO | None = None
    if args.log_jsonl:
        jsonl_path = Path(args.log_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("a", encoding="utf-8")

    grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
    max_steps = int(args.max_steps or train_cfg.get("max_steps", 1000))
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    log_interval_cfg = train_cfg.get("log_interval", 50)
    log_interval = int(args.log_interval or log_interval_cfg or 50)
    log_interval = max(1, log_interval)

    ema_beta = float(train_cfg.get("loss_ema_beta", 0.9))
    if args.loss_ema_beta is not None:
        ema_beta = float(args.loss_ema_beta)
    ema_beta = min(max(ema_beta, 0.0), 0.9999)

    batch_size = int(train_cfg.get("batch_size", 1))
    LOGGER.info(
        "Stage A config: batch_size=%d, grad_accum=%d, max_updates=%d, log_interval=%d, dataset_size=%d",
        batch_size,
        grad_accum,
        max_steps,
        log_interval,
        dataset_size,
    )

    inj_model.train()
    optimiser.zero_grad()
    update_step = 0
    epoch = 1
    total_tokens = 0
    total_samples = 0
    ema_loss: float | None = None
    start_time = time.time()

    data_iter = iter(dataloader)
    batches_processed = 0

    try:
        while update_step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(dataloader)
                LOGGER.info("Epoch %d completed (%d samples); reshuffling and continuing", epoch - 1, dataset_size)
                continue

            batches_processed += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            question_time = batch["question_time"].to(device)

            total_tokens += int(attention_mask.sum().item())
            total_samples += int(input_ids.size(0))

            logits = inj_model(input_ids, attention_mask=attention_mask, question_time=question_time)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            ) / grad_accum
            loss.backward()
            if batches_processed % grad_accum == 0:
                optimiser.step()
                optimiser.zero_grad()
                update_step += 1

                current_loss = loss.item() * grad_accum
                ema_loss = current_loss if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * current_loss

                should_log = (
                    update_step % log_interval == 0
                    or update_step == 1
                    or update_step == max_steps
                )
                if should_log:
                    elapsed = max(time.time() - start_time, 1e-6)
                    updates_per_sec = update_step / elapsed
                    tokens_per_sec = total_tokens / elapsed
                    samples_per_sec = total_samples / elapsed
                    eta_seconds = None
                    if updates_per_sec > 0 and max_steps:
                        eta_seconds = max((max_steps - update_step) / updates_per_sec, 0.0)
                    gpu_mem = None
                    if torch.cuda.is_available() and device.type == "cuda":  # type: ignore[attr-defined]
                        try:
                            device_index = device.index if device.index is not None else torch.cuda.current_device()
                            gpu_mem = torch.cuda.max_memory_reserved(device_index) / (1024 ** 3)
                        except Exception:  # pragma: no cover - best effort metric
                            gpu_mem = None
                    emit_progress(
                        {
                            "update_step": update_step,
                            "max_updates": max_steps,
                            "loss": current_loss,
                            "ema_loss": ema_loss,
                            "lr": optimiser.param_groups[0]["lr"],
                            "tokens_per_sec": tokens_per_sec,
                            "samples_per_sec": samples_per_sec,
                            "tokens_seen": total_tokens,
                            "samples_seen": total_samples,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": eta_seconds,
                            "gpu_mem_gb": gpu_mem,
                            "epoch": epoch,
                        },
                        jsonl_handle,
                    )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    LOGGER.info("Training complete (Stage A) – updates run: %d", update_step)


if __name__ == "__main__":
    main()