"""End-to-end offline pipeline for KBLaM++ Plan B datasets.

This script orchestrates the synthetic world generation, QA synthesis,
key/value encoding and FAISS index construction steps using the
configuration files under ``configs/``.  It ensures Stage A training has
all required artefacts in place without manual shell bookkeeping.

Example usage::

    python offline/run_pipeline.py --config configs/synth_world.yaml

By default every step runs (world generation, QA train/dev generation,
encoding and index).  Existing artefacts are skipped unless ``--force``
is supplied.  Individual steps can be selected via ``--steps``.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "python"


def run_cmd(cmd: List[str]) -> None:
    print(f"[pipeline] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KBLaM++ offline pipeline")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument(
        "--steps",
        nargs="*",
        choices=["world", "qa_train", "qa_dev", "encode", "index"],
        default=["world", "qa_train", "qa_dev", "encode", "index"],
        help="Subset of steps to execute in order",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing artefacts")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    store_cfg = cfg.get("store", {})
    pipeline_cfg = cfg.get("pipeline", {})

    five_tuple_path = ROOT / data_cfg["five_tuple_path"]
    qa_train_path = ROOT / data_cfg["qa_train_path"]
    qa_dev_path = ROOT / data_cfg["qa_dev_path"]
    store_root = ROOT / store_cfg.get("root", "store")
    index_subdir = store_cfg.get("index_subdir", "index_hnsw")

    # --- Step: world generation ---
    if "world" in args.steps:
        if args.force or not five_tuple_path.exists():
            ensure_parent(five_tuple_path)
            world_cfg = pipeline_cfg.get("world", {})
            cmd = [
                PYTHON,
                str(ROOT / "offline" / "gen_synth_world.py"),
                str(five_tuple_path),
                "--num_entities",
                str(world_cfg.get("num_entities", 12)),
                "--num_facts",
                str(world_cfg.get("num_facts", 64)),
                "--seed",
                str(world_cfg.get("seed", 42)),
            ]
            run_cmd(cmd)
        else:
            print(f"[pipeline] Skipping world generation (exists: {five_tuple_path})")

    # --- Step: QA generation ---
    def run_qa(output: Path, num_questions: int, seed: int) -> None:
        ensure_parent(output)
        cmd = [
            PYTHON,
            str(ROOT / "offline" / "gen_synth_qa.py"),
            str(five_tuple_path),
            str(output),
            "--num_questions",
            str(num_questions),
            "--seed",
            str(seed),
        ]
        run_cmd(cmd)

    qa_cfg = pipeline_cfg.get("qa", {})
    if "qa_train" in args.steps:
        if args.force or not qa_train_path.exists():
            run_qa(
                qa_train_path,
                qa_cfg.get("train_questions", 200),
                qa_cfg.get("seed", 1337),
            )
        else:
            print(f"[pipeline] Skipping QA train generation (exists: {qa_train_path})")

    if "qa_dev" in args.steps:
        if args.force or not qa_dev_path.exists():
            run_qa(
                qa_dev_path,
                qa_cfg.get("dev_questions", 50),
                qa_cfg.get("seed", 4242),
            )
        else:
            print(f"[pipeline] Skipping QA dev generation (exists: {qa_dev_path})")

    # --- Step: encode keys/values ---
    if "encode" in args.steps:
        if args.force or not (store_root / "K.npy").exists():
            ensure_parent(store_root / "K.npy")
            enc_cfg = pipeline_cfg.get("encoder", {})
            cmd = [
                PYTHON,
                str(ROOT / "offline" / "encode_kv.py"),
                str(five_tuple_path),
                str(store_root),
                "--d_k",
                str(enc_cfg.get("d_k", 384)),
                "--d_v",
                str(enc_cfg.get("d_v", 384)),
                "--d_ctx",
                str(enc_cfg.get("d_ctx", 384)),
                "--d_tau",
                str(enc_cfg.get("d_tau", 32)),
                "--model",
                enc_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
                "--batch_size",
                str(enc_cfg.get("batch_size", 64)),
                "--max_length",
                str(enc_cfg.get("max_length", 128)),
            ]
            run_cmd(cmd)
        else:
            print(f"[pipeline] Skipping encode (found existing tensors in {store_root})")

    # --- Step: build FAISS index ---
    if "index" in args.steps:
        index_dir = store_root / index_subdir
        if args.force or not index_dir.exists():
            ensure_parent(index_dir / "dummy")
            idx_cfg = pipeline_cfg.get("index", {})
            cmd = [
                PYTHON,
                str(ROOT / "offline" / "build_index.py"),
                "--store_dir",
                str(store_root),
                "--method",
                idx_cfg.get("method", "hnsw"),
            ]
            run_cmd(cmd)
        else:
            print(f"[pipeline] Skipping index build (exists: {index_dir})")

    print("[pipeline] Completed requested steps")


if __name__ == "__main__":
    main()
