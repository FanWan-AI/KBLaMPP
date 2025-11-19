"""Stage A training loop for KBLaM++.

This script demonstrates how to put together the backbone model,
knowledge injection modules and data loader for end‑to‑end training
of KBLaM++.  In stage A you freeze the backbone weights and
optimise only the knowledge modules.  Once the pipeline is
functional, you can expand it to stage B (LoRA fine tuning).

Note: This script uses dummy tokens and does not produce
meaningful results out of the box.  Replace the data loading,
forward pass and loss computation with your real logic.
"""

from __future__ import annotations

import argparse
import yaml
import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from kblampp.knowledge_index import KnowledgeIndex
from kblampp.kb_store import KBValueStore
from kblampp.scorers import ContextScorer, TimeScorer
from kblampp.selector import KBSelector
from kblampp.fusion import KBFusionLayer
from kblampp.injection_wrapper import KBInjectedModel
from .dataloader import get_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train KBLaM++ Stage A")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, required=True, help="QA dataset for training")
    parser.add_argument("--store", type=str, default="store", help="Path to store directory")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Load backbone config
    with open(cfg["backbone_config"]) as f:
        bcfg = yaml.safe_load(f)
    # Load backbone model
    model_name = bcfg["backbone"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    backbone.requires_grad_(False)

    # Knowledge components
    index = KnowledgeIndex.load(f"{args.store}/index_hnsw")
    kb_store = KBValueStore(args.store, device)
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
    ).to(device)

    optimiser = AdamW(inj_model.parameters(), lr=bcfg["train"]["lr"], weight_decay=bcfg["train"]["weight_decay"])
    dataloader = get_dataloader(args.dataset, bcfg["train"]["batch_size"])

    inj_model.train()
    for step, batch in enumerate(dataloader):
        if step >= bcfg["train"]["max_steps"]:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        # TODO: question_time should come from dataset
        question_time = batch["question_time"].to(device)
        optimiser.zero_grad()
        logits = inj_model(input_ids, attention_mask=None)
        # TODO: compute loss.  This stub just uses MSE between logits and labels
        # Replace with cross entropy on the answer positions
        loss = ((logits - torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()) ** 2).mean()
        loss.backward()
        optimiser.step()
        if step % bcfg["train"]["log_interval"] == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
    print("Training complete (Stage A stub)")


if __name__ == "__main__":
    main()