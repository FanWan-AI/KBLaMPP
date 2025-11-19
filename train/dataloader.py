"""Data loading utilities for KBLaM++.

This module defines a simple dataset and dataloader for
question/answer pairs.  Each sample should contain a tokenised
``input_ids`` tensor, a ``labels`` tensor (for the answer) and
optionally ``question_time`` and ``supporting_facts``.  The latter
can be used to guide the selection weights during training.

In this Plan B skeleton we return dummy tensors.  Replace the
``__getitem__`` implementation with real logic: tokenise
``question``, convert answer to labels and provide time windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader


class QADataset(Dataset):
    """A Dataset for KBLaM++ question answering tasks."""

    def __init__(self, qa_path: str) -> None:
        self.records: List[Dict[str, object]] = []
        with open(qa_path) as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        # TODO: tokenise question, create input_ids and labels
        # Here we return fixed dummy tensors for demonstration only
        input_ids = torch.randint(0, 100, (32,), dtype=torch.long)
        labels = torch.randint(0, 100, (32,), dtype=torch.long)
        question_time = torch.zeros(2)  # [start, end]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "question_time": question_time,
        }


def get_dataloader(path: str, batch_size: int) -> DataLoader:
    ds = QADataset(path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)