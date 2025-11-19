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
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class QADataset(Dataset):
    """A Dataset for KBLaM++ question answering tasks."""

    def __init__(self, qa_path: str, tokenizer, max_seq_len: int = 512, prompt_template: str = "Question: {question}\nAnswer: ") -> None:
        self.records: List[Dict[str, object]] = []
        with open(qa_path) as f:
            for line in f:
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or tokenizer.bos_token_id or 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        question = rec.get("question", "")
        answer = rec.get("answer", "")
        q_prompt = self.prompt_template.format(question=question)

        question_enc = self.tokenizer(
            q_prompt,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=True,
            return_tensors="pt",
        )
        answer_suffix = answer + (self.tokenizer.eos_token or "")
        answer_enc = self.tokenizer(
            answer_suffix,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
            return_tensors="pt",
        )

        question_ids = question_enc["input_ids"].squeeze(0).tolist()
        answer_ids = answer_enc["input_ids"].squeeze(0).tolist()
        input_ids = question_ids + answer_ids
        labels = [-100] * len(question_ids) + answer_ids

        if len(input_ids) > self.max_seq_len:
            overflow = len(input_ids) - self.max_seq_len
            input_ids = input_ids[:-overflow]
            labels = labels[:-overflow]

        attention_mask = [1] * len(input_ids)
        question_time = rec.get("question_time", {})
        q_start = self._parse_time(question_time.get("start"))
        q_end = self._parse_time(question_time.get("end"))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "question_time": torch.tensor([q_start, q_end], dtype=torch.float32),
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_len = max(item["input_ids"].size(0) for item in batch)
        input_ids = torch.stack([
            F.pad(item["input_ids"], (0, pad_len - item["input_ids"].size(0)), value=self.pad_token_id)
            for item in batch
        ])
        labels = torch.stack([
            F.pad(item["labels"], (0, pad_len - item["labels"].size(0)), value=-100)
            for item in batch
        ])
        attention_mask = torch.stack([
            F.pad(item["attention_mask"], (0, pad_len - item["attention_mask"].size(0)), value=0)
            for item in batch
        ])
        question_time = torch.stack([item["question_time"] for item in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "question_time": question_time,
        }

    @staticmethod
    def _parse_time(value: str | None) -> float:
        if value is None:
            return 0.0
        try:
            dt = datetime.fromisoformat(value)
            epoch = datetime(1970, 1, 1)
            return (dt - epoch).total_seconds()
        except ValueError:
            return 0.0


def get_dataloader(path: str, tokenizer, max_seq_len: int, batch_size: int) -> DataLoader:
    ds = QADataset(path, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)