"""Generate synthetic question/answer pairs for a synthetic world."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def load_world(world_path: Path) -> List[dict]:
    facts: List[dict] = []
    with world_path.open(encoding="utf-8") as f:
        for line in f:
            facts.append(json.loads(line))
    return facts


def get_time_window(fact: dict) -> tuple[str, str]:
    tw = fact.get("time_window", {})
    start = tw.get("start")
    end = tw.get("end") or start
    if start is None:
        return "", ""
    return start, end


def build_single_question(fact: dict, fact_idx: int, dataset: str) -> dict:
    start, end = get_time_window(fact)
    head = fact["head"]["name"]
    relation = fact["relation"]["name"]
    tail = fact["tail"]["name"]
    question = f"What is the {relation} of {head}?"
    return {
        "qid": f"{dataset}_{fact_idx:05d}",
        "dataset": dataset,
        "question": question,
        "answer": tail,
        "type": "single-hop",
        "supporting_facts": [fact_idx],
        "question_time": {"start": start, "end": end},
    }


def build_temporal_question(fact: dict, fact_idx: int, dataset: str) -> dict:
    start, end = get_time_window(fact)
    head = fact["head"]["name"]
    relation = fact["relation"]["name"]
    tail = fact["tail"]["name"]
    question = f"Who was the {relation} of {head} in {start[:4]}?"
    return {
        "qid": f"{dataset}_temporal_{fact_idx:05d}",
        "dataset": dataset,
        "question": question,
        "answer": tail,
        "type": "temporal",
        "supporting_facts": [fact_idx],
        "question_time": {"start": start, "end": end},
    }


def build_multi_question(fact_a: dict, fact_b: dict, idx_a: int, idx_b: int, dataset: str, ordinal: int) -> dict:
    start_a, end_a = get_time_window(fact_a)
    start_b, end_b = get_time_window(fact_b)
    head_a = fact_a["head"]["name"]
    tail_a = fact_a["tail"]["name"]
    relation_b = fact_b["relation"]["name"]
    tail_b = fact_b["tail"]["name"]
    question = f"If {head_a} {fact_a['relation']['name']} {tail_a}, who {relation_b} {tail_a}?"
    window_start = start_a or start_b
    window_end = end_b or end_a
    return {
        "qid": f"{dataset}_multi_{ordinal:05d}",
        "dataset": dataset,
        "question": question,
        "answer": tail_b,
        "type": "multi-hop",
        "supporting_facts": [idx_a, idx_b],
        "question_time": {"start": window_start, "end": window_end},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic QA from a synthetic world")
    parser.add_argument("world", type=str, help="Path to world JSONL")
    parser.add_argument("output", type=str, help="Output QA JSONL")
    parser.add_argument("--num_questions", type=int, default=100, help="Number of QA pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic output")
    args = parser.parse_args()

    random.seed(args.seed)
    facts = load_world(Path(args.world))
    if not facts:
        raise SystemExit("Synthetic world is empty. Run gen_synth_world.py first.")

    dataset = Path(args.output).stem
    singles = int(args.num_questions * 0.7)
    multis = int(args.num_questions * 0.2)
    temporals = args.num_questions - singles - multis
    qas: List[dict] = []
    fact_count = len(facts)
    for i in range(singles):
        fi = random.randrange(fact_count)
        qas.append(build_single_question(facts[fi], fi, dataset))
    for i in range(multis):
        fi = random.randrange(fact_count)
        tail_id = facts[fi]["tail"]["id"]
        candidates = [j for j, fact in enumerate(facts) if fact["head"]["id"] == tail_id]
        if not candidates:
            continue
        fj = random.choice(candidates)
        qas.append(build_multi_question(facts[fi], facts[fj], fi, fj, dataset, i))
    for i in range(temporals):
        fi = random.randrange(fact_count)
        qas.append(build_temporal_question(facts[fi], fi, dataset))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for qa in qas:
            json.dump(qa, f, ensure_ascii=False)
            f.write("\n")
    print(f"Generated {len(qas)} QA pairs at {out_path}")


if __name__ == "__main__":
    main()