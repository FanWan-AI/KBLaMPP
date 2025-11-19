"""Generate a synthetic knowledge world using a language model.

This script calls an LLM (e.g. GPT‑4 or Qwen) to generate a small
fictional knowledge base as a list of five‑tuples.  Each run
produces a different world.  The resulting JSONL file can then be
encoded and used for training KBLaM++.

Plan B cannot run LLM calls in this environment, so this file
contains only a template.  You should implement the call to your
favourite LLM API and handle authentication.  See the README for
examples of prompts.
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List


RELATION_TEMPLATES = [
    ("chief executive officer", "ORG", "PERSON"),
    ("founded", "ORG", "DATE"),
    ("headquartered in", "ORG", "LOC"),
    ("partnered with", "ORG", "ORG"),
    ("appointed", "PERSON", "ROLE"),
    ("graduated from", "PERSON", "ORG"),
    ("won award", "PERSON", "AWARD"),
]


def random_date(start_year: int = 2010, end_year: int = 2024) -> datetime:
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return datetime(year, month, day)


def format_time_window(start: datetime, duration_years: int | None) -> dict:
    end_dt = start + timedelta(days=(duration_years or 0) * 365)
    return {
        "start": start.date().isoformat(),
        "end": end_dt.date().isoformat() if duration_years else None,
        "source": "synthetic",
    }


def build_context(head: str, relation: str, tail_str: str, year: int) -> dict:
    sentence = f"In {year}, {head} {relation} {tail_str}."
    return {
        "source": "synthetic",
        "page_title": head,
        "sent_span": sentence,
        "disamb": {"country": "US"},
        "ver": {"url": None, "rev_id": None},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic KB world")
    parser.add_argument("output", type=str, help="Output JSONL path for five‑tuples")
    parser.add_argument("--num_entities", type=int, default=12, help="Number of entities to generate")
    parser.add_argument("--num_facts", type=int, default=64, help="Number of facts to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    random.seed(args.seed)
    entities = []
    for i in range(args.num_entities):
        entity_type = random.choice(["ORG", "PERSON", "LOC"])
        entities.append({
            "id": f"E{i:03d}",
            "name": f"Entity_{i}",
            "type": entity_type,
        })
    facts: List[dict] = []
    for idx in range(args.num_facts):
        relation, head_type, tail_type = random.choice(RELATION_TEMPLATES)
        head = random.choice(entities)
        tail_choice = random.choice(entities)
        if tail_type == "DATE":
            tail = {"id": f"V{idx:04d}", "name": str(random_date().date()), "type": "DATE"}
        elif tail_type == "ROLE":
            tail = {"id": f"V{idx:04d}", "name": random.choice(["CTO", "CFO", "Advisor"]), "type": "ROLE"}
        elif tail_type == "AWARD":
            tail = {"id": f"V{idx:04d}", "name": random.choice(["Innovation Prize", "Best Startup"]), "type": "AWARD"}
        else:
            tail = tail_choice
        start_date = random_date()
        duration = random.choice([None, 1, 2, 3])
        fact = {
            "head": head,
            "relation": {"id": f"R{idx:03d}", "name": relation},
            "tail": tail,
            "context": build_context(head["name"], relation, tail["name"], start_date.year),
            "time_window": format_time_window(start_date, duration),
        }
        facts.append(fact)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for fact in facts:
            json.dump(fact, f, ensure_ascii=False)
            f.write("\n")
    print(f"Generated {len(facts)} synthetic facts at {out_path}")


if __name__ == "__main__":
    main()