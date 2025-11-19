"""Visualise top‑k attention weights from evidence chains.

This script reads the JSON output of ``dump_evidence.py`` and
produces a simple text‑based bar chart of the alpha weights for
each question.  In a real project you might use matplotlib or
seaborn, but for Plan B a textual representation is sufficient.
"""

import argparse
import json
from pathlib import Path


def bar(weight: float, width: int = 50) -> str:
    """Return a simple ASCII bar for a given weight."""
    filled = int(weight * width)
    return "#" * filled + "-" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="Visualise alpha weights")
    parser.add_argument("--input", type=str, help="Evidence JSONL file")
    parser.add_argument("--top_n", type=int, default=5, help="Number of examples to show")
    args = parser.parse_args()
    with Path(args.input).open() as f:
        for i, line in enumerate(f):
            if i >= args.top_n:
                break
            obj = json.loads(line)
            print(f"Q{i}: {obj['question']}")
            for fact in obj["top_k_facts"]:
                print(f"  {bar(fact['alpha'])}  {fact['alpha']:.2f} :: {fact['head']} --{fact['relation']}--> {fact['tail']}")
            print()


if __name__ == "__main__":
    main()