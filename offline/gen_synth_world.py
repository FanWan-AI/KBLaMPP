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
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic KB world")
    parser.add_argument("output", type=str, help="Output JSONL path for five‑tuples")
    parser.add_argument("--num_entities", type=int, default=12, help="Number of entities to generate")
    parser.add_argument("--num_facts", type=int, default=64, help="Number of facts to generate")
    args = parser.parse_args()
    out_path = Path(args.output)
    # TODO: call your LLM here with a properly designed prompt.  For
    # example, use openai.ChatCompletion.create() and the prompt
    # described in the README.  The model should return a JSON array
    # of five‑tuples.  This stub simply writes an empty array.
    facts = []
    with out_path.open("w") as f:
        for fact in facts:
            json.dump(fact, f)
            f.write("\n")
    print(f"Wrote {len(facts)} facts to {out_path}")


if __name__ == "__main__":
    main()