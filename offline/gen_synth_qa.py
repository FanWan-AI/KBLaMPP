"""Generate synthetic question/answer pairs for a synthetic world.

Given a JSON file produced by ``gen_synth_world.py``, this script
should create a list of questions that can be answered by looking
up one or more of the facts in the world.  Each QA pair should
include the indices of the supporting facts and optionally a time
constraint.  The output is written as a JSONL file in
``data/qa``.

As with the world generator, this file contains only a template.  You
should implement the logic to call an LLM and to produce the
appropriate JSON structure (question, answer, type, supporting_facts
and question_time).  See README for guidelines on prompts.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic QA from a world file")
    parser.add_argument("world", type=str, help="Path to world JSONL")
    parser.add_argument("output", type=str, help="Output QA JSONL")
    parser.add_argument("--num_questions", type=int, default=100, help="Number of QA pairs")
    args = parser.parse_args()
    # TODO: load the world (list of five‑tuples) and call LLM to
    # generate questions.  Each QA should be a dict with keys:
    # "qid", "dataset", "question", "answer", "type",
    # "supporting_facts", "question_time".
    qas = []
    out_path = Path(args.output)
    with out_path.open("w") as f:
        for qa in qas:
            json.dump(qa, f)
            f.write("\n")
    print(f"Wrote {len(qas)} questions to {out_path}")


if __name__ == "__main__":
    main()