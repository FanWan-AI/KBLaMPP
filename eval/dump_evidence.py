"""Dump evidence chains for manual inspection.

Given a KBLaM++ model and a set of questions, this script will
run the model and output the top‑k knowledge facts (with weights)
that were used for each token when answering the question.  It
produces a JSON for each question containing the question, the
answer, the prediction and the top‑k facts with their ``alpha``
weights.  This helps you understand whether the model is
selecting relevant evidence.

In Plan B we do not implement the full logic.  The script
illustrates the intended output format using stub predictions and
randomly selected facts.  You should replace the stub logic with a
call to your trained model and ANN index.
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Dump KBLaM++ evidence chains (stub)")
    parser.add_argument("--model", type=str, help="Path to model (unused in stub)")
    parser.add_argument("--data", type=str, help="QA file to process")
    parser.add_argument("--facts", type=str, help="Five‑tuple file for KB")
    parser.add_argument("--out", type=str, help="Output JSONL file")
    parser.add_argument("--k", type=int, default=3, help="Number of top facts to show")
    args = parser.parse_args()
    # Load facts into memory
    facts = []
    with open(args.facts) as f:
        for i, line in enumerate(f):
            facts.append(json.loads(line))
    # Process questions
    with Path(args.data).open() as qf, Path(args.out).open("w") as outf:
        for line in qf:
            qa = json.loads(line)
            # Stub: pretend model predicts correctly and selects random facts
            top_ids = random.sample(range(len(facts)), args.k)
            top_facts = []
            for idx in top_ids:
                fact = facts[idx]
                top_facts.append({
                    "index": idx,
                    "head": fact["head"]["name"],
                    "relation": fact["relation"]["name"],
                    "tail": fact["tail"]["name"],
                    "context": fact["context"]["sent_span"],
                    "time_window": [fact["time_window"].get("start"), fact["time_window"].get("end")],
                    "alpha": round(random.random(), 3),
                })
            out_obj = {
                "qid": qa.get("qid", ""),
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "pred_answer": qa.get("answer"),  # stub matches answer
                "top_k_facts": top_facts,
            }
            json.dump(out_obj, outf)
            outf.write("\n")
    print(f"Wrote evidence for {len(open(args.data).readlines())} questions to {args.out}")


if __name__ == "__main__":
    main()