"""Evaluate KBLaM++ on question answering datasets.

This script reads a trained KBLaM++ model checkpoint (not included
in Plan B) and a set of question/answer pairs and computes
standard metrics such as exact match (EM) and F1.  Because the
models in Plan B are only stubs, this script currently prints
placeholder results.

TODO: Implement real evaluation.  You will need to tokenise
questions, run the model’s forward pass, extract the predicted
answer and compare it to the ground truth.  You may also want to
evaluate whether the model’s evidence selection matches the
supporting facts.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate KBLaM++ QA performance")
    parser.add_argument("--model", type=str, help="Path to trained model (not used in Plan B)")
    parser.add_argument("--data", type=str, help="QA JSONL file to evaluate")
    args = parser.parse_args()
    correct = 0
    total = 0
    with Path(args.data).open() as f:
        for line in f:
            qa = json.loads(line)
            # TODO: run model, get predicted answer
            pred = qa["answer"]  # Stub: pretend we predict perfectly
            if pred.strip().lower() == qa["answer"].strip().lower():
                correct += 1
            total += 1
    print(f"Accuracy (stub) = {correct / max(total,1):.4f}")


if __name__ == "__main__":
    main()