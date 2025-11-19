"""Extract five‑tuples from HotpotQA supporting facts.

HotpotQA provides multi‑hop questions with supporting facts.  This
script demonstrates how to turn those supporting sentences into
KBLaM++ five‑tuples.  For Plan B you should sample a handful of
questions and parse only the facts needed to answer them.

TODO: Implement the actual parsing of HotpotQA.  Load the JSON
dataset and iterate through each question, extracting the
supporting facts and mapping them to (h,r,t) with a context sentence.
"""

import json
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Build five‑tuples from HotpotQA")
    parser.add_argument("input", type=str, help="Path to HotpotQA JSON file")
    parser.add_argument("output", type=str, help="Output JSONL path for five‑tuples")
    parser.add_argument("--max_questions", type=int, default=1000, help="Number of QA pairs to process")
    args = parser.parse_args()
    count = 0
    with Path(args.input).open() as infile, Path(args.output).open("w") as outfile:
        data = json.load(infile)
        for item in data:
            if count >= args.max_questions:
                break
            # TODO: Use item['supporting_facts'] to pull the supporting
            # sentences and map them to (h,r,t) five‑tuples.  Each
            # supporting_fact is a tuple (title, sentence_index).
            # You will need to load the corresponding Wikipedia
            # paragraphs from item['context'] to extract the text.
            # See the official HotpotQA docs for more details.
            count += 1
        print(f"Processed {count} questions")


if __name__ == "__main__":
    main()