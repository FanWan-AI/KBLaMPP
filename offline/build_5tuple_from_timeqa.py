"""Parse TimeQA data into five‑tuples.

TimeQA is a dataset of questions requiring temporal reasoning.
Each record typically contains a question, candidate answer entities,
supporting passages and time information.  This script outlines
how to extract the supporting facts as five‑tuples for KBLaM++.

TODO: Implement parsing of your specific TimeQA version.  If the
dataset contains explicit (h,r,t) triples and supporting passages,
convert each into a five‑tuple.  Otherwise you may need to run a
relation extraction model.  For Plan B, consider using a small
hand‑crafted subset.
"""

import argparse
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description="Build five‑tuples from TimeQA")
    parser.add_argument("input", type=str, help="Path to TimeQA data")
    parser.add_argument("output", type=str, help="Output JSONL path for five‑tuples")
    parser.add_argument("--max_records", type=int, default=1000, help="Max number of records")
    args = parser.parse_args()
    with Path(args.input).open() as infile, Path(args.output).open("w") as outfile:
        for i, line in enumerate(infile):
            if i >= args.max_records:
                break
            record = json.loads(line)
            # TODO: Inspect record structure and extract facts.
            # Each fact should have a head id/name/type, relation id/name,
            # tail id/name/type, context, and time_window.
            pass
        print(f"Processed {min(args.max_records, i+1)} records")


if __name__ == "__main__":
    main()