"""Verify five‑tuple JSONL file.

Use this script to ensure that your five‑tuple file conforms to the
expected schema before encoding it.  It performs basic checks on
required keys, data types and time window formats.  In Plan B the
checks are minimal but you can expand them as you handle more
complex data.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sanity check five‑tuple file")
    parser.add_argument("input", type=str, help="Path to five‑tuple JSONL")
    args = parser.parse_args()
    path = Path(args.input)
    ok = True
    with path.open() as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            for key in ["head", "relation", "tail", "context", "time_window"]:
                if key not in rec:
                    ok = False
                    print(f"Line {i}: missing key {key}")
            # Check context fields
            if "sent_span" not in rec["context"]:
                ok = False
                print(f"Line {i}: missing context.sent_span")
    print("OK" if ok else "Errors found")


if __name__ == "__main__":
    main()