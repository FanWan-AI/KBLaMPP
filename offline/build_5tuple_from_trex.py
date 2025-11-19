"""Convert a T‑REx subset into KBLaM++ five‑tuples.

This script reads a subset of the T‑REx dataset (or any similar
triples file) and extracts (h,r,t) along with an aligned evidence
sentence to produce a JSONL file of five‑tuples.  For Plan B, you
should sample a small number of triples to fit on a small GPU.

TODO: Implement the actual parsing of T‑REx data.  Below is a
template illustrating what to do.  You might need to install the
`kilt` dataset from HuggingFace or download the T‑REx JSON files.
"""

import json
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Build KBLaM++ five‑tuples from T‑REx subset")
    parser.add_argument("input", type=str, help="Path to T‑REx JSONL or similar")
    parser.add_argument("output", type=str, help="Output JSONL path for five‑tuples")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of triples to process")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    count = 0
    with input_path.open() as infile, output_path.open("w") as outfile:
        for line in infile:
            if count >= args.max_samples:
                break
            data = json.loads(line)
            # TODO: Parse T‑REx record.  Each record typically has
            # fields: {"h": entity_id, "r": relation_id, "t": tail_id,
            #         "evidence": [list of sentences], ...}
            # You will need to map entity IDs to names via Wikidata or
            # other metadata.  For Plan B, consider picking only a few
            # relations and using the first evidence sentence.
            # The output five‑tuple should look like:
            # {
            #   "head": {"id": h_id, "name": h_name, "type": h_type},
            #   "relation": {"id": r_id, "name": r_name},
            #   "tail": {"id": t_id, "name": t_name, "type": t_type},
            #   "context": {"source": "trex", "page_title": ..., "sent_span": ...},
            #   "time_window": {"start": null, "end": null, "source": null}
            # }
            # Write as JSON line.
            pass
        print(f"Processed {count} triples")


if __name__ == "__main__":
    main()