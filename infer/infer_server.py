"""A simple inference server for KBLaM++ (stub).

This script illustrates how you might wrap a trained KBLaM++
model into a REST API or command line interface.  For Plan B
we provide a CLI that reads a question from stdin and prints a
dummy answer.  To extend this into a real service, you would load
your checkpoint, perform the forward pass with knowledge injection
and return the predicted answer along with the evidence chain.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="KBLaM++ inference server (stub)")
    parser.add_argument("--model", type=str, help="Path to trained model")
    args = parser.parse_args()
    print("KBLaM++ inference (stub). Type a question and press ENTER (Ctrl-D to quit).")
    for line in sys.stdin:
        question = line.strip()
        if not question:
            continue
        # TODO: tokenise question, run model, produce answer and evidence chain
        print(f"You asked: {question}")
        print("Answer: <stub answer>")
        print("Evidence: ...")


if __name__ == "__main__":
    main()