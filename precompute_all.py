import argparse
import os
import subprocess
import sys

STEPS = [
    ("Embedding candidates + JD (slow, ~33 min)", "precompute/embedded_candidates.py", True),
    ("Building rule features", "precompute/build_features.py", True),
    ("Flagging honeypots", "precompute/flag_honeypots.py", True),
    ("Downloading cross-encoder reranker (needs network)", "precompute/setup_reranker.py", False),
    ("Building rerank texts", "precompute/build_rerank_texts.py", True),
]


def main():
    parser = argparse.ArgumentParser(
        description="Precompute step (slow, network allowed). Builds everything in artifacts/ "
                    "so the ranking step (rank.py) can run offline within 5 minutes."
    )
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.candidates):
        sys.exit(
            f"ERROR: candidate file not found at '{args.candidates}'.\n"
            "It is not shipped in this repo (~487MB). Download candidates.jsonl from the "
            "hackathon source and place it at data/candidates.jsonl, or pass --candidates <path>."
        )
    os.makedirs("artifacts", exist_ok=True)

    for label, script, needs_candidates in STEPS:
        cmd = [sys.executable, script]
        if needs_candidates:
            cmd += ["--candidates", args.candidates]
        print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("\nPrecompute complete. Now run the ranking step offline:")
    print("    python rank.py --out submission.csv")


if __name__ == "__main__":
    main()
