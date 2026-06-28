"""One-time, ONLINE precompute: download the cross-encoder reranker and save it
into the repo (artifacts/reranker/) so the ranking step can load it locally with
no network — which is what makes Stage-3 sandbox reproduction work.

Run once:  python precompute/setup_reranker.py
"""
import argparse

# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--out", default="artifacts/reranker")
    args = parser.parse_args()

    print(f"Downloading cross-encoder '{args.model}' (needs internet, ~120MB)...")
    model = CrossEncoder(args.model)

    # sentence-transformers renamed save()->save_pretrained() across versions;
    # support both so this doesn't break on whatever is installed.
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(args.out)
    else:
        model.save(args.out)

    print(f"Saved reranker to {args.out}/ — rank.py will load it offline from here.")


if __name__ == "__main__":
    main()
