"""Precompute (cheap, ~1-2 min): a compact, truncation-friendly document per
candidate for the cross-encoder reranker. Kept separate from the slow embedding
step so text tweaks never re-trigger the 33-min encode.

Writes two aligned arrays to artifacts/:
  - candidate_rerank_ids.npy    (candidate_id order)
  - candidate_rerank_texts.npy  (the rerank document for each id)

rank.py joins them by id, so this does not need to match candidate_ids.npy order.
"""
import argparse
import json
import os

import numpy as np

MAX_CHARS = 1500  # ~400 tokens; the cross-encoder truncates anyway, this just
                  # keeps predict() fast and the artifact small.


def rerank_text(candidate: dict) -> str:
    p = candidate["profile"]
    history = candidate.get("career_history", [])
    current = next((j for j in history if j.get("is_current")), None)
    if current is None and history:
        current = history[0]
    recent_desc = current["description"] if current else ""

    skills = ", ".join(s["name"] for s in candidate.get("skills", [])[:12])
    text = (
        f"{p['current_title']} at {p.get('current_company', '')}. "
        f"{p.get('headline', '')}. {p.get('summary', '')} "
        f"Recent role: {recent_desc} Skills: {skills}"
    )
    text = " ".join(text.split())  # collapse whitespace/newlines
    return text[:MAX_CHARS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    ids, texts = [], []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            ids.append(candidate["candidate_id"])
            texts.append(rerank_text(candidate))

    np.save(os.path.join(args.out_dir, "candidate_rerank_ids.npy"), np.array(ids))
    np.save(os.path.join(args.out_dir, "candidate_rerank_texts.npy"), np.array(texts, dtype=object))
    print(f"Wrote rerank texts for {len(ids)} candidates to {args.out_dir}/")


if __name__ == "__main__":
    main()
