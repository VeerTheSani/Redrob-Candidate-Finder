"""Offline eval: how much did the Phase-2 cross-encoder change the ranking vs
Phase-1 alone? Reports top-list overlap and the biggest moves so the reranker's
effect is visible and sanity-checkable (we have no ground truth / leaderboard).

Run:  python compare_rankings.py
"""
import pandas as pd

import rank


def top_ids(df, score_col, n=100):
    ordered = df.sort_values([score_col, "candidate_id"], ascending=[False, True])
    return ordered["candidate_id"].head(n).tolist()


def main():
    df = rank.load_merged()
    rank.compute_phase1(df)
    p1 = top_ids(df, "phase1_score", 100)

    if not rank.reranker_available():
        print("Reranker artifacts missing — nothing to compare.")
        return
    ranked = rank.rerank(df)
    p2 = top_ids(ranked, "final_score", 100)

    p1_rank = {c: i + 1 for i, c in enumerate(p1)}
    p2_rank = {c: i + 1 for i, c in enumerate(p2)}

    print("Top-list overlap (Phase-1 vs Phase-2 reranked):")
    for k in (10, 50, 100):
        ov = len(set(p1[:k]) & set(p2[:k]))
        print(f"  overlap@{k}: {ov}/{k}")

    print("\nBiggest promotions by the reranker (old -> new rank):")
    moves = []
    for c in p2:
        old = p1_rank.get(c)
        moves.append(((old - p2_rank[c]) if old else 10**6, c, old, p2_rank[c]))
    for delta, c, old, new in sorted(moves, reverse=True)[:10]:
        tag = "NEW" if old is None else f"{old}"
        print(f"  {c}: {tag:>4} -> {new}")

    dropped = [c for c in p1 if c not in set(p2)]
    print(f"\nDropped out of the top 100 ({len(dropped)}):")
    for c in dropped[:10]:
        print(f"  {c} (was #{p1_rank[c]})")


if __name__ == "__main__":
    main()
