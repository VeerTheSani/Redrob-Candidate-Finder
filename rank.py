import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "precompute"))
# pyrefly: ignore [missing-import]
from reasoning import generate_reasoning

W_SIM, W_SKILL, W_SYSTEM, W_EXP, W_PROD = 0.40, 0.30, 0.15, 0.10, 0.05
CORE_ML_TITLE_BONUS = 0.06


LOCATION_FLOOR = 0.60
NOTICE_FLOOR = 0.90
ENGAGEMENT_FLOOR = 0.80

# Recruiter responsiveness penalty: no penalty at/above RESPONSE_OK, then a steep
# linear drop to RESPONSE_FLOOR at response_rate 0 (a near-zero responder keeps ~40%).
RESPONSE_OK = 0.40
RESPONSE_FLOOR = 0.40

# Experience band (JD headline: 5-9 yrs). FIRM two-sided penalty: lose BAND_PER_YEAR
# per year outside [BAND_MIN, BAND_MAX], down to BAND_FLOOR. So <5 or >9 is clearly
# down-weighted — but floored (not zeroed) so a truly exceptional outlier can survive.
BAND_MIN, BAND_MAX = 5, 9
BAND_PER_YEAR = 0.30    # HARD: 1 yr out = -30%, 2 yrs out = -60%
BAND_FLOOR = 0.20       # far out-of-band keeps only 20%

SAVED_CAP = 10
SEARCH_CAP = 50

PENALTY_CONSULTING_ONLY = 0.15
PENALTY_VISION_SPEECH = 0.15
PENALTY_RECENT_AI = 0.30
PENALTY_TITLE_CHASER = 0.40
PENALTY_ARCHITECT = 0.50
PENALTY_PURE_RESEARCH = 0.15      # JD: academic/research-only, no production
PENALTY_FRAMEWORK = 0.25         # JD: LangChain-wrapper hobbyists, no real depth
# Disabled by default (1.0 = no-op): the available proxy flags ~21% and mostly
# means "private engineer", not "no external validation". Set < 1.0 to enable.
PENALTY_CLOSED_SOURCE = 1.0

RERANK_MODEL_DIR = "reranker"
RERANK_IDS_FILE = "candidate_rerank_ids.npy"
RERANK_TEXTS_FILE = "candidate_rerank_texts.npy"
SHORTLIST_K = 400
ALPHA = 0.5
JD_QUERY = (
    "Senior AI Engineer for production embeddings-based retrieval and ranking: "
    "sentence-transformers, BGE, E5, vector databases (Pinecone, Weaviate, Qdrant, "
    "Milvus, FAISS, OpenSearch, Elasticsearch), hybrid search, LLM re-ranking, "
    "evaluation with NDCG / MRR / MAP and A/B testing, strong Python, product "
    "company, 5-9 years experience, located in or relocating to Pune or Noida India."
)

BOOL_COLS = [
    "is_consulting_only", "is_vision_speech_only", "is_title_chaser",
    "is_architect_not_coding", "is_recent_ai_only", "is_irrelevant_title",
    "is_core_ml_title", "open_to_work_flag", "is_honeypot",
    "is_pure_research", "is_framework_enthusiast", "is_closed_source_no_validation",
]


def compute_similarity(embeddings_path, jd_path):
    emb = np.load(embeddings_path)
    jd = np.load(jd_path)
    return emb @ jd  # cosine == dot product (both L2-normalized)


def coerce_bools(df):
    """Booleans round-trip through CSV as 'True'/'False' strings; reading them
    back naively makes 'False' truthy, so coerce explicitly."""
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().isin(["true", "1"])
    return df


def pct(x):
    # percentile rank in (0,1]: a candidate's standing vs the whole pool on this
    # signal. No caps/floors, robust to outliers, and ties are averaged — so the
    # max-offering candidate rises and nobody is flattened at a ceiling.
    return pd.Series(np.asarray(x, dtype=float)).rank(pct=True).to_numpy()


def skill_score(df):
    # Continuous, uncapped skill evidence: graded strengths (not binary hits), plus a
    # little credit for raw text mentions, scaled by the evidence ratio (anti-stuffing).
    # Percentile-ranked so it differentiates across the whole pool with no ceiling.
    raw = (df["verified_required_strength"]
           + 0.4 * df["verified_nice_strength"]
           + 0.15 * df["required_skill_hits"])
    raw = raw * (0.6 + 0.4 * df["skill_evidence_ratio"])
    return pct(raw)


def engagement_score(df):
    saved_norm = np.minimum(df["saved_by_recruiters_30d"] / SAVED_CAP, 1.0)
    search_norm = np.minimum(df["search_appearance_30d"] / SEARCH_CAP, 1.0)
    open_score = np.where(df["open_to_work_flag"], 1.0, 0.5)
    parts = np.vstack([
        df["behavioral_modifier"].to_numpy(dtype=float),
        df["interview_completion_rate"].to_numpy(dtype=float),
        df["offer_acceptance_rate"].to_numpy(dtype=float),
        open_score,
        saved_norm,
        search_norm,
    ])
    return parts.mean(axis=0)


def load_merged(artifacts="artifacts"):
    sims = compute_similarity(os.path.join(artifacts, "candidate_embeddings.npy"),
                              os.path.join(artifacts, "jd_embedding.npy"))
    ids = np.load(os.path.join(artifacts, "candidate_ids.npy"))
    df_sim = pd.DataFrame({"candidate_id": ids, "cosine_sim": sims})
    df_feat = pd.read_csv(os.path.join(artifacts, "features.csv"))
    df_hp = pd.read_csv(os.path.join(artifacts, "honeypot_flags.csv"))
    df = df_sim.merge(df_feat, on="candidate_id").merge(df_hp, on="candidate_id")
    return coerce_bools(df)


def compute_phase1(df):
    """Phase-1 score (relevance core × soft modifiers × hard gates). Sets and
    returns df['phase1_score']."""
    sim_norm = pct(df["cosine_sim"])
    sk = skill_score(df)
    # "built a JD-type system" signal: saturating curve so non-builders sit at 0 and
    # builders scale up (the JD's headline fit: shipped a ranking/search/rec system).
    system_signal = 1.0 - np.exp(-df["system_build_score"].to_numpy(dtype=float) / 1.5)
    relevance = (
        W_SIM * sim_norm
        + W_SKILL * sk
        + W_SYSTEM * system_signal
        + W_EXP * df["experience_fit"].to_numpy(dtype=float)
        + W_PROD * df["product_company_fraction"].to_numpy(dtype=float)
        + CORE_ML_TITLE_BONUS * df["is_core_ml_title"].to_numpy(dtype=float)
    )
    relevance = np.clip(relevance, 0.0, 1.0)

    location_mod = LOCATION_FLOOR + (1 - LOCATION_FLOOR) * df["location_fit"].to_numpy(dtype=float)
    notice_mod = NOTICE_FLOOR + (1 - NOTICE_FLOOR) * df["notice_period_fit"].to_numpy(dtype=float)
    engagement_mod = ENGAGEMENT_FLOOR + (1 - ENGAGEMENT_FLOOR) * engagement_score(df)

    yrs = df["years_of_experience"].to_numpy(dtype=float)
    band_dist = np.maximum(BAND_MIN - yrs, 0.0) + np.maximum(yrs - BAND_MAX, 0.0)
    band_mod = np.clip(1.0 - band_dist * BAND_PER_YEAR, BAND_FLOOR, 1.0)

    resp = df["recruiter_response_rate"].to_numpy(dtype=float)
    response_mod = np.where(
        resp >= RESPONSE_OK,
        1.0,
        RESPONSE_FLOOR + (1 - RESPONSE_FLOOR) * (resp / RESPONSE_OK),
    )

    score = relevance * location_mod * notice_mod * engagement_mod * band_mod * response_mod

    score = np.where(df["is_consulting_only"], score * PENALTY_CONSULTING_ONLY, score)
    score = np.where(df["is_vision_speech_only"], score * PENALTY_VISION_SPEECH, score)
    score = np.where(df["is_recent_ai_only"], score * PENALTY_RECENT_AI, score)
    score = np.where(df["is_title_chaser"], score * PENALTY_TITLE_CHASER, score)
    score = np.where(df["is_architect_not_coding"], score * PENALTY_ARCHITECT, score)
    score = np.where(df["is_pure_research"], score * PENALTY_PURE_RESEARCH, score)
    score = np.where(df["is_framework_enthusiast"], score * PENALTY_FRAMEWORK, score)
    score = np.where(df["is_closed_source_no_validation"], score * PENALTY_CLOSED_SOURCE, score)
    score = np.where(df["is_irrelevant_title"], 0.0, score)
    score = np.where(df["is_honeypot"], 0.0, score)

    df["phase1_score"] = np.clip(score, 0.0, 1.0)
    return df


def reranker_available(artifacts="artifacts"):
    return (os.path.isdir(os.path.join(artifacts, RERANK_MODEL_DIR))
            and os.path.exists(os.path.join(artifacts, RERANK_IDS_FILE))
            and os.path.exists(os.path.join(artifacts, RERANK_TEXTS_FILE)))


def rerank(df, artifacts="artifacts"):
    """Cross-encoder reranks the top-K Phase-1 shortlist (traps excluded), then
    blends with the Phase-1 score. Returns the shortlist with 'final_score'."""
    # pyrefly: ignore [missing-import]
    from sentence_transformers import CrossEncoder

    rids = np.load(os.path.join(artifacts, RERANK_IDS_FILE), allow_pickle=True)
    rtexts = np.load(os.path.join(artifacts, RERANK_TEXTS_FILE), allow_pickle=True)
    id2text = dict(zip(rids.tolist(), rtexts.tolist()))

    eligible = df[~df["is_honeypot"] & ~df["is_irrelevant_title"]]
    shortlist = eligible.sort_values(
        ["phase1_score", "candidate_id"], ascending=[False, True]
    ).head(SHORTLIST_K).copy()

    texts = [id2text.get(cid, "") for cid in shortlist["candidate_id"]]
    model = CrossEncoder(os.path.join(artifacts, RERANK_MODEL_DIR))
    ce_raw = np.asarray(
        model.predict(list(zip([JD_QUERY] * len(texts), texts)), show_progress_bar=False),
        dtype=float,
    )

    ce_norm = pct(ce_raw)
    p1_norm = pct(shortlist["phase1_score"])
    shortlist["final_score"] = np.round((1 - ALPHA) * p1_norm + ALPHA * ce_norm, 6)
    return shortlist


def main():
    parser = argparse.ArgumentParser(description="Ranking step (offline, CPU, <=5 min). Reads precomputed artifacts/.")
    parser.add_argument("--out", default="submission.csv", help="output CSV path")
    parser.add_argument("--artifacts", default="artifacts", help="dir with precomputed artifacts")
    args = parser.parse_args()

    start_time = time.time()

    print("Loading artifacts...")
    df = load_merged(args.artifacts)

    print("Computing Phase-1 scores...")
    compute_phase1(df)

    if reranker_available(args.artifacts):
        print(f"Reranking top-{SHORTLIST_K} shortlist with cross-encoder (alpha={ALPHA})...")
        ranked = rerank(df, args.artifacts)
    else:
        print("Reranker artifacts not found — falling back to Phase-1 score only.")
        ranked = df.copy()
        ranked["final_score"] = np.round(ranked["phase1_score"], 6)

    print("Sorting and generating submission...")
    # descending score; deterministic tie-break by candidate_id ascending,
    # decided on the *rounded* score (matches validate_submission.py).
    ranked = ranked.sort_values(["final_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    top_100 = ranked.head(100).copy()
    top_100["rank"] = np.arange(1, 101)
    top_100["reasoning"] = [generate_reasoning(r) for _, r in top_100.iterrows()]

    submission = top_100[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    submission.rename(columns={"final_score": "score"}, inplace=True)

    out_path = args.out
    submission.to_csv(out_path, index=False)

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f} seconds. Wrote to {out_path}.")

    print("\n--- Top 10 Candidates ---")
    for _, r in submission.head(10).iterrows():
        print(f"{r['rank']}. {r['candidate_id']} | Score: {r['score']:.4f}")
        print(f"   {r['reasoning']}")


if __name__ == "__main__":
    main()
