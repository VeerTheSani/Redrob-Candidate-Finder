import os

# Hard guarantee: no network at rank time. The Stage-3 sandbox reproduces this
# step with networking OFF, so the cross-encoder must load from the local repo.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Phase-1 scoring knobs. The model is NOT one big product of everything (that
# let a single weak factor veto a great candidate). Instead:
#   phase1 = relevance_core  ×  soft_modifiers  ×  hard_gates
# relevance_core is an *additive* weighted sum (no single 0 can kill it), soft
# modifiers only nudge within a band, hard gates are the only things that crush.
# ----------------------------------------------------------------------------

W_SIM, W_SKILL, W_EXP, W_PROD = 0.45, 0.35, 0.12, 0.08
CORE_ML_TITLE_BONUS = 0.06

TARGET_REQ_HITS = 4
TARGET_NICE_HITS = 2

LOCATION_FLOOR = 0.60
NOTICE_FLOOR = 0.90
ENGAGEMENT_FLOOR = 0.80

# Over-experience: a TINY nudge only. The JD says "5-9 is a range, not a
# requirement... we'll seriously consider candidates outside the band if other
# signals are strong" — so we must NOT punish seniority hard. Floor 0.92 means a
# 16yr+ candidate loses at most ~8%, enough to break ties toward the 6-8 sweet
# spot but never enough to exclude a strong over-band candidate.
OVEREXP_START = 10
OVEREXP_FULL = 18
OVEREXP_FLOOR = 0.92

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

# ----------------------------------------------------------------------------
# Phase-2 cross-encoder reranker knobs.
#   final = (1 - ALPHA) * phase1_norm  +  ALPHA * cross_encoder_norm
# computed over the top-K shortlist only (the full 100k would blow the budget).
# ----------------------------------------------------------------------------
RERANK_MODEL_PATH = "artifacts/reranker"
RERANK_IDS_PATH = "artifacts/candidate_rerank_ids.npy"
RERANK_TEXTS_PATH = "artifacts/candidate_rerank_texts.npy"
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


def minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


def skill_score(df):
    """0-1 evidence-weighted skill match. Verified hits (corroborated by tenure /
    endorsements / assessment) count most; raw text hits a little; scaled by the
    skill-evidence ratio so keyword stuffers are pulled down."""
    req = 0.7 * np.minimum(df["verified_required_hits"] / TARGET_REQ_HITS, 1.0) \
        + 0.3 * np.minimum(df["required_skill_hits"] / TARGET_REQ_HITS, 1.0)
    nice = 0.5 * np.minimum(df["verified_nice_hits"] / TARGET_NICE_HITS, 1.0) \
        + 0.5 * np.minimum(df["nice_to_have_hits"] / TARGET_NICE_HITS, 1.0)
    base = 0.8 * req + 0.2 * nice
    return base * (0.6 + 0.4 * df["skill_evidence_ratio"])


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


def _clean(v):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip()


def generate_reasoning(row):
    """Specific, varied, honest 1-2 sentence justification built only from fields
    that exist on the candidate (Stage-4 checks for named facts, JD connection,
    honest concerns, no hallucination)."""
    title = _clean(row["current_title"]) or "Engineer"
    company = _clean(row["current_company"])
    yrs = row["years_of_experience"]
    skills = _clean(row["matched_skill_names"])
    concern = _clean(row["primary_concern"])
    vreq = int(row["verified_required_hits"])
    rreq = int(row["required_skill_hits"])
    prod = float(row["product_company_fraction"])

    where = f" at {company}" if company else ""
    lead = f"{title}{where}, {yrs:g} yrs"

    if vreq >= 3:
        strength = "strong, evidenced retrieval/ranking depth"
    elif vreq >= 1:
        strength = "demonstrated retrieval/ranking skills"
    elif rreq >= 1:
        strength = "some retrieval/ranking exposure, lightly evidenced"
    else:
        strength = "adjacent ML background with limited direct retrieval signal"
    if skills:
        strength += f" ({skills})"
    sentence1 = f"{lead} — {strength}."

    if prod >= 0.7:
        ctx = "Largely product-company experience, matching the JD's preference."
    elif prod <= 0.3:
        ctx = "Mostly services-company background."
    else:
        ctx = "Mixed product/services background."

    if concern:
        ctx += f" Concern: {concern}."
    elif vreq >= 3:
        ctx += " No major red flags."

    return f"{sentence1} {ctx}".strip()


def load_merged():
    sims = compute_similarity("artifacts/candidate_embeddings.npy", "artifacts/jd_embedding.npy")
    ids = np.load("artifacts/candidate_ids.npy")
    df_sim = pd.DataFrame({"candidate_id": ids, "cosine_sim": sims})
    df_feat = pd.read_csv("artifacts/features.csv")
    df_hp = pd.read_csv("artifacts/honeypot_flags.csv")
    df = df_sim.merge(df_feat, on="candidate_id").merge(df_hp, on="candidate_id")
    return coerce_bools(df)


def compute_phase1(df):
    """Phase-1 score (relevance core × soft modifiers × hard gates). Sets and
    returns df['phase1_score']."""
    sim_norm = minmax(df["cosine_sim"].to_numpy(dtype=float))
    sk = skill_score(df).to_numpy(dtype=float)
    relevance = (
        W_SIM * sim_norm
        + W_SKILL * sk
        + W_EXP * df["experience_fit"].to_numpy(dtype=float)
        + W_PROD * df["product_company_fraction"].to_numpy(dtype=float)
        + CORE_ML_TITLE_BONUS * df["is_core_ml_title"].to_numpy(dtype=float)
    )
    relevance = np.clip(relevance, 0.0, 1.0)

    location_mod = LOCATION_FLOOR + (1 - LOCATION_FLOOR) * df["location_fit"].to_numpy(dtype=float)
    notice_mod = NOTICE_FLOOR + (1 - NOTICE_FLOOR) * df["notice_period_fit"].to_numpy(dtype=float)
    engagement_mod = ENGAGEMENT_FLOOR + (1 - ENGAGEMENT_FLOOR) * engagement_score(df)

    yrs = df["years_of_experience"].to_numpy(dtype=float)
    overexp_mod = np.clip(
        1.0 - np.maximum(yrs - OVEREXP_START, 0.0) / (OVEREXP_FULL - OVEREXP_START) * (1 - OVEREXP_FLOOR),
        OVEREXP_FLOOR, 1.0,
    )

    score = relevance * location_mod * notice_mod * engagement_mod * overexp_mod

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


def reranker_available():
    return (os.path.isdir(RERANK_MODEL_PATH)
            and os.path.exists(RERANK_IDS_PATH)
            and os.path.exists(RERANK_TEXTS_PATH))


def rerank(df):
    """Cross-encoder reranks the top-K Phase-1 shortlist (traps excluded), then
    blends with the Phase-1 score. Returns the shortlist with 'final_score'."""
    # pyrefly: ignore [missing-import]
    from sentence_transformers import CrossEncoder

    rids = np.load(RERANK_IDS_PATH, allow_pickle=True)
    rtexts = np.load(RERANK_TEXTS_PATH, allow_pickle=True)
    id2text = dict(zip(rids.tolist(), rtexts.tolist()))

    eligible = df[~df["is_honeypot"] & ~df["is_irrelevant_title"]]
    shortlist = eligible.sort_values(
        ["phase1_score", "candidate_id"], ascending=[False, True]
    ).head(SHORTLIST_K).copy()

    texts = [id2text.get(cid, "") for cid in shortlist["candidate_id"]]
    model = CrossEncoder(RERANK_MODEL_PATH)
    ce_raw = np.asarray(
        model.predict(list(zip([JD_QUERY] * len(texts), texts)), show_progress_bar=False),
        dtype=float,
    )

    ce_norm = minmax(ce_raw)
    p1_norm = minmax(shortlist["phase1_score"].to_numpy(dtype=float))
    shortlist["final_score"] = np.round((1 - ALPHA) * p1_norm + ALPHA * ce_norm, 6)
    return shortlist


def main():
    start_time = time.time()

    print("Loading artifacts...")
    df = load_merged()

    print("Computing Phase-1 scores...")
    compute_phase1(df)

    if reranker_available():
        print(f"Reranking top-{SHORTLIST_K} shortlist with cross-encoder (alpha={ALPHA})...")
        ranked = rerank(df)
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

    out_path = "submission.csv"
    submission.to_csv(out_path, index=False)

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f} seconds. Wrote to {out_path}.")

    print("\n--- Top 10 Candidates ---")
    for _, r in submission.head(10).iterrows():
        print(f"{r['rank']}. {r['candidate_id']} | Score: {r['score']:.4f}")
        print(f"   {r['reasoning']}")


if __name__ == "__main__":
    main()
