###   
# 
# 
# 
# ### Sandbox demo (HuggingFace Space, Gradio).
import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import sys
import json
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "precompute"))

import numpy as np
import pandas as pd
import gradio as gr
from sentence_transformers import SentenceTransformer, CrossEncoder

import build_features as bf
import flag_honeypots as fh
import embedded_candidates as ec
import build_rerank_texts as brt
import rank

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
MAX_CANDIDATES = 100

print("Loading models (first run downloads ~210MB)...")
_embedder = SentenceTransformer(EMBED_MODEL)
_reranker = CrossEncoder(RERANK_MODEL)
with open("data/job_description.txt", encoding="utf-8") as f:
    _jd_emb = _embedder.encode([ec.jd_embedding_text(f.read())], normalize_embeddings=True)[0]


def _load_candidates(path):
    text = open(path, encoding="utf-8").read().strip()
    try:
        data = json.loads(text)
        return [data] if isinstance(data, dict) else data
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def rank_candidates(file_obj, top_n):
    if file_obj is None:
        return pd.DataFrame(), None
    path = file_obj.name if hasattr(file_obj, "name") else file_obj
    candidates = _load_candidates(path)[:MAX_CANDIDATES]
    if not candidates:
        return pd.DataFrame(), None

    today = datetime.date.today()
    feat = pd.DataFrame([bf.build_row(c, today) for c in candidates])
    hp = pd.DataFrame([fh.evaluate_candidate(c) for c in candidates])
    embs = _embedder.encode([ec.candidate_text(c) for c in candidates], normalize_embeddings=True)
    sim = pd.DataFrame({
        "candidate_id": [c["candidate_id"] for c in candidates],
        "cosine_sim": embs @ _jd_emb,
    })

    df = sim.merge(feat, on="candidate_id").merge(hp, on="candidate_id")
    df = rank.coerce_bools(df)
    rank.compute_phase1(df)

    by_id = {c["candidate_id"]: c for c in candidates}
    elig = df[~df["is_honeypot"] & ~df["is_irrelevant_title"]].copy()
    elig = elig.sort_values(["phase1_score", "candidate_id"], ascending=[False, True])
    texts = [brt.rerank_text(by_id[cid]) for cid in elig["candidate_id"]]
    if texts:
        ce = np.asarray(
            _reranker.predict(list(zip([rank.JD_QUERY] * len(texts), texts)), show_progress_bar=False),
            dtype=float,
        )
        elig["final_score"] = np.round(
            (1 - rank.ALPHA) * rank.pct(elig["phase1_score"].to_numpy(float))
            + rank.ALPHA * rank.pct(ce), 6,
        )
    elig = elig.sort_values(["final_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    n = min(int(top_n), len(elig))
    out = elig.head(n).copy()
    out["rank"] = np.arange(1, n + 1)
    out["reasoning"] = [rank.generate_reasoning(r) for _, r in out.iterrows()]
    result = out[["candidate_id", "rank", "final_score", "reasoning"]].rename(columns={"final_score": "score"})

    csv_path = "ranked_sample.csv"
    result.to_csv(csv_path, index=False)
    return result, csv_path


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown(
        "# Redrob Candidate Ranker\n"
        "Upload a small candidate sample (≤100, a JSON array or JSONL matching the "
        "hackathon schema) and rank them against the **Senior AI Engineer** JD. "
        "The full pipeline runs live: embeddings → rule features → honeypot filter → "
        "scoring → cross-encoder rerank."
    )
    with gr.Row():
        inp = gr.File(label="candidates (.json / .jsonl)", file_types=[".json", ".jsonl"])
        topn = gr.Slider(1, 100, value=20, step=1, label="Top N to return")
    btn = gr.Button("Rank", variant="primary")
    out_df = gr.Dataframe(label="Ranking", wrap=True)
    out_csv = gr.File(label="Download ranked CSV")
    btn.click(rank_candidates, inputs=[inp, topn], outputs=[out_df, out_csv])
    gr.Examples(examples=[["data/sample_candidates.json", 20]], inputs=[inp, topn])


if __name__ == "__main__":
    demo.launch()
