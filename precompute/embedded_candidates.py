

import argparse
import json
import os

import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

##embedddin model
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 456


# The ONLY thing we embed for the JD. MiniLM truncates at 256 tokens, so this is a
# tight, candidate-shaped restatement of the ideal profile that fits inside that
# window — every term taken from job_description.txt (must-haves + stated "ideal
# candidate"), nothing invented. Embedding the full JD (~1900 tokens) would be 86%
# truncated AND dilute the vector with narrative, so we just don't, lol.
JD_FOCUS = (
    "Ideal candidate: Senior AI Engineer owning the ranking, retrieval and matching "
    "systems of a product. 5-9 years experience (ideal 6-8, 4-5 in applied ML/AI at "
    "product companies, not services). Core requirements: production embeddings-based "
    "retrieval (sentence-transformers, BGE, E5, OpenAI embeddings); vector databases "
    "and hybrid search (Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, "
    "Elasticsearch); ranking, hybrid retrieval and LLM re-ranking; evaluation of "
    "ranking systems (NDCG, MRR, MAP, A/B testing); strong production Python. Has "
    "shipped an end-to-end ranking, search, or recommendation system to real users at "
    "scale. Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank, "
    "HR-tech. Located in or relocating to Pune or Noida, India."
)


def jd_embedding_text(raw: str) -> str:
    """We embed ONLY the tight ideal-profile summary. The full JD is ~1900 tokens and
    MiniLM truncates at 256, so appending it would be dropped anyway and dilute the
    query. `raw` is kept for signature compatibility with main()."""
    return JD_FOCUS


def candidate_text(candidate: dict) -> str:
    # Weighting scheme (documented on purpose):
    #    career-history descriptions = strongest evidence of real applied work,
    #     so they get 2x weight.
    #   current title + headline + summary + skills each contribute once.
    profile = candidate["profile"]
    history_text = " ".join(job["description"] for job in candidate["career_history"])
    skills_text = ", ".join(s["name"] for s in candidate.get("skills", []))

    return (
        f"{profile['current_title']}. {profile['headline']}. "
        f"{profile['summary']} "
        f"{history_text} {history_text} "
        f"Skills: {skills_text}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--jd", default="data/job_description.txt")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    ## os.makedirs(args.out_dir, exist_ok=True) #if you dont have artifact folder , this will safely create it tho ive alr put it but still, maybe letys
    model = SentenceTransformer(MODEL_NAME)

    # --- JD embedding (just one vector) ---
    with open(args.jd, "r", encoding="utf-8") as f:
        jd_text = jd_embedding_text(f.read())
    jd_embedding = model.encode([jd_text], normalize_embeddings=True)[0]
    np.save(os.path.join(args.out_dir, "jd_embedding.npy"), jd_embedding)
    print("Saved jd_embedding.npy")

    # --- candidate embeddings (all of them, batched) ---
    ids, texts = [], []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            ids.append(candidate["candidate_id"])
            texts.append(candidate_text(candidate))

    print(f"Embedding {len(texts)} candidates in batches of {BATCH_SIZE}... "
          f"(this is the a freaking slow step (increase batch size if your pc is high-end), let it run)")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # this sht hides some nonsense i really , relevent to cosine but will see
        show_progress_bar=True, # its just for seeing, show progress..                        what? you expected a movie?
    )

    np.save(os.path.join(args.out_dir, "candidate_embeddings.npy"), embeddings)
    np.save(os.path.join(args.out_dir, "candidate_ids.npy"), np.array(ids))

    print(f"Saved candidate_embeddings.npy + candidate_ids.npy for {len(ids)} candidates")


if __name__ == "__main__":
    main()
