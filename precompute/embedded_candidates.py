

import argparse
import json
import os

import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

##embedddin model
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 700


def dedup_lines(text: str) -> str:
    """The JD repeats its 'Things you absolutely need' block verbatim. Repeated
    identical lines add no semantic signal to the embedding, so keep only the
    first occurrence (blank lines preserved)."""
    seen, out = set(), []
    for line in text.splitlines():
        key = line.strip()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out)


# A concise, requirement-forward restatement of the JD. Every term here is taken
# directly from job_description.txt (the must-haves + the stated "ideal candidate")
# — nothing is invented. We prepend it so the averaged JD vector is dominated by
# what the role actually needs, instead of the conversational tone/framing that
# makes up ~1/3 of the raw posting and dilutes the signal.
JD_FOCUS = (
    "Ideal candidate: Senior AI Engineer to own the intelligence layer — the ranking, "
    "retrieval and matching systems — of an AI-native talent platform. 5-9 years "
    "experience (ideal 6-8, with 4-5 in applied ML/AI roles at product companies, not "
    "services). Core requirements: production embeddings-based retrieval deployed to "
    "real users (sentence-transformers, BGE, E5, OpenAI embeddings; embedding drift, "
    "index refresh, retrieval-quality regression); vector databases and hybrid search "
    "(Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch); ranking, "
    "hybrid retrieval and LLM-based re-ranking; rigorous evaluation of ranking systems "
    "(NDCG, MRR, MAP, offline-to-online correlation, A/B testing); strong production "
    "Python and code quality. Has shipped at least one end-to-end ranking, search, or "
    "recommendation system to real users at meaningful scale, with strong, defensible "
    "opinions about retrieval, evaluation and LLM integration. Scrappy product-"
    "engineering attitude. Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), "
    "learning-to-rank, HR-tech. Located in or willing to relocate to Pune or Noida, "
    "India; active and reachable on the platform."
)

# JD sections that would MISLEAD a bi-encoder (it would pull the JD vector toward the
# very things we want to avoid) or add pure noise. We drop them from the embedding;
# the "do NOT want" rules are enforced as disqualifier features in build_features.py.
_DROP_SECTIONS = {
    "explicitly do not want": "on location",           # negatives -> features, not embedding
    "the vibe check": "how to read between the lines",  # culture, not skills
    "final note for the participants": None,            # hackathon meta -> drop to end
}


def strip_negative_sections(text: str) -> str:
    out, skip, resume = [], False, None
    for line in text.splitlines():
        low = line.strip().lower()
        if not skip:
            hit = next((r for s, r in _DROP_SECTIONS.items() if s in low), "MISS")
            if hit != "MISS":
                skip, resume = True, hit
                continue
            out.append(line)
        elif resume is not None and resume in low:
            skip = False
            out.append(line)
    return "\n".join(out)


def jd_embedding_text(raw: str) -> str:
    """Curated positive focus first, then the JD with misleading negative/noise
    sections removed (so the bi-encoder isn't dragged toward 'do NOT want' terms)."""
    return JD_FOCUS + "\n\n" + dedup_lines(strip_negative_sections(raw))


def candidate_text(candidate: dict) -> str:
    # Weighting scheme (documented on purpose):
    #   - career-history descriptions = strongest evidence of real applied work,
    #     so they get 2x weight.
    #   - current title + headline + summary + skills each contribute once.
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
        normalize_embeddings=True,  # this sht hides some nonsense i really dont understand it , relevent to cosine but will see
        show_progress_bar=True, # its just for seeing, show progress..                        what? you expected a movie?
    )

    np.save(os.path.join(args.out_dir, "candidate_embeddings.npy"), embeddings)
    np.save(os.path.join(args.out_dir, "candidate_ids.npy"), np.array(ids))

    print(f"Saved candidate_embeddings.npy + candidate_ids.npy for {len(ids)} candidates")


if __name__ == "__main__":
    main()