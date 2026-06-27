

import argparse
import json
import os

import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

##embedddin model
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def candidate_text(candidate: dict) -> str:

   ##the job description has twice weightage than candidates profile, bcz its more pracctical experience thaat matters more ?idk bruh
    profile = candidate["profile"]

    ## sshuity codefasc history_text = " ".join(job["description"] for job in candidate["career_history"])
## be bettter 
    descriptions = []                                    
    for job in candidate["career_history"]:             
        descriptions.append(job["description"])          
    history_text = " ".join(descriptions)

    ## be better, just goated
    skills = []
    for skill in candidate["skills"]:               
        skills.append(skill["name"])
    skills_text = ", ".join(skills)

    return (
        f"{history_text} {history_text} " ##intenational, tiwce the force double the fall, master count dooku said this trash
        f"{profile['headline']}. {profile['summary']} "
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
        jd_text = f.read()
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