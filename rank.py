import os
import time
import numpy as np
import pandas as pd

def compute_similarity(embeddings_path, jd_path):
    emb = np.load(embeddings_path)
    jd = np.load(jd_path)
    # Cosine similarity is just dot product since they are normalized
    return emb @ jd

def generate_reasoning(row, rank):
    title = row['current_title']
    yrs = row['years_of_experience']
    req = int(row['required_skill_hits'])
    nice = int(row['nice_to_have_hits'])
    resp = row['recruiter_response_rate']
    
    # Base facts
    base = f"{title} with {yrs} yrs of experience."
    
    # Specific JD connection
    if req > 0:
        skills = f"Solid fit for JD requirements with {req} core retrieval/ranking skills and {nice} nice-to-have skills."
    else:
        skills = f"Lacks core retrieval skills, but included for general ML background ({nice} nice-to-have skills)."
        
    # Tone adjustment based on rank
    if rank <= 20:
        tone = "Top-tier candidate. "
    elif rank <= 50:
        tone = "Strong candidate. "
    elif rank <= 80:
        tone = "Solid alternative. "
    else:
        tone = "Borderline candidate. "
        
    # Signal facts
    signal = f"Response rate: {resp:.2f}."
        
    return f"{tone}{base} {skills} {signal}"

def main():
    start_time = time.time()
    
    print("Loading artifacts...")
    sims = compute_similarity("artifacts/candidate_embeddings.npy", "artifacts/jd_embedding.npy")
    ids = np.load("artifacts/candidate_ids.npy")
    
    df_sim = pd.DataFrame({"candidate_id": ids, "cosine_sim": sims})
    df_feat = pd.read_csv("artifacts/features.csv")
    df_hp = pd.read_csv("artifacts/honeypot_flags.csv")
    
    print("Merging data...")
    df = df_sim.merge(df_feat, on="candidate_id").merge(df_hp, on="candidate_id")
    
    print("Computing final scores...")
    # Base multiplier from cosine sim and features
    score = df["cosine_sim"] * df["experience_fit"] * df["location_fit"] * df["notice_period_fit"] * df["behavioral_modifier"]
    
    # Massive boost for required skills (this JD cares about actual retrieval/ranking experience)
    score = score * (1 + df["required_skill_hits"] * 0.5 + df["nice_to_have_hits"] * 0.15)
    
    # Boost core ML titles massively
    score = np.where(df["is_core_ml_title"], score * 2.0, score)
    
    # Penalize disqualifiers according to the JD
    score = np.where(df["is_consulting_only"], score * 0.1, score)
    score = np.where(df["is_vision_speech_only"], score * 0.1, score)
    score = np.where(df["is_title_chaser"], score * 0.1, score)
    score = np.where(df["is_recent_ai_only"], score * 0.1, score)
    score = np.where(df["is_architect_not_coding"], score * 0.1, score)
    
    # Irrelevant titles are an instant fail (trap candidates)
    score = np.where(df["is_irrelevant_title"], 0.0, score)
    
    # Product company preference
    score = score * (0.8 + 0.2 * df["product_company_fraction"])
    
    # Zero out honeypots completely
    score = np.where(df["is_honeypot"], 0.0, score)
    
    # Calculate Theoretical Maximum Possible Score for Absolute Percentage
    t_sim = 1.0
    t_exp = df["experience_fit"].max()
    t_loc = df["location_fit"].max()
    t_not = df["notice_period_fit"].max()
    t_beh = df["behavioral_modifier"].max()
    t_req = df["required_skill_hits"].max()
    t_nice = df["nice_to_have_hits"].max()
    t_prod = df["product_company_fraction"].max()
    
    theoretical_max = (t_sim * t_exp * t_loc * t_not * t_beh) * \
                      (1 + t_req * 0.5 + t_nice * 0.15) * \
                      2.0 * \
                      (0.8 + 0.2 * t_prod)
                      
    if theoretical_max > 0:
        score = score / theoretical_max
        
    # Clip just in case, though it shouldn't exceed 1.0
    score = np.clip(score, 0.0, 1.0)
    
    df["final_score"] = score
    
    print("Sorting and generating submission...")
    # Sort descending by score, tie-break by candidate_id ascending deterministically
    df = df.sort_values(["final_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    
    top_100 = df.head(100).copy()
    top_100["rank"] = np.arange(1, 101)
    
    # Generate reasoning per candidate
    reasonings = []
    for idx, row in top_100.iterrows():
        reasonings.append(generate_reasoning(row, row["rank"]))
    top_100["reasoning"] = reasonings
    
    # Format and save
    submission = top_100[["candidate_id", "rank", "final_score", "reasoning"]].copy()
    submission.rename(columns={"final_score": "score"}, inplace=True)
    
    out_path = "submission.csv"
    submission.to_csv(out_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f} seconds. Wrote to {out_path}.")
    
    # Some quick stats on the top 10
    print("\n--- Top 10 Candidates ---")
    for i, r in submission.head(10).iterrows():
        print(f"{r['rank']}. {r['candidate_id']} | Score: {r['score']:.4f}")
        print(f"   Reasoning: {r['reasoning']}")

if __name__ == "__main__":
    main()
