

import argparse
import json
from datetime import date

import pandas as pd


CONSULTING_ONLY_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
}
SERVICE_INDUSTRY_KEYWORDS = {"it services", "consulting", "bpo", "outsourcing"}

REQUIRED_SKILL_KEYWORDS = {
    "embeddings", "sentence-transformers", "bge", "e5",
    "vector database", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss", "retrieval", "ranking",
    "ndcg", "mrr", "map", "evaluation",
}
NICE_TO_HAVE_KEYWORDS = {
    "lora", "qlora", "peft", "fine-tuning", "learning-to-rank", "xgboost",
}
ALL_AI_KEYWORDS = REQUIRED_SKILL_KEYWORDS | NICE_TO_HAVE_KEYWORDS

VISION_SPEECH_ROBOTICS_KEYWORDS = {
    "image classification", "speech recognition", "tts", "computer vision",
    "robotics", "gans",
}
NLP_IR_KEYWORDS = {"nlp", "retrieval", "ranking", "rag", "embeddings", "search"}

ARCHITECT_TITLES = {"architect", "tech lead", "technical lead", "engineering manager", "director", "vp", "head of"}

PREFERRED_LOCATIONS = {"pune", "noida"}
ACCEPTABLE_LOCATIONS = {"hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram", "new delhi"}

JD_MIN_YEARS, JD_MAX_YEARS =4,10 ## afual was arounf 6 9 but jd says it can be less punishing

IRRELEVANT_TITLE_KEYWORDS = {
    "hr", "human resources", "marketing", "graphic", "civil", 
    "mechanical", "accountant", "sales", "operations", "content writer", 
    "customer support", "project manager", "business analyst"
}

CORE_ML_KEYWORDS = {
    "ai", "machine learning", "ml", "data scientist", "nlp", 
    "search", "recommendation", "applied scientist", "software"
}


def text_blob_for_skills(candidate: dict) -> str:
    ##i put this in case no skills are given, tho not possible but its 100k data who knows
    return " ".join(s["name"].lower() for s in candidate.get("skills", []))


def experience_fit(years: float) -> float:
    if JD_MIN_YEARS <= years <= JD_MAX_YEARS:
        return 1.0
    distance = (JD_MIN_YEARS - years) if years < JD_MIN_YEARS else (years - JD_MAX_YEARS)
    return max(0.3, 1.0 - distance * 0.1)


def location_fit(profile: dict, signals: dict) -> float:
    country = (profile.get("country") or "").lower()
    loc = (profile.get("location") or "").lower()
    
    if country != "india":
        return 0.4  ## not in india and still applyng? i must give you penaltiy for that
        
    if signals.get("willing_to_relocate", False):
        return 1.0
        
    pref_mode = (signals.get("preferred_work_mode") or "").lower()
    if pref_mode in ["remote", "flexible"]:
        return 0.95
        
    if any(city in loc for city in PREFERRED_LOCATIONS):
        return 1.0
    if any(city in loc for city in ACCEPTABLE_LOCATIONS):
        return 0.8
    return 0.5  # elsewhere in India? no issue cause you aint gettin job anyway


def notice_period_fit(signals: dict) -> float:
    days = signals.get("notice_period_days")
    if days is None:
        return 0.6
    if days <= 30:
        return 1.0
    return max(0.3, 1.0 - (days - 30) / 120)  # okk fine, gn


def is_consulting_only(candidate: dict) -> bool:
    companies = {c["company"].lower() for c in candidate["career_history"]}
    return len(companies) > 0 and companies.issubset(CONSULTING_ONLY_FIRMS)


def is_vision_speech_only(candidate: dict) -> bool:
    skill_names = {s["name"].lower() for s in candidate.get("skills", [])}
    has_vision_speech = bool(skill_names & VISION_SPEECH_ROBOTICS_KEYWORDS)
    has_nlp_ir = bool(skill_names & NLP_IR_KEYWORDS)
    return has_vision_speech and not has_nlp_ir


def is_title_chaser(candidate: dict) -> bool:
   ##stictly terminate those suckers who are replacing company every 1.5 years
    history = candidate["career_history"]
    if len(history) < 3:
        return False
    avg_tenure_months = sum(j["duration_months"] for j in history) / len(history)
    return avg_tenure_months < 18

### this seems is not a good way to calc a lead architect's working/coding capacity, we must evaluate his github activity
## later , make it pendiing and use a mid quality rn
def is_architect_not_coding(candidate: dict) -> bool:
    
    current_job = next((j for j in candidate["career_history"] if j["is_current"]), None)
    if current_job is None:
        return False
    title_lower = current_job["title"].lower()
    is_architect_title = any(kw in title_lower for kw in ARCHITECT_TITLES)
    return is_architect_title and current_job["duration_months"] >= 18


def is_recent_ai_only(candidate: dict) -> bool:
   ## terminate those ai wrappers newbie like myself lmao
    ai_skill_months = [
        s["duration_months"] for s in candidate.get("skills", [])
        if any(kw in s["name"].lower() for kw in ALL_AI_KEYWORDS)
    ]
    if not ai_skill_months:
        return False
    years = candidate["profile"]["years_of_experience"]
    return max(ai_skill_months) < 12 and years > 3


def product_company_fraction(candidate: dict) -> float:
    ## priorotise work in product base company as jd says , give product guys more love
    total, product_months = 0, 0
    for j in candidate["career_history"]:
        total += j["duration_months"]
        if not any(kw in j["industry"].lower() for kw in SERVICE_INDUSTRY_KEYWORDS):
            product_months += j["duration_months"]
    return product_months / total if total else 0.5


def skill_keyword_score(candidate: dict):
    ##    Just checking the skills duh
    blob = text_blob_for_skills(candidate)
    required_hits = sum(1 for kw in REQUIRED_SKILL_KEYWORDS if kw in blob)
    nice_hits = sum(1 for kw in NICE_TO_HAVE_KEYWORDS if kw in blob)
    return required_hits, nice_hits


def days_since(date_str, today: date) -> int:
    try:
        d = date.fromisoformat(date_str)
        return (today - d).days
    except (TypeError, ValueError):
        return 9999


def behavioral_modifier(signals: dict, today: date) -> float:
    inactivity_days = days_since(signals.get("last_active_date"), today)
    activity_score = max(0.0, 1.0 - inactivity_days / 180)
    response_rate = signals.get("recruiter_response_rate") or 0.0
    github = signals.get("github_activity_score", -1)
    github_score = 0.5 if github == -1 else github / 100
    completeness = (signals.get("profile_completeness_score") or 0) / 100
    return (activity_score + response_rate + github_score + completeness) / 4


def build_row(candidate: dict, today: date) -> dict:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    required_hits, nice_hits = skill_keyword_score(candidate)

    title_lower = profile["current_title"].lower()
    is_irrelevant_title = any(kw in title_lower for kw in IRRELEVANT_TITLE_KEYWORDS)
    is_core_ml_title = any(kw in title_lower for kw in CORE_ML_KEYWORDS)

    return {
        "candidate_id": candidate["candidate_id"],
        "years_of_experience": profile["years_of_experience"],
        "current_title": profile["current_title"],
        "experience_fit": experience_fit(profile["years_of_experience"]),
        "location_fit": location_fit(profile, signals),
        "notice_period_fit": notice_period_fit(signals),
        "required_skill_hits": required_hits,
        "nice_to_have_hits": nice_hits,
        "product_company_fraction": round(product_company_fraction(candidate), 3),
        "is_consulting_only": is_consulting_only(candidate),
        "is_vision_speech_only": is_vision_speech_only(candidate),
        "is_title_chaser": is_title_chaser(candidate),
        "is_architect_not_coding": is_architect_not_coding(candidate),
        "is_recent_ai_only": is_recent_ai_only(candidate),
        "behavioral_modifier": round(behavioral_modifier(signals, today), 4),
        "recruiter_response_rate": signals.get("recruiter_response_rate") or 0.0,
        "is_irrelevant_title": is_irrelevant_title,
        "is_core_ml_title": is_core_ml_title,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--out", default="artifacts/features.csv")
    args = parser.parse_args()

    today = date.today()
    rows = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            rows.append(build_row(candidate, today))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()