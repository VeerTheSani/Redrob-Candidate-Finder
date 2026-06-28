import argparse
import json
import re
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

SKILL_SYNONYMS = {
    "sentence-transformers": ["sbert", "sentence transformer", "sentence transformers"],
    "vector database": ["vector db", "vectordb", "vector store", "ann index",
                         "approximate nearest neighbor", "hnsw", "ivf"],
    "retrieval": ["semantic search", "neural search", "dense retrieval",
                  "information retrieval", "hybrid search"],
    "ranking": ["re-ranking", "reranking", "re-rank", "rerank", "cross-encoder"],
    "embeddings": ["embedding", "vector embeddings"],
    "elasticsearch": ["elastic search"],
    "fine-tuning": ["finetuning", "fine tuning"],
    "learning-to-rank": ["learning to rank", "ltr"],
    "evaluation": ["offline eval", "ab test", "a/b test", "ab testing"],
    "retrieval-augmented": ["rag"],
}

VISION_SPEECH_ROBOTICS_KEYWORDS = {
    "image classification", "speech recognition", "tts", "computer vision",
    "robotics", "gans",
}
NLP_IR_KEYWORDS = {"nlp", "retrieval", "ranking", "rag", "embeddings", "search"}

ARCHITECT_TITLES = {"architect", "tech lead", "technical lead", "engineering manager", "director", "vp", "head of"}

PREFERRED_LOCATIONS = {"pune", "noida"}
ACCEPTABLE_LOCATIONS = {"hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram", "new delhi"}

JD_MIN_YEARS, JD_MAX_YEARS = 4, 10  ## afual was arounf 6 9 but jd says it can be less punishing

IRRELEVANT_TITLE_KEYWORDS = {
    "hr", "human resources", "marketing", "graphic", "civil",
    "mechanical", "accountant", "sales", "operations", "content writer",
    "customer support", "project manager", "business analyst",
}

CORE_ML_KEYWORDS = {
    "ai", "machine learning", "ml", "data scientist", "nlp",
    "search", "recommendation", "applied scientist",
}

ACADEMIC_KEYWORDS = {
    "university", "institute", "academia", "postdoc", "postdoctoral",
    "professor", "lecturer", "research fellow", "research scholar",
    "research assistant", "research intern",
}
PRODUCTION_KEYWORDS = {
    "production", "deployed", "deploy", "users", "scale", "shipped", "serving",
    "latency", "real-time", "customers", "throughput", "in prod", "live",
}
WRAPPER_FRAMEWORK_KEYWORDS = {
    "langchain", "llamaindex", "llama index", "autogpt", "auto-gpt", "crewai", "babyagi",
}

SKILL_MIN_MONTHS = 6
SKILL_MIN_ENDORSEMENTS = 5
SKILL_MIN_ASSESSMENT = 60


def build_matchers(keywords: set) -> dict:
    matchers = {}
    for kw in keywords:
        variants = [kw] + SKILL_SYNONYMS.get(kw, [])
        pattern = "|".join(r"\b" + re.escape(v) + r"\b" for v in variants)
        matchers[kw] = re.compile(pattern)
    return matchers


REQUIRED_MATCHERS = build_matchers(REQUIRED_SKILL_KEYWORDS)
NICE_MATCHERS = build_matchers(NICE_TO_HAVE_KEYWORDS)
AI_MATCHERS = build_matchers(ALL_AI_KEYWORDS)
CORE_ML_MATCHERS = build_matchers(CORE_ML_KEYWORDS)
IRRELEVANT_MATCHERS = build_matchers(IRRELEVANT_TITLE_KEYWORDS)


def matched_keywords(matchers: dict, text: str) -> set:
    return {kw for kw, pat in matchers.items() if pat.search(text)}


def any_match(matchers: dict, text: str) -> bool:
    return any(pat.search(text) for pat in matchers.values())


def full_text_blob(candidate: dict) -> str:
    profile = candidate["profile"]
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    parts += [job.get("description", "") for job in candidate["career_history"]]
    parts += [s["name"] for s in candidate.get("skills", [])]
    return " ".join(parts).lower()


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
        s.get("duration_months", 0) for s in candidate.get("skills", [])
        if any_match(AI_MATCHERS, s["name"].lower())
    ]
    if not ai_skill_months:
        return False
    years = candidate["profile"]["years_of_experience"]
    return max(ai_skill_months) < 12 and years > 3


def is_pure_research(candidate: dict, blob: str) -> bool:
    role_text = " ".join(
        f"{j['title']} {j['industry']} {j.get('company', '')}"
        for j in candidate["career_history"]
    ).lower()
    role_text += " " + candidate["profile"]["current_title"].lower()
    academic = any(kw in role_text for kw in ACADEMIC_KEYWORDS)
    has_production = any(kw in blob for kw in PRODUCTION_KEYWORDS)
    return academic and not has_production


def is_framework_enthusiast(blob: str, verified_required_hits: int, required_hits: int) -> bool:
    has_wrapper = any(kw in blob for kw in WRAPPER_FRAMEWORK_KEYWORDS)
    return has_wrapper and verified_required_hits == 0 and required_hits <= 1


def is_closed_source_no_validation(candidate: dict) -> bool:
    signals = candidate["redrob_signals"]
    years = candidate["profile"]["years_of_experience"]
    gh = signals.get("github_activity_score", -1)
    no_github = gh is None or gh <= 0
    no_certs = len(candidate.get("certifications", [])) == 0
    no_linkedin = not signals.get("linkedin_connected", False)
    return years >= 5 and no_github and no_certs and no_linkedin


def product_company_fraction(candidate: dict) -> float:
    ## priorotise work in product base company as jd says , give product guys more love
    total, product_months = 0, 0
    for j in candidate["career_history"]:
        total += j["duration_months"]
        if not any(kw in j["industry"].lower() for kw in SERVICE_INDUSTRY_KEYWORDS):
            product_months += j["duration_months"]
    return product_months / total if total else 0.5


def assessment_lookup(signals: dict) -> dict:
    scores = signals.get("skill_assessment_scores") or {}
    return {str(k).lower(): v for k, v in scores.items()}


def skill_is_evidenced(skill: dict, assess: dict) -> bool:
    if skill.get("duration_months", 0) >= SKILL_MIN_MONTHS:
        return True
    if skill.get("endorsements", 0) >= SKILL_MIN_ENDORSEMENTS:
        return True
    score = assess.get(skill["name"].lower())
    return score is not None and score >= SKILL_MIN_ASSESSMENT


def skill_evidence(candidate: dict):
    signals = candidate["redrob_signals"]
    assess = assessment_lookup(signals)

    verified_required, verified_nice = set(), set()
    claimed_ai, evidenced_ai = 0, 0
    relevant = []

    for s in candidate.get("skills", []):
        name_l = s["name"].lower()
        if not any_match(AI_MATCHERS, name_l):
            continue
        claimed_ai += 1
        evidenced = skill_is_evidenced(s, assess)
        strength = (
            assess.get(name_l, 0),
            s.get("endorsements", 0),
            s.get("duration_months", 0),
        )
        relevant.append((evidenced, strength, s["name"]))
        if evidenced:
            evidenced_ai += 1
            verified_required |= matched_keywords(REQUIRED_MATCHERS, name_l)
            verified_nice |= matched_keywords(NICE_MATCHERS, name_l)

    relevant.sort(key=lambda r: (r[0], r[1]), reverse=True)
    matched_names = ", ".join(name for _, _, name in relevant[:4])

    evidence_ratio = (evidenced_ai / claimed_ai) if claimed_ai else 0.5
    return {
        "verified_required_hits": len(verified_required),
        "verified_nice_hits": len(verified_nice),
        "skill_evidence_ratio": round(evidence_ratio, 3),
        "matched_skill_names": matched_names,
    }


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


def neutral_rate(value, neutral: float = 0.5) -> float:
    if value is None or value < 0:
        return neutral
    return float(value)


def primary_concern(feat: dict) -> str:
    if feat["is_pure_research"]:
        return "pure research/academic background with no production deployment"
    if feat["is_consulting_only"]:
        return "experience is entirely at consulting/services firms, not product teams"
    if feat["is_vision_speech_only"]:
        return "background is vision/speech, with no NLP or retrieval signal"
    if feat["is_framework_enthusiast"]:
        return "LLM-wrapper projects without evidenced retrieval/ranking depth"
    if feat["is_recent_ai_only"]:
        return "AI skills are recent additions to an otherwise non-AI career"
    if feat["is_title_chaser"]:
        return "short average tenure (<18 months) suggests frequent job changes"
    if feat["is_architect_not_coding"]:
        return "long-tenured architect/lead title — may be hands-off on coding"
    if feat["skill_evidence_ratio"] < 0.34:
        return "several claimed AI skills lack tenure, endorsements, or assessment backing"
    if feat["verified_required_hits"] == 0:
        return "core retrieval/ranking skills are not clearly evidenced"
    if feat["notice_period_days"] and feat["notice_period_days"] > 60:
        return f"{int(feat['notice_period_days'])}-day notice period"
    if feat["location_fit"] <= 0.5:
        return "located outside the Pune/Noida-relocatable target cities"
    if not (JD_MIN_YEARS <= feat["years_of_experience"] <= JD_MAX_YEARS):
        return "total experience sits outside the 5-9 year target band"
    return ""


def build_row(candidate: dict, today: date) -> dict:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]

    blob = full_text_blob(candidate)
    required_hits = len(matched_keywords(REQUIRED_MATCHERS, blob))
    nice_hits = len(matched_keywords(NICE_MATCHERS, blob))
    evidence = skill_evidence(candidate)

    title_lower = profile["current_title"].lower()
    is_irrelevant_title = any_match(IRRELEVANT_MATCHERS, title_lower)
    is_core_ml_title = any_match(CORE_ML_MATCHERS, title_lower)

    row = {
        "candidate_id": candidate["candidate_id"],
        "years_of_experience": profile["years_of_experience"],
        "current_title": profile["current_title"],
        "current_company": profile.get("current_company", ""),
        "experience_fit": experience_fit(profile["years_of_experience"]),
        "location_fit": location_fit(profile, signals),
        "notice_period_fit": notice_period_fit(signals),
        "notice_period_days": signals.get("notice_period_days"),
        "required_skill_hits": required_hits,
        "nice_to_have_hits": nice_hits,
        "verified_required_hits": evidence["verified_required_hits"],
        "verified_nice_hits": evidence["verified_nice_hits"],
        "skill_evidence_ratio": evidence["skill_evidence_ratio"],
        "matched_skill_names": evidence["matched_skill_names"],
        "product_company_fraction": round(product_company_fraction(candidate), 3),
        "is_consulting_only": is_consulting_only(candidate),
        "is_vision_speech_only": is_vision_speech_only(candidate),
        "is_title_chaser": is_title_chaser(candidate),
        "is_architect_not_coding": is_architect_not_coding(candidate),
        "is_recent_ai_only": is_recent_ai_only(candidate),
        "is_pure_research": is_pure_research(candidate, blob),
        "is_framework_enthusiast": is_framework_enthusiast(
            blob, evidence["verified_required_hits"], required_hits
        ),
        "is_closed_source_no_validation": is_closed_source_no_validation(candidate),
        "behavioral_modifier": round(behavioral_modifier(signals, today), 4),
        "recruiter_response_rate": signals.get("recruiter_response_rate") or 0.0,
        "interview_completion_rate": neutral_rate(signals.get("interview_completion_rate")),
        "offer_acceptance_rate": neutral_rate(signals.get("offer_acceptance_rate")),
        "open_to_work_flag": bool(signals.get("open_to_work_flag", False)),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d") or 0,
        "search_appearance_30d": signals.get("search_appearance_30d") or 0,
        "is_irrelevant_title": is_irrelevant_title,
        "is_core_ml_title": is_core_ml_title,
    }
    row["primary_concern"] = primary_concern(row)
    return row


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
