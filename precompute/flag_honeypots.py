
import argparse
import json
from datetime import date, datetime

import pandas as pd

SUSPICIOUS_PROFICIENCIES = {"advanced", "expert"}


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        return None


def check_skill_inconsistencies(candidate: dict):
    ## claims to be expert but a looser, must be terminated ASAP!
    reasons = []
    for skill in candidate.get("skills", []):
        if skill["proficiency"] in SUSPICIOUS_PROFICIENCIES and skill["duration_months"] <= 1:
            reasons.append(f"skill '{skill['name']}' claims {skill['proficiency']} with ~0 months used")
    return reasons


def check_experience_math(candidate: dict):
    

    ## doesnt exceed more than 6 months than claime, must be the wind
    reasons = []
    total_months = sum(job["duration_months"] for job in candidate["career_history"])
    claimed_years = candidate["profile"]["years_of_experience"]
    claimed_months = claimed_years * 12

   
    if total_months > claimed_months + 6:  ## fi dude had some issue lets add 6 months exttra
        reasons.append(
            f"career_history totals {total_months} months but profile claims "
            f"only {claimed_years} years ({claimed_months:.0f} months)"
        )
    return reasons


def check_date_logic(candidate: dict):
    ##@# end date before start date, or duration__months not matching the actual date range then he a imposter
    reasons = []
    for job in candidate["career_history"]:
        start = parse_date(job["start_date"])
        end = parse_date(job["end_date"]) if job["end_date"] else date.today()
        if start and end and end < start:
            reasons.append(f"job at {job['company']} ends before it starts")
            continue
        if start and end:
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            if abs(actual_months - job["duration_months"]) > 3:
                reasons.append(
                    f"job at {job['company']} dates imply ~{actual_months} months "
                    f"but duration_months says {job['duration_months']}"
                )
    return reasons


def check_overlapping_jobs(candidate: dict):
    ## two jobs at the same time usually not legit or hes a alien so staight up disqualify that sucker, peak cinema, i'd do that all day
    reasons = []
    ranges = []
    for job in candidate["career_history"]:
        start = parse_date(job["start_date"])
        end = parse_date(job["end_date"]) if job["end_date"] else date.today()
        if start and end:
            ranges.append((start, end, job["company"]))

    ranges.sort()
    for i in range(len(ranges) - 1):
        if ranges[i][1] > ranges[i + 1][0]:
            reasons.append(f"dude has 2 or 2+ job runnning at a time, peak employement : {ranges[i][2]} and {ranges[i + 1][2]}")
    return reasons


def evaluate_candidate(candidate: dict) -> dict:
    reasons = (
        check_skill_inconsistencies(candidate)
        + check_experience_math(candidate)
        + check_date_logic(candidate)
        + check_overlapping_jobs(candidate)
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "is_honeypot": len(reasons) > 0,
        "honeypot_reasons": "; ".join(reasons),
    }

### finalising all in one 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--out", default="artifacts/honeypot_flags.csv")
    args = parser.parse_args()

    rows = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            rows.append(evaluate_candidate(candidate))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    flagged = int(df["is_honeypot"].sum())
    print(f"Wrote {len(df)} rows to {args.out} , flagged {flagged} as possible honeypots, instant termination as meat")


if __name__ == "__main__":
    main()