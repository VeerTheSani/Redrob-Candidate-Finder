
import numpy as np

JD_MIN_YEARS, JD_MAX_YEARS = 5, 9  # the band named in the JD


def _clean(v):
    return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip()


def _strengths(row):
    """Pros, most->least compelling, from real profile facts."""
    pros = []
    highlight = _clean(row.get("evidence_highlight"))
    if highlight:
        pros.append(highlight)  # e.g. "built retrieval/ranking systems at Zomato and Google"
    vreq = int(row.get("verified_required_hits", 0))
    if vreq:
        avg = float(row.get("verified_required_strength", 0)) / vreq
        pros.append(f"{vreq} verified core skill{'s' if vreq != 1 else ''} (avg evidence {avg:.2f}/1)")
    elif _clean(row.get("matched_skill_names")):
        pros.append("relevant skills: " + _clean(row.get("matched_skill_names")))
    if float(row.get("product_company_fraction", 0)) >= 0.7:
        pros.append("mostly product-company experience")
    if JD_MIN_YEARS <= float(row.get("years_of_experience", 0)) <= JD_MAX_YEARS:
        pros.append(f"within the {JD_MIN_YEARS}-{JD_MAX_YEARS} experience band")
    if float(row.get("recruiter_response_rate", 0)) >= 0.7:
        pros.append("highly responsive to recruiters")
    return pros


def _concerns(row):
    """Cons, most->least severe, from the SAME signals that moved the score."""
    c = []
    if row.get("is_pure_research"):        c.append("pure-research background with no production signal")
    if row.get("is_consulting_only"):      c.append("career entirely at services firms, not product teams")
    if row.get("is_vision_speech_only"):   c.append("vision/speech background with no NLP/retrieval")
    if row.get("is_framework_enthusiast"): c.append("LLM-wrapper projects without evidenced retrieval depth")
    if row.get("is_recent_ai_only"):       c.append("AI is a recent add-on to an otherwise non-AI career")
    if row.get("is_title_chaser"):         c.append("short median tenure suggests frequent job changes")
    if row.get("is_architect_not_coding"): c.append("long-tenured lead title, likely hands-off on code")
    if int(row.get("verified_required_hits", 0)) == 0:
        c.append("core retrieval/ranking skills not clearly evidenced")
    if float(row.get("skill_evidence_ratio", 1)) < 0.5:
        c.append("several claimed skills thin on tenure/endorsements")
    if float(row.get("location_fit", 1)) <= 0.5:
        c.append("outside the Pune/Noida target and not relocating")
    nd = row.get("notice_period_days")
    if nd is not None and not (isinstance(nd, float) and np.isnan(nd)) and float(nd) > 60:
        c.append(f"{int(float(nd))}-day notice period")
    yrs = float(row.get("years_of_experience", 7))
    if float(row.get("experience_fit", 1)) < 0.999:
        side = "below" if yrs < JD_MIN_YEARS else "above"
        c.append(f"{yrs:g} yrs, {side} the {JD_MIN_YEARS}-{JD_MAX_YEARS} target band")
    if float(row.get("recruiter_response_rate", 1)) < 0.4:
        c.append("low recruiter responsiveness — availability risk")
    if float(row.get("behavioral_modifier", 1)) < 0.4:
        c.append("limited recent platform activity")
    return c


def generate_reasoning(row):
    """Organised, honest justification: Title -> Strengths -> Concerns. Only uses
    fields present on the candidate."""
    title = _clean(row.get("current_title")) or "Engineer"
    company = _clean(row.get("current_company"))
    yrs = float(row.get("years_of_experience", 0))
    where = f" at {company}" if company else ""

    pros = _strengths(row)
    cons = _concerns(row)

    out = [f"{title}{where}, {yrs:g} yrs."]
    if pros:
        out.append("Strengths: " + "; ".join(pros[:3]) + ".")
    out.append(("Concerns: " + "; ".join(cons[:2]) + ".") if cons else "Concerns: none major.")
    return " ".join(out)
