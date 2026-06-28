
## 0. The 30-second version

Your job: out of **100,000** candidates, output the **top 100** for one job
description (JD), best first. You're graded mostly on the **top 10** (NDCG@10 =
50% of the score), so the entire game is *"get the front of the list right."*

The old code got a working list out the door (good!). But it had four habits
that quietly hurt the top of the list:

1. It matched skills with dumb text search (`"ai"` matched "**ai**rflow").
2. It believed every claimed skill (keyword stuffers slipped through).
3. It combined all signals by **multiplying** them, so one weak number could
   sink a great candidate.
4. Its "reasons" were a fill-in-the-blank template (which the judges penalize).

The new code fixes all four, then adds a second, smarter model (a
**cross-encoder**) that re-reads the top ~400 and re-sorts them. Everything still
runs offline, CPU-only, in ~40 seconds.

---

## 1. The mental model: a funnel (recall → precision)

Think of it like hiring in real life:

```
100,000 resumes
      │   cheap, fast filter  (embeddings + rules)   ← "who's roughly relevant?"
      ▼
    ~400 shortlist
      │   slow, careful read  (cross-encoder)         ← "who's actually best?"
      ▼
    top 100  → submission.csv
```

- **Recall stage** (cheap): never miss a good candidate, even if you let some
  mediocre ones through. This is the embeddings + feature rules over all 100k.
- **Precision stage** (expensive): from the survivors, get the *ordering* exactly
  right. This is the cross-encoder, and it only runs on 400 people because
  running it on 100k would blow the 5-minute limit.

This funnel is the single most important idea. Old code only had the first half.

---

## 2. The pipeline, file by file

| Step | File | What it does | When it runs |
|---|---|---|---|
| 1 | `precompute/embedded_candidates.py` | turns JD + each profile into a vector (list of numbers capturing *meaning*) | precompute (~33 min, once) |
| 2 | `precompute/build_features.py` | rule-based facts per candidate (experience, skills, location…) → `features.csv` | precompute (~1-2 min) |
| 3 | `precompute/flag_honeypots.py` | flags impossible/fake profiles → `honeypot_flags.csv` | precompute (seconds) |
| 4 | `precompute/setup_reranker.py` | downloads the cross-encoder once, saves it locally | precompute (once) |
| 5 | `precompute/build_rerank_texts.py` | a short text blurb per candidate for the cross-encoder | precompute (seconds) |
| 6 | `rank.py` | combines everything → shortlist → rerank → **top 100** | the graded step (~40s, offline) |

**Key rule:** precompute can be slow and use the internet. `rank.py` cannot — it
must be ≤5 min, CPU-only, **no network**. That's why models are downloaded in
precompute and saved to `artifacts/` so `rank.py` loads them locally.

---

## 3. Before vs After — the big picture

| Area | BEFORE | AFTER | Why it matters |
|---|---|---|---|
| Skill matching | `"kw" in text` substring | word-boundary regex + synonyms | `"ai"` no longer matches "airflow"; "sbert" now counts as sentence-transformers |
| What text is searched | skills list only | headline + summary + **job descriptions** + skills | real evidence lives in the descriptions |
| Trusting skills | every claimed skill counted | skills must be **corroborated** (tenure / endorsements / assessment) | keyword stuffers get pulled down |
| Combining signals | everything **multiplied** | additive core × bounded nudges × hard gates | one weak number can't sink a star |
| Title boost | `"software"` gave a 2× boost | removed; ML titles get a small +0.06 | every "Software Engineer" was wrongly boosted |
| Signals used | ~7 of 23 | + interview/offer/open-to-work/saves | more "is this person real & available?" |
| Reasoning | fixed template | specific, varied, honest, per-candidate | judges penalize templated text |
| Honeypots | 4 checks | + education-date sanity | fewer fakes in the list |
| Final ordering | embedding score only | **cross-encoder reranks** the top 400 | the big top-10 quality win |
| Dead code | `theoretical_max` division | removed | it never changed the ranking anyway |

---

## 4. Deep dive on the four changes that matter most

### 4.1 Keyword matching: from "substring" to "word with meaning"

**The bug.** Old code did `if keyword in text`. `in` checks if the letters
appear *anywhere*, even inside other words:

```python
"ai" in "airflow"      # True  ❌ (counts Airflow as an AI skill)
"map" in "roadmap"     # True  ❌ (MAP is a ranking metric, "roadmap" isn't)
"software" in title    # True  → gave EVERY "Software Engineer" a 2× boost ❌
```

**The fix.** Word-boundary regex (`\bai\b`) only matches `ai` as a whole word.
Plus a synonym map so different spellings of the same thing count:

```python
"sbert"          → counts as "sentence-transformers"
"vector db", "hnsw", "ann index" → count as "vector database"
"semantic search", "dense retrieval" → count as "retrieval"
```

And we now search the **summary + every job description**, not just the skills
list — because someone might *describe* building a retrieval system without
tagging "retrieval" as a skill.

> **Why it matters for the score:** garbage skill counts → wrong candidates near
> the top. Clean counts → the right people surface.

### 4.2 Anti-stuffing: don't believe every claimed skill

The dataset deliberately plants **keyword stuffers** — profiles that list
"Milvus, FAISS, Pinecone, RAG…" with no real backing, hoping a dumb matcher
ranks them high.

**The fix — corroboration.** A claimed skill only counts as *verified* if at
least one of these is true (see `skill_is_evidenced` in `build_features.py`):

- used it ≥ 6 months, **or**
- ≥ 5 endorsements, **or**
- scored ≥ 60 on the platform's skill assessment for it.

We then emit two numbers per candidate:
- `required_skill_hits` — raw count (recall).
- `verified_required_hits` — corroborated count (trust).

…plus `skill_evidence_ratio` = (evidenced AI skills) / (claimed AI skills). A
stuffer who claims 8 skills but backs up 1 gets ratio 0.125, which **shrinks
their skill score by up to 40%**. A genuine engineer with backed-up skills keeps
full credit.

> This is also the main defense against honeypots reaching the top 10 (a Stage-3
> disqualifier if >10% of your top 100 are honeypots).

### 4.3 The scoring rewrite (the heart of it)

**Old way — multiply everything:**

```python
score = cosine × experience_fit × location_fit × notice_fit × behavioral
```

Multiplication means **every factor has veto power**. Each number is between 0
and 1, so multiplying many of them drives the result toward 0, and a single low
one is fatal. Example: a perfect retrieval engineer with a 90-day notice period
(`notice_fit ≈ 0.5`) instantly loses *half* their score — even though the JD says
notice period is a minor concern. That's wrong.

**New way — three layers with different jobs:**

```
final = relevance_core   ×   soft_modifiers   ×   hard_gates
        (the driver)         (gentle nudges)      (the only vetoes)
```

1. **Relevance core** — an **addition**, not a multiplication:
   ```
   0.45·similarity + 0.35·verified_skills + 0.12·experience + 0.08·product_company (+0.06 if ML title)
   ```
   Because it's a sum, a zero in one part just removes that part's contribution
   — it can't annihilate the whole score. This is the "how good a match is this
   person, really?" number.

2. **Soft modifiers** — bounded multipliers that can only *nudge*:
   - location: ×[0.60–1.0]
   - notice period: ×[0.90–1.0] (minor, per the JD)
   - engagement (active? responsive? open to work?): ×[0.80–1.0]

   The worst a soft factor can do is shave a slice. It can't veto.

3. **Hard gates** — the *only* things allowed to crush a score:
   - honeypot → **0**
   - irrelevant title (HR, sales, civil…) → **0**
   - genuine mismatches get strong penalties: consulting-only ×0.15,
     vision/speech-only ×0.15, recent-AI-only ×0.30, job-hopper ×0.40,
     hands-off architect ×0.50.

> **The principle:** relevance decides the ranking; soft stuff fine-tunes;
> only true deal-breakers are allowed to kill a candidate. All the weights are
> named constants at the top of `rank.py` so you can tune them.

### 4.4 The cross-encoder reranker (Phase 2 — the new brain)

This is the biggest *quality* upgrade. To understand it you need two terms:

- **Bi-encoder** (your embeddings, step 1): turns the JD into a vector and each
  candidate into a vector *separately*, then compares them. Fast (you can do
  100k), but it never reads them side-by-side, so it's only *roughly* right.

- **Cross-encoder** (the new model): takes the JD **and** one candidate
  **together** as a single input and outputs "how well do these match?" Much more
  accurate, because it can reason about them jointly — but slow, so you can't run
  it on 100k.

**The trick:** use the fast bi-encoder + rules to cut 100k → **top 400**, then
run the accurate cross-encoder *only on those 400* and re-sort them. Best of
both: scale **and** precision. (`ms-marco-MiniLM-L-12-v2`, runs on CPU in ~35s
for 400 candidates.)

**Blending.** We don't throw away the Phase-1 score (it knows about location,
honeypots, disqualifiers — things the cross-encoder can't see). We combine them:

```python
final = (1 - ALPHA)·phase1  +  ALPHA·cross_encoder      # ALPHA = 0.5
```

- `ALPHA = 0` → ignore the cross-encoder (pure Phase-1).
- `ALPHA = 1` → trust only the cross-encoder.
- `ALPHA = 0.5` → equal mix (current default).

Honeypots and irrelevant titles are **excluded from the 400 shortlist entirely**,
so the reranker can never resurrect a trap.

> **Proof it's doing something sane:** `compare_rankings.py` showed overlap@10 =
> 10/10 (it agrees on the elite — reassuring) but only 86/100 overall — it does
> its real work reshuffling the uncertain middle of the list.

---

## 5. Honeypots, briefly

`flag_honeypots.py` flags **impossible** profiles (not just weak ones): a skill
claimed at "expert" with ~0 months of use, career months that exceed claimed
experience, dates that end before they start, two full-time jobs overlapping,
graduation before enrollment. Anything flagged is forced to score **0** in
`rank.py`.

> Lesson learned during the build: I first added a "skill used longer than your
> career" check — but people legitimately use skills since college, so it
> false-flagged ~10% of normal profiles. I removed it. **A honeypot check must
> catch the *impossible*, never the merely unusual.**

---

## 6. The reasoning column (why it's not a template anymore)

Stage-4 judges read 10 random reasonings and **penalize** templated, identical,
or hallucinated text. They reward specific facts + honest concerns.

Old: 4 fixed tone buckets ("Top-tier candidate. …"). New: built from the
candidate's **real** fields — actual matched skills, employer, exact years — with
a concern surfaced when one exists:

> *"Senior ML Engineer at Zomato, 7.2 yrs — strong, evidenced retrieval/ranking
> depth (Weaviate, Learning to Rank, Embeddings). Largely product-company
> experience, matching the JD's preference."*

It only ever names skills the candidate actually has (no hallucination), and all
100 came out distinct.

---

## 7. The numbers that prove the change

| Check | Result |
|---|---|
| Validator | `Submission is valid.` |
| Honeypots in top 100 | **0** |
| Disqualifiers in top 100 | **0** |
| Distinct reasonings | **100 / 100** |
| Top-100 median experience | 5.9 yrs (JD wants 5-9 ✅) |
| Top-100 median verified skills | 4 |
| `rank.py` runtime | ~40s (limit: 5 min) |
| Reranker effect | overlap@10 10/10, @100 86/100 |

---

## 8. Tweak cheat-sheet (all in `rank.py` top, unless noted)

| Want to… | Change |
|---|---|
| Trust the cross-encoder more / less | `ALPHA` (0.5) |
| Bigger / smaller shortlist | `SHORTLIST_K` (400) |
| Weight semantic match vs skills | `W_SIM`, `W_SKILL`, `W_EXP`, `W_PROD` |
| Punish job-hoppers harder | `PENALTY_TITLE_CHASER` (0.40) |
| Make location matter more | `LOCATION_FLOOR` (0.60 → lower = harsher) |
| Push out over-experienced (>9 yrs) candidates | `OVEREXP_START/FULL/FLOOR` in `rank.py` (start 10, full 18, floor 0.70) |
| Change the JD text being matched | `JD_FOCUS` in `embedded_candidates.py` (re-embed the JD vector afterwards) |
| Change which skills count | the keyword sets / `SKILL_SYNONYMS` in `build_features.py` |
| Change what "verified" means | `SKILL_MIN_MONTHS/ENDORSEMENTS/ASSESSMENT` in `build_features.py` |
| See the effect of any change | re-run `rank.py` then `compare_rankings.py` |

---

## 8b. Later refinements (JD fidelity pass)

After re-reading the **full** JD (`hackathon_ps/job_description.txt` — the `data/`
copy we'd been embedding was truncated and missing ~half the signal):

- **Experience is a soft range, not a cutoff.** The JD literally says *"5-9 is a
  range, not a requirement… we'll seriously consider candidates outside the band
  if other signals are strong."* So the over-experience penalty is now a **tiny
  nudge** (`OVEREXP_FLOOR = 0.92`, ≤8% max) — enough to favor the 6-8 sweet spot,
  never enough to exclude a strong senior.
- **We now embed the full JD**, with the "things we do NOT want" / culture /
  hackathon-note sections stripped out (a bi-encoder would wrongly match the words
  it says to avoid). See `strip_negative_sections` + `JD_FOCUS` in
  `embedded_candidates.py`.
- **Three JD disqualifiers added** (from the "do NOT want" list): `is_pure_research`
  (academic, no production) and `is_framework_enthusiast` (LangChain-wrapper
  hobbyist, no real depth) are **active**; `is_closed_source_no_validation` is
  **wired but disabled** (`PENALTY_CLOSED_SOURCE = 1.0`) because the only available
  proxy flags ~21% of people and mostly means "private", not "unvalidated".

> **Lesson:** always rank against the *authoritative* JD, and always check a new
> rule's flag-rate before trusting it — a disqualifier that fires on 21% of
> candidates is a liability, not a feature.

## 9. Mini-glossary

- **Embedding / vector:** a list of numbers that captures the *meaning* of text,
  so similar meanings → similar numbers.
- **Cosine similarity:** how aligned two vectors are (1 = identical direction).
- **Bi-encoder:** embeds two texts separately, then compares. Fast, approximate.
- **Cross-encoder:** reads two texts together, scores the pair. Slow, accurate.
- **NDCG@10:** the grade for how good your *top 10 ordering* is (50% of score).
- **Honeypot:** a deliberately impossible fake profile planted to catch lazy
  systems.
- **Precompute vs ranking step:** precompute = slow setup (allowed network);
  ranking = the timed, offline `rank.py` that makes the CSV.
```
