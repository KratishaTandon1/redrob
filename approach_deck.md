# Candidate Discovery & Ranking AI System
## Finding Founding-Team Senior AI Engineers | Redrob Hackathon

---

## 1. The Challenge & Problem Statement
- **Recruitment Gap**: Keyword matching fails to distinguish between keyword-stuffer profiles and genuine hands-on builders.
- **The Challenge**: Rank 100,000 candidates to select the top 100 fits for a Senior AI Engineer role on a Series A founding team.
- **Honeypot Risk**: The dataset contains synthetic "honeypot" profiles with subtly impossible details. Traditional keyword-based search systems rank them highly, resulting in immediate disqualification (rules enforce < 10% honeypot rate in the top 100).
- **Performance Constraint**: Must run end-to-end on CPU only, with no network access, under 5 minutes.

---

## 2. The Trap: Honeypot Profiling
- **The Problem**: Keyword-matching scripts are easily fooled by synthetic profiles with stuffed skills. 
- **Baseline Diagnostic**: Evaluating the baseline ranker revealed a **48% honeypot rate** in the top 100 candidates, which would trigger instant disqualification.
- **Subtle Anomaly Classes Checked & Filtered**:
  1. **Zero-Duration Skills**: Experts with 0 months of experience.
  2. **Job Duration Inflation**: Claimed duration exceeds calendar date differences.
  3. **Age Discrepancies**: Working > 15 years before starting college.
  4. **Impossible Skill Launch Years**: Claiming 5+ years of experience with technologies that did not exist (e.g. QLoRA launched in 2023, LangChain/LlamaIndex in 2022).
  5. **Skill Duration > YoE Anomaly**: Skill durations exceeding the candidate's professional career length by > 3 years.

---

## 3. System Architecture
Our system implements a modular, high-speed filtering and scoring pipeline:

```
[Candidate JSONL Pool]
       │
       ▼
┌───────────────────────────────┐
│  Honeypot & Hard Filters      │  <─── Fast-pass timeline/geography checks
└──────────────┬────────────────┘
               │ (Passes ~11,000 valid candidates)
               ▼
┌───────────────────────────────┐
│  Technical Scoring            │  <─── Experience fit, title fit, company pedigree
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Plain-Language Matcher       │  <─── Evaluates career history descriptions
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  Behavioral Modifiers         │  <─── Notice period, active days, assessments, saves
└──────────────┬────────────────┘
               │
               ▼
   [Top 100 Ranked Shortlist]
```

---

## 4. Technical Scoring & Plain-Language Matcher
- **Experience Fit (0-25 pts)**: Peaks around 5-9 years of experience, penalizing overly junior profiles and slowly decaying for highly senior roles.
- **Title Relevance (0-25 pts)**: Scores current title/headline and historical ML roles (e.g. NLP Engineer, Applied Scientist).
- **Pedigree Bonus (0-10 pts)**: Rewards candidates who have shipped systems at product-centric tech firms.
- **Plain-Language Matching (0-20 pts)**:
  - Scans job descriptions for key engineering work keywords (e.g., search, retrieval, ranking, recommendation) and evaluation metrics (NDCG, MRR, MAP).
  - Rewards true builders who describe their achievements in plain language without stuffing buzzwords in their skills list.

---

## 5. Trust & Platform Signals
- **GitHub Activity (0-5 pts)**: Rewards open-source contributions and external validation.
- **Platform Skill Assessments (0-5 pts)**: Validates actual competence by rewarding scores >= 70% in relevant platform exams (e.g., NLP, ML, Python).
- **Recruiter saves (0-3 pts)**: Incorporates market demand signal.
- **Title-Chaser/Job-Hopper Penalty (-10 pts)**: Penalizes candidates with a history of shifting companies every 1.5 years or less.
- **Coded-in-18-Months Penalty (-15 pts)**: Penalizes senior engineers who transitioned to pure management.
- **Behavioral Multipliers (0.4x - 1.3x)**: Scales score based on notice period (prefers sub-30 days), geographic relocation readiness, active login dates, and recruiter response rates.

---

## 6. Performance & Results
- **Honeypot Rate in Top 100**: **0%** (Exactly 0 honeypot profiles are present in the final list).
- **Execution Speed**: **~2.5 minutes** to parse and rank 100,000 profiles.
- **Top 10 Highlights**: Surfaced candidates with exceptional product experience (e.g., Senior MLEs from Swiggy, Flipkart, Ola, CRED) with high platform response rates (70-90%) and optimal logistics fit.
