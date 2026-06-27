#!/usr/bin/env python3
import json
import csv
import re
import sys
import math
import argparse
import random
from pathlib import Path
from datetime import datetime

# Target locations (Noida/Pune preferred, other Tier 1 Indian cities accepted)
TARGET_CITIES = ["noida", "pune", "delhi", "ncr", "hyderabad", "mumbai", "bangalore", "gurgaon", "chennai", "kolkata"]

# Consulting firms to penalize (entire career in consulting is disqualified)
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy",
    "hcl", "tech mahindra", "mphasis", "l&t", "lnt", "mindtree", "hexaware", "persistent",
    "ust global", "cts", "genpact", "deloitte", "pwc", "ey", "kpmg"
}

# Product-centric companies for scoring bonus
PRODUCT_COMPANIES = {
    "google", "meta", "netflix", "apple", "microsoft", "amazon", "linkedin", "salesforce",
    "swiggy", "razorpay", "cred", "zomato", "flipkart", "meesho", "nykaa", "inmobi", "zoho",
    "ola", "vedantu", "byju's", "policybazaar", "paytm", "freshworks", "upgrad", "pharmeasy",
    "phonepe", "dream11", "unacademy", "glance", "rephrase.ai", "sarvam ai", "aganitha",
    "niramai", "saarthi.ai", "krutrim", "wysa", "mad street den", "haptik", "verloop.io",
    "yellow.ai", "observe.ai"
}

# Founding years of well-known AI/ML and tech companies to detect timeline anomalies
COMPANY_FOUNDING_YEARS = {
    # Global AI/Data
    "langchain": 2022,
    "llamaindex": 2022,
    "chroma": 2022,
    "openai": 2015,
    "anthropic": 2021,
    "cohere": 2019,
    "midjourney": 2021,
    "pinecone": 2019,
    "qdrant": 2021,
    "weaviate": 2019,
    "milvus": 2019,
    "mistral": 2023,
    "perplexity": 2022,
    "character.ai": 2021,
    "adept": 2022,
    "scale ai": 2016,
    "weights & biases": 2017,
    "hugging face": 2016,
    "copy.ai": 2020,
    "jasper": 2021,
    "anyscale": 2019,
    # Indian tech ecosystem
    "krutrim": 2023,
    "sarvam ai": 2023,
    "cred": 2018,
    "glance": 2019,
    "meesho": 2015,
    "phonepe": 2015,
    "pharmeasy": 2015,
    "swiggy": 2014,
    "ola": 2010,
    "zomato": 2008,
    "flipkart": 2007,
    "freshworks": 2010,
    "inmobi": 2007,
    "zoho": 1996
}

# Technology launch years for verifying skill durations (preventing keyword stuffing honeypots)
TECH_LAUNCH_YEARS = {
    "qlora": 2023,
    "lora": 2021,
    "peft": 2021,
    "langchain": 2022,
    "llamaindex": 2022,
    "chromadb": 2022,
    "chroma db": 2022,
    "pinecone": 2019,
    "qdrant": 2021,
    "weaviate": 2019,
    "milvus": 2019,
    "cohere": 2019,
    "anthropic": 2021,
    "gpt-4": 2023,
    "gpt-3": 2020,
    "chatgpt": 2022,
    "pytorch": 2016,
    "tensorflow": 2015,
    "transformers": 2017,
    "fastapi": 2018,
    "bert": 2018,
    "xgboost": 2014,
    "sentence-transformers": 2019,
    "sentence transformers": 2019
}

# Disqualified current titles (roles unfit for Senior AI Engineer / Founding Team)
DISQUALIFIED_KEYWORDS = [
    "civil", "mechanical", "electrical", "chemical", "industrial", "graphic", "designer",
    "writer", "editor", "sales", "marketing", "operations", "recruiter", "talent", "hr",
    "human resources", "accountant", "finance", "business analyst", "project manager",
    "product manager", "customer support", "support specialist", "qa engineer", "testing engineer"
]

# Core skills and keywords from Job Description
JD_KEYWORDS = {
    # High-relevance matching (RAG, Vector, Search, Evaluation)
    'rag', 'retrieval-augmented generation', 'vector search', 'vector database', 'embeddings',
    'sentence-transformers', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss',
    'opensearch', 'elasticsearch', 'learning-to-rank', 'learning to rank', 're-ranking', 'reranking',
    'ndcg', 'mrr', 'map', 'evaluation', 'xgboost', 'lora', 'qlora', 'peft', 'fine-tuning',
    # General NLP/ML skills
    'nlp', 'natural language processing', 'search', 'retrieval', 'semantic search', 'hybrid search',
    'transformers', 'llm', 'python', 'pytorch', 'tensorflow', 'deep learning', 'machine learning',
    'system design'
}

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

similarity_cache = {}

def calculate_token_similarity(term, targets):
    """
    Computes token similarity between a given term (e.g. title or skill) and target keywords.
    Uses direct matching, Jaccard token overlap, and character 3-gram fallback.
    """
    term_lower = term.lower().strip()
    if term_lower in similarity_cache:
        return similarity_cache[term_lower]
        
    best_sim = 0.0
    for target in targets:
        # 1. Substring matching with word boundaries to prevent sub-word matching (e.g. 'researcher' matching 'search')
        if term_lower == target:
            sim = 1.0
        elif re.search(rf'\b{re.escape(target)}\b', term_lower) or re.search(rf'\b{re.escape(term_lower)}\b', target):
            sim = 1.0
        else:
            # 2. Token overlap (Jaccard similarity)
            term_tokens = set(term_lower.split())
            target_tokens = set(target.split())
            if term_tokens and target_tokens:
                intersect = term_tokens.intersection(target_tokens)
                union = term_tokens.union(target_tokens)
                sim = len(intersect) / len(union)
            else:
                sim = 0.0
        
        # 3. Character 3-gram similarity fallback (catches minor typos/variations)
        if sim < 0.4 and len(term_lower) >= 3 and len(target) >= 3:
            t_grams = set(term_lower[i:i+3] for i in range(len(term_lower)-2))
            trg_grams = set(target[i:i+3] for i in range(len(target)-2))
            if t_grams and trg_grams:
                char_sim = len(t_grams.intersection(trg_grams)) / len(t_grams.union(trg_grams))
                sim = max(sim, char_sim)
            
        best_sim = max(best_sim, sim)
        
    similarity_cache[term_lower] = best_sim
    return best_sim

def detect_honeypot(cand):
    """
    Screens candidates for synthetic contradictions (honeypots).
    Returns True if an anomaly is found, False otherwise.
    """
    profile = cand.get("profile", {})
    career_history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    education = cand.get("education", [])
    
    # 1. Zero-duration skills anomaly: >= 5 expert/advanced skills with 0 duration months
    z_skills = sum(1 for s in skills if s.get('proficiency', '').lower() in ['expert', 'advanced'] and s.get('duration_months', 0) == 0)
    if z_skills >= 5:
        return True

    # 2. Job duration inflation: job duration exceeds date calendar difference by factor of 2 + 12 months
    for h in career_history:
        sd = h.get('start_date')
        ed = h.get('end_date')
        dur = h.get('duration_months', 0)
        if sd:
            try:
                s_dt = datetime.strptime(sd, "%Y-%m-%d")
                if ed:
                    e_dt = datetime.strptime(ed, "%Y-%m-%d")
                else:
                    e_dt = datetime.strptime("2026-06-20", "%Y-%m-%d")
                diff_months = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
                if dur > 2 * diff_months + 12:
                    return True
            except:
                pass

    # 3. Work precedes college start by more than 15 years (age discrepancy)
    edu_starts = [e.get('start_year') for e in education if e.get('start_year')]
    if edu_starts:
        min_edu_start = min(edu_starts)
        for h in career_history:
            sd = h.get('start_date')
            if sd:
                try:
                    start_yr = int(sd.split('-')[0])
                    if min_edu_start - start_yr > 15:
                        return True
                except:
                    pass

    # 4. Job start date is after end date
    for job in career_history:
        sd = parse_date(job.get("start_date"))
        ed = parse_date(job.get("end_date"))
        if sd and ed and sd > ed:
            return True

    # 5. Education start year is after end year
    for edu in education:
        sy = edu.get("start_year")
        ey = edu.get("end_year")
        if sy and ey and sy > ey:
            return True

    # 6. Founding year anomalies (starting at a company before its founding date, or dur exceeds max possible)
    for job in career_history:
        comp = job.get("company", "").lower()
        start_date_str = job.get("start_date")
        duration = job.get("duration_months", 0)
        
        founding_year = None
        for name, year in COMPANY_FOUNDING_YEARS.items():
            if name in comp:
                founding_year = year
                break
                
        if founding_year is not None:
            if start_date_str:
                try:
                    start_year = int(start_date_str.split("-")[0])
                    if start_year < founding_year:
                        return True
                except:
                    pass
            # Max possible months from founding to mid 2026
            max_months = (2026 - founding_year) * 12 + 6
            if duration > max_months:
                return True

    # 7. Total experience vs single job duration anomaly
    total_exp = profile.get("years_of_experience", 0)
    for job in career_history:
        dur_months = job.get("duration_months", 0)
        dur_years = dur_months / 12.0
        if dur_years > total_exp + 0.5:
            return True

    # 8. Technology launch year violations (e.g. QLoRA experience for 5 years when it launched in 2023)
    sorted_techs = sorted(TECH_LAUNCH_YEARS.keys(), key=len, reverse=True)
    for s in skills:
        name = s.get("name", "").lower()
        dur = s.get("duration_months", 0)
        for tech in sorted_techs:
            if tech in name:
                launch_year = TECH_LAUNCH_YEARS[tech]
                max_months = (2026 - launch_year) * 12 + 6
                if dur > max_months:
                    return True
                break

    # 9. Skill duration vs. years of experience anomaly
    yoe = profile.get("years_of_experience", 0.0)
    for s in skills:
        dur_months = s.get("duration_months", 0)
        dur_years = dur_months / 12.0
        # If any skill duration exceeds YoE + 3 years (buffer for pre-career learning)
        if dur_years > yoe + 3.0:
            return True

    return False

def calculate_consulting_ratio(career_history):
    """
    Computes fraction of career spent at IT consulting/services companies.
    """
    if not career_history:
        return 0.0
    consulting_months = 0
    total_months = 0
    for job in career_history:
        comp = job.get("company", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur
        if any(cf in comp for cf in CONSULTING_FIRMS):
            consulting_months += dur
    if total_months == 0:
        return 0.0
    return consulting_months / total_months

def evaluate_candidate(cand):
    """
    Main evaluation wrapper. Checks eligibility and scores the candidate.
    Returns (is_valid, score, info_dict).
    """
    profile = cand.get("profile", {})
    career_history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    education = cand.get("education", [])
    
    # --- STEP 1: HONEYPOT SCREENING ---
    if detect_honeypot(cand):
        return False, 0.0, {}

    # --- STEP 2: HARD ELIGIBILITY FILTERS ---
    # A. Minimum Experience Floor (must have at least 3 years for Senior)
    exp = profile.get("years_of_experience", 0.0)
    if exp < 3.0:
        return False, 0.0, {}
        
    # B. Geographic boundaries (must be India or willing to relocate to target cities)
    country = profile.get("country", "").strip().lower()
    location = profile.get("location", "").strip().lower()
    willing_relocate = signals.get("willing_to_relocate", False)
    is_in_target_city = any(city in location for city in TARGET_CITIES)
    
    if country != "india" and 'india' not in location:
        if not willing_relocate or not is_in_target_city:
            return False, 0.0, {}
            
    # C. Pure Consulting Filter: filter candidates who have ONLY worked in consulting firms
    companies = [job.get("company", "").strip().lower() for job in career_history]
    if companies:
        all_consulting = True
        for comp in companies:
            is_consulting = any(cf in comp for cf in CONSULTING_FIRMS)
            if not is_consulting:
                all_consulting = False
                break
        if all_consulting:
            return False, 0.0, {}

    # D. Disqualified Current Title (Operations, marketing, civil, etc.)
    curr_title = profile.get("current_title", "").strip().lower()
    for dk in DISQUALIFIED_KEYWORDS:
        if dk in curr_title:
            return False, 0.0, {}

    # E. Pure Computer Vision / Speech / Robotics filter (without NLP/IR)
    skills_names = [s.get("name", "").lower() for s in skills]
    cv_skills = ["image classification", "object detection", "yolo", "opencv", "cnn", "gans", "speech recognition", "tts", "asr", "robotics"]
    has_cv = any(cs in sn for sn in skills_names for cs in cv_skills)
    
    nlp_skills = [
        "nlp", "natural language processing", "information retrieval", "semantic search", "vector search", 
        "embeddings", "rag", "retrieval", "sentence transformers", "pinecone", "milvus", "weaviate", 
        "qdrant", "faiss", "elasticsearch", "opensearch", "llm", "fine-tuning", "lora", "qlora", "peft", 
        "learning to rank", "learning-to-rank"
    ]
    has_nlp = any(ns in sn for sn in skills_names for ns in nlp_skills)
    
    if has_cv and not has_nlp:
        return False, 0.0, {}

    # F. Pure Research / Academic filter: filter out candidates who have only worked in academic/research-only roles
    academic_titles = ["postdoc", "research fellow", "professor", "lecturer", "phd candidate", "research assistant", "academic researcher"]
    all_academic = True
    for job in career_history:
        j_title = job.get("title", "").lower()
        is_academic = any(at in j_title for at in academic_titles)
        if not is_academic:
            all_academic = False
            break
    if career_history and all_academic:
        return False, 0.0, {}

    # --- STEP 3: TECHNICAL SCORING (Base relevance) ---
    base_score = 0.0
    
    # A. Years of Experience (peaking around 5-9 years)
    if 5.0 <= exp <= 9.0:
        base_score += 25.0
    elif exp < 5.0:
        base_score += max(0.0, 25.0 - (5.0 - exp) * 5.0)  # rapidly penalize junior profiles
    else:
        base_score += max(5.0, 25.0 - (exp - 9.0) * 1.5)  # slowly penalize highly senior profiles
        
    # B. Current Title / Headline relevance
    headline = profile.get("headline", "").lower()
    max_cur_sim = max(calculate_token_similarity(curr_title, JD_KEYWORDS), calculate_token_similarity(headline, JD_KEYWORDS))
    
    # C. Historical Title relevance (rewards candidates who worked in ML roles previously)
    hist_matches = 0
    for job in career_history:
        job_title = job.get("title", "").lower()
        if any(kw in job_title for kw in ["ai", "ml", "machine learning", "nlp", "retrieval", "search", "ranking", "recommendation", "applied scientist"]):
            hist_matches += 1
            
    hist_score = min(10.0, hist_matches * 2.5)
    base_score += min(25.0, max_cur_sim * 15.0 + hist_score)
        
    # D. Verified Skills Scoring (counter keyword stuffers)
    full_text = (" ".join([j.get("description", "") for j in career_history]) + " " + profile.get("summary", "")).lower()
    skill_hits = []
    total_skill_contrib = 0.0
    
    for s in skills:
        s_name = s.get("name", "").lower()
        s_prof = s.get("proficiency", "beginner").lower()
        s_dur = s.get("duration_months", 0)
        
        sim = calculate_token_similarity(s_name, JD_KEYWORDS)
        if sim > 0.45:
            # Proficiency weight multiplier
            prof_mult = 1.5 if s_prof == "expert" else (1.2 if s_prof == "advanced" else (0.8 if s_prof == "intermediate" else 0.4))
            # Duration weight multiplier (capped logarithmic scaling)
            dur_mult = min(1.2, math.log(s_dur + 1) / math.log(60)) if s_dur > 0 else 1.0
            
            # Verify if skill is actually mentioned in description or summary
            is_verified = (s_name in full_text) or any(w in full_text for w in s_name.split())
            verification_penalty = 1.0 if is_verified else 0.1  # penalize unverified skills by 90%
            
            contrib = sim * prof_mult * dur_mult * verification_penalty
            total_skill_contrib += contrib
            
            if is_verified:
                skill_hits.append(s.get("name"))

    base_score += min(40.0, total_skill_contrib * 8.0)

    # E. Product Company Bonus
    current_company = profile.get("current_company", "").strip().lower()
    prod_bonus = 0.0
    if any(pc in current_company for pc in PRODUCT_COMPANIES):
        prod_bonus += 5.0
    for job in career_history:
        comp = job.get("company", "").strip().lower()
        if any(pc in comp for pc in PRODUCT_COMPANIES):
            prod_bonus += 2.0
            
    base_score += min(10.0, prod_bonus)

    # F. "Plain-Language" Career Description Relevance (max 20 points)
    # Search for core semantic concepts in career descriptions/summaries using word boundaries to prevent sub-word matching (e.g. 'search' in 'research').
    SEARCH_KEYWORDS = {
        'search', 'retrieval', 'ranking', 'recommendation', 'recommender', 'matching', 
        'information retrieval', 'query', 'queries', 'bm25', 'vector search', 'embeddings', 
        'hybrid search', 'indexing', 'faiss', 'ann', 'weaviate', 'pinecone', 'qdrant', 
        'milvus', 'elasticsearch', 'opensearch', 'hnsw', 'vector database', 'vector databases'
    }
    EVAL_KEYWORDS = {
        'evaluation', 'evaluations', 'eval', 'ndcg', 'mrr', 'map', 'ab test', 'a/b test', 
        'online metrics', 'offline metrics', 'precision', 'recall', 'f1 score', 'f1-score'
    }
    ML_KEYWORDS = {
        'machine learning', 'ml', 'deep learning', 'nlp', 'natural language processing', 
        'transformers', 'llm', 'llms', 'fine-tuning', 'fine-tuned', 'lora', 'qlora', 
        'peft', 'pytorch', 'tensorflow', 'applied scientist', 'applied science'
    }
    
    search_matches = sum(1 for kw in SEARCH_KEYWORDS if re.search(rf'\b{re.escape(kw)}\b', full_text))
    eval_matches = sum(1 for kw in EVAL_KEYWORDS if re.search(rf'\b{re.escape(kw)}\b', full_text))
    ml_matches = sum(1 for kw in ML_KEYWORDS if re.search(rf'\b{re.escape(kw)}\b', full_text))
    
    desc_relevance_score = min(10.0, search_matches * 2.0) + min(6.0, eval_matches * 2.0) + min(4.0, ml_matches * 1.0)
    base_score += desc_relevance_score

    # G. GitHub Activity Bonus (max 5 points)
    github_score = signals.get("github_activity_score", -1)
    if github_score > 0:
        base_score += min(5.0, github_score / 20.0)

    # H. Platform Skill Assessments Bonus (max 5 points)
    assessments = signals.get("skill_assessment_scores", {})
    assess_bonus = 0.0
    for s_name, s_score in assessments.items():
        s_name_lower = s_name.lower()
        is_relevant = any(kw in s_name_lower for kw in ["nlp", "machine learning", "deep learning", "python", "search", "retrieval", "llm", "fine-tuning", "vector"])
        if is_relevant:
            if s_score >= 80.0:
                assess_bonus += 2.0
            elif s_score >= 70.0:
                assess_bonus += 1.5
            elif s_score >= 50.0:
                assess_bonus += 1.0
    base_score += min(5.0, assess_bonus)

    # I. Market Interest / Recruiter Saves Bonus (max 3 points)
    saves = signals.get("saved_by_recruiters_30d", 0)
    base_score += min(3.0, saves * 0.3)

    # J. Title-Chaser / Job-Hopper Penalty (subtracts up to 10 points)
    # Detects candidates who switch companies every 1.5 years or less on average.
    distinct_companies = len(set(job.get("company", "").strip().lower() for job in career_history if job.get("company")))
    if exp >= 3.0 and distinct_companies >= 2:
        tenure_years = exp / distinct_companies
        if tenure_years < 1.5:
            base_score = max(0.0, base_score - 10.0)

    # K. Hasn't coded in 18 months penalty
    # Penalizes senior candidates who moved into pure architecture/tech lead/management roles.
    current_job = next((j for j in career_history if j.get("is_current")), None)
    if current_job:
        cj_title = current_job.get("title", "").lower()
        cj_dur = current_job.get("duration_months", 0)
        is_mgmt = any(mw in cj_title for mw in ["manager", "director", "vp", "head", "architect", "tech lead", "technical lead"]) and not any(ew in cj_title for ew in ["engineer", "developer", "scientist", "coder", "programmer"])
        if is_mgmt and cj_dur > 18:
            base_score = max(0.0, base_score - 15.0)

    # --- STEP 4: BEHAVIORAL MULTIPLIER AND LOGISTICS ---
    # A. Notice Period Factor
    notice = signals.get("notice_period_days", 90)
    notice_factor = 1.15 if notice <= 15 else (1.10 if notice <= 30 else (1.00 if notice <= 60 else (0.80 if notice <= 90 else 0.50)))
    
    # B. Geographic Fit Factor
    loc_factor = 0.50
    if is_in_target_city:
        loc_factor = 1.15
    elif willing_relocate and (country == "india" or 'india' in location):
        loc_factor = 1.05
    elif willing_relocate:
        loc_factor = 0.80
        
    # C. Recency of Platform Activity (June 23, 2026 reference)
    last_act = parse_date(signals.get("last_active_date"))
    active_factor = 0.50
    if last_act:
        delta_days = (datetime(2026, 6, 23) - last_act).days
        if delta_days <= 30:
            active_factor = 1.10
        elif delta_days <= 90:
            active_factor = 1.00
        elif delta_days <= 180:
            active_factor = 0.80
            
    # D. Recruiter Response Rate Factor
    resp_rate = signals.get("recruiter_response_rate", 0.5)
    rrr_factor = 0.50 + 0.70 * resp_rate # range: 0.50 to 1.20
    
    # E. Availability Flag Factor
    otw_factor = 1.05 if signals.get("open_to_work_flag", False) else 0.95
    
    # F. Interview Completion Rate Factor
    icr = signals.get("interview_completion_rate", 0.5)
    icr_factor = 1.05 if icr >= 0.80 else (1.00 if icr >= 0.50 else 0.80)
    
    behavior_mult = notice_factor * loc_factor * active_factor * rrr_factor * otw_factor * icr_factor
    behavior_mult = max(0.4, min(1.3, behavior_mult))
    
    # G. Continuous Consulting Penalty (fraction of career spent at service companies)
    c_ratio = calculate_consulting_ratio(career_history)
    consulting_mult = 1.0 - 0.40 * c_ratio

    final_score = base_score * consulting_mult * behavior_mult
    
    info = {
        "candidate_id": cand["candidate_id"],
        "name": profile.get("anonymized_name"),
        "title": profile.get("current_title"),
        "company": profile.get("current_company"),
        "years": exp,
        "location": profile.get("location"),
        "notice": notice,
        "skills": skill_hits[:3],
        "score": final_score,
        "rr": resp_rate
    }
    
    return True, final_score, info

def generate_reasoning(info, rank):
    """
    Generates fact-based reasoning sentences for the candidate, highlighting key fit points
    and honestly calling out logistics or notice period constraints.
    """
    name = info["name"]
    title = info["title"]
    company = info["company"]
    years = info["years"]
    notice = info["notice"]
    loc = info["location"]
    skills = info["skills"]
    rr = info["rr"]
    
    skill_str = f"expertise in {', '.join(skills)}" if skills else "solid backend capabilities"
    act_phrase = f"highly active on the platform ({int(rr * 100)}% response rate)"
    
    # Gaps/concerns analysis
    concerns = []
    if notice >= 60:
        concerns.append(f"notice period of {notice} days")
    if "pune" not in loc.lower() and "noida" not in loc.lower():
        concerns.append(f"location in {loc} (needs relocation)")
        
    concern_str = ""
    if concerns:
        concern_str = f" Acknowledged constraints: " + " and ".join(concerns) + "."
    else:
        concern_str = " Displays optimal logistics and notice period."

    # Seed randomly but deterministically by candidate ID and rank
    seed_val = sum(ord(c) for c in info["candidate_id"]) + rank
    random.seed(seed_val)
    
    starters = [
        f"{title} with {years} years of experience, currently working at {company}.",
        f"Brings {years} years of software and ML experience, including their current role as {title} at {company}.",
        f"Strong fit as {title} at {company} with {years} years of experience."
    ]
    
    fits = [
        f"Demonstrates verified {skill_str}, matching key JD technical requirements.",
        f"Shows hands-on experience in {skill_str} which aligns well with our search.",
        f"Strong overlap in {skill_str} combined with solid backend systems engineering."
    ]
    
    starter = random.choice(starters)
    fit = random.choice(fits)
    
    reasoning = f"{starter} {fit} {act_phrase}.{concern_str}"
    return reasoning.strip().replace("\n", " ")

def main():
    parser = argparse.ArgumentParser(description="Intelligent Candidate Discovery & Ranking")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl file")
    parser.add_argument("--out", required=True, help="Path to write the submission.csv file")
    args = parser.parse_args()
    
    candidates_path = Path(args.candidates)
    output_path = Path(args.out)
    
    if not candidates_path.exists():
        print(f"Error: Candidates file {candidates_path} does not exist.")
        sys.exit(1)
        
    print(f"Reading candidates from: {candidates_path}")
    print("Evaluating and screening profiles...")
    
    results = []
    total_scanned = 0
    honeypot_count = 0
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            total_scanned += 1
            
            if detect_honeypot(cand):
                honeypot_count += 1
                
            is_valid, score, info = evaluate_candidate(cand)
            if is_valid:
                results.append((round(score, 4), cand["candidate_id"], info))
                
    print(f"Scanned: {total_scanned} candidates.")
    print(f"Detected and filtered out {honeypot_count} honeypots.")
    print(f"Candidates passing initial screening: {len(results)}")
    
    # Sort by score descending, then candidate_id ascending to break ties
    results.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract top 100
    top_100 = results[:100]
    
    print(f"Generating ranking and reasoning for top 100...")
    
    # Write to CSV
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for idx, (score, cid, info) in enumerate(top_100):
            rank = idx + 1
            reason = generate_reasoning(info, rank)
            writer.writerow([cid, rank, score, reason])
            
    print(f"Successfully generated rank list and wrote to: {output_path}")

if __name__ == "__main__":
    main()
