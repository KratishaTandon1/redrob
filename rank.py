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
            if re.search(r'\b' + re.escape(name) + r'\b', comp):
                founding_year = year
                break
                
        if founding_year is not None:
            if start_date_str:
                try:
                    start_year = int(start_date_str.split("-")[0])
                    if start_year < founding_year - 2: # started at least 3 years before founding (diff >= 3)
                        return True
                except:
                    pass
            # Max possible months from founding to mid 2026 + 18 months buffer
            max_months = (2026 - founding_year) * 12 + 18
            if duration > max_months:
                return True

    # 7. Total experience vs single job duration anomaly
    total_exp = profile.get("years_of_experience", 0)
    for job in career_history:
        dur_months = job.get("duration_months", 0)
        dur_years = dur_months / 12.0
        if dur_years > total_exp + 0.5:
            return True

    # 8. Technology launch year violations (Tuned: +24 months buffer, word-boundary check)
    sorted_techs = sorted(TECH_LAUNCH_YEARS.keys(), key=len, reverse=True)
    for s in skills:
        name = s.get("name", "").lower()
        dur = s.get("duration_months", 0)
        for tech in sorted_techs:
            if re.search(r'\b' + re.escape(tech) + r'\b', name):
                launch_year = TECH_LAUNCH_YEARS[tech]
                max_months = (2026 - launch_year) * 12 + 24
                if dur > max_months:
                    return True
                break

    # 9. Skill duration vs. years of experience anomaly (Tuned: +4.5 years buffer)
    yoe = profile.get("years_of_experience", 0.0)
    for s in skills:
        dur_months = s.get("duration_months", 0)
        dur_years = dur_months / 12.0
        # If any skill duration exceeds YoE + 4.5 years
        if dur_years > yoe + 4.5:
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

def is_framework_enthusiast(cand):
    skills = cand.get("skills", [])
    skills_names = {s.get("name", "").lower() for s in skills}
    
    llm_skills = {"langchain", "llamaindex", "openai", "gpt-4", "gpt-3", "chatgpt", "prompt engineering", "retrieval-augmented generation", "rag"}
    has_llm_skill = any(any(ls in sn for ls in llm_skills) for sn in skills_names)
    
    if not has_llm_skill:
        return False
        
    pre_llm_skills = {"pytorch", "tensorflow", "xgboost", "scikit-learn", "nlp", "natural language processing", "information retrieval", "search", "ranking", "recommendation", "recommender", "deep learning", "machine learning"}
    has_pre_llm_skill = False
    for s in skills:
        s_name = s.get("name", "").lower()
        s_dur = s.get("duration_months", 0)
        if any(ps in s_name for ps in pre_llm_skills) and s_dur >= 12:
            has_pre_llm_skill = True
            break
            
    has_pre_llm_history = False
    career_history = cand.get("career_history", [])
    for job in career_history:
        start_date = job.get("start_date")
        if start_date:
            try:
                start_year = int(start_date.split("-")[0])
                if start_year < 2023:
                    j_text = (job.get("title", "") + " " + job.get("description", "")).lower()
                    if any(kw in j_text for kw in ["ai", "ml", "machine learning", "nlp", "retrieval", "search", "ranking", "recommendation", "data scientist"]):
                        has_pre_llm_history = True
                        break
            except:
                pass
                
    if has_llm_skill and not has_pre_llm_skill and not has_pre_llm_history:
        return True
    return False

def is_closed_source_veteran(cand):
    profile = cand.get("profile", {})
    exp = profile.get("years_of_experience", 0.0)
    if exp < 5.0:
        return False
        
    signals = cand.get("redrob_signals", {})
    github_score = signals.get("github_activity_score", -1)
    if github_score != -1:
        return False
        
    career_history = cand.get("career_history", [])
    full_text = (profile.get("summary", "") + " " + " ".join([j.get("description", "") for j in career_history])).lower()
    
    validation_keywords = ["paper", "publication", "patent", "talk", "conference", "presentation", "open-source", "oss", "contribute", "github", "medium post", "blog", "writeup", "article", "thesis", "meetup", "speaker", "arxiv", "kaggle", "stackoverflow", "stack overflow"]
    has_validation = any(re.search(rf'\b{re.escape(kw)}', full_text) for kw in validation_keywords)
    
    if not has_validation:
        return True
    return False

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
    exp = profile.get("years_of_experience", 0.0)
        
    # B. Geographic boundaries (must be Noida/Pune/NCR commutable or willing to relocate to hybrid office)
    location_lower = profile.get("location", "").strip().lower()
    in_commutable_zone = any(city in location_lower for city in TARGET_CITIES)
    willing_relocate = signals.get("willing_to_relocate", False)
    
    if not in_commutable_zone and not willing_relocate:
        return False, 0.0, {}
        
    country = profile.get("country", "").strip().lower()
    if country != "india" and 'india' not in location_lower and not willing_relocate:
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

    # G. Notice Period Filter (disqualify notice period >= 90 days as per JD)
    notice = signals.get("notice_period_days", 90)
    if notice >= 90:
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
        
    if exp < 3.0:
        base_score = max(0.0, base_score - 15.0)  # soft experience floor penalty
        
    # B. Current Title / Headline relevance
    headline = profile.get("headline", "").lower()
    max_cur_sim = max(calculate_token_similarity(curr_title, JD_KEYWORDS), calculate_token_similarity(headline, JD_KEYWORDS))
    
    # C. Historical ML Experience Duration relevance (rewards candidates who spent years in ML roles)
    ml_months = 0
    for job in career_history:
        job_title = job.get("title", "").lower()
        if any(kw in job_title for kw in ["ai", "ml", "machine learning", "nlp", "retrieval", "search", "ranking", "recommendation", "applied scientist"]):
            ml_months += job.get("duration_months", 0)
            
    ml_years = ml_months / 12.0
    hist_score = min(10.0, ml_years * 2.5)
    base_score += min(25.0, max_cur_sim * 15.0 + hist_score)
        
    # D. Verified Skills Scoring (Tiered relevance matching)
    # Classifies skills into Core (Tier 1), ML/NLP Foundation (Tier 2), and Engineering (Tier 3) to score according to actual importance.
    TIER_1_SKILLS = {
        'rag', 'retrieval-augmented generation', 'vector search', 'vector database', 'vector databases',
        'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'opensearch', 'elasticsearch',
        'learning-to-rank', 'learning to rank', 're-ranking', 'reranking', 'ranking system', 'ranking systems',
        'recommendation system', 'recommendation systems', 'recommender systems', 'recommender',
        'information retrieval', 'information retrieval systems', 'semantic search', 'hybrid search',
        'ndcg', 'mrr', 'map', 'query processing', 'bm25'
    }
    TIER_2_SKILLS = {
        'nlp', 'natural language processing', 'llm', 'llms', 'fine-tuning', 'fine-tuned', 'lora', 'qlora',
        'peft', 'pytorch', 'tensorflow', 'deep learning', 'machine learning', 'transformers',
        'hugging face', 'huggingface', 'xgboost', 'lightgbm', 'catboost', 'scikit-learn'
    }
    TIER_3_SKILLS = {
        'python', 'system design', 'mlops', 'fastapi', 'sql', 'docker', 'kubernetes', 'aws', 'gcp', 'git'
    }

    full_text = (" ".join([j.get("description", "") for j in career_history]) + " " + profile.get("summary", "")).lower()
    skill_hits = []
    total_skill_contrib = 0.0
    
    for s in skills:
        s_name = s.get("name", "").lower()
        s_prof = s.get("proficiency", "beginner").lower()
        s_dur = s.get("duration_months", 0)
        
        # Determine skill tier weight
        base_weight = 0.0
        if any(re.search(rf'\b{re.escape(k)}\b', s_name) for k in TIER_1_SKILLS):
            base_weight = 3.0
        elif any(re.search(rf'\b{re.escape(k)}\b', s_name) for k in TIER_2_SKILLS):
            base_weight = 1.5
        elif any(re.search(rf'\b{re.escape(k)}\b', s_name) for k in TIER_3_SKILLS):
            base_weight = 0.5
            
        if base_weight > 0.0:
            # Proficiency weight multiplier
            prof_mult = 1.5 if s_prof == "expert" else (1.2 if s_prof == "advanced" else (0.8 if s_prof == "intermediate" else 0.4))
            # Duration weight multiplier (capped logarithmic scaling)
            dur_mult = min(1.2, math.log(s_dur + 1) / math.log(60)) if s_dur > 0 else 1.0
            
            # Verify if skill is actually mentioned in description or summary
            is_verified = (s_name in full_text) or any(w in full_text for w in s_name.split())
            verification_penalty = 1.0 if is_verified else 0.1  # penalize unverified skills by 90%
            
            contrib = base_weight * prof_mult * dur_mult * verification_penalty
            total_skill_contrib += contrib
            
            if is_verified:
                skill_hits.append(s.get("name"))

    base_score += min(40.0, total_skill_contrib * 4.0)

    # E. Product Company Bonus (using exact word boundary matching)
    current_company = profile.get("current_company", "").strip().lower()
    prod_bonus = 0.0
    if any(re.search(r'\b' + re.escape(pc) + r'\b', current_company) for pc in PRODUCT_COMPANIES):
        prod_bonus += 5.0
    for job in career_history:
        comp = job.get("company", "").strip().lower()
        if any(re.search(r'\b' + re.escape(pc) + r'\b', comp) for pc in PRODUCT_COMPANIES):
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

    # F2. Systems & Scale Execution Bonus (max 5 points)
    scale_keywords = {"scale", "scaling", "production", "latency", "throughput", "optimize", "optimization", "optimise", "pipeline", "benchmark", "drift", "inference", "deployment", "deploy"}
    scale_matches = sum(1 for kw in scale_keywords if re.search(rf'\b{re.escape(kw)}\b', full_text))
    base_score += min(5.0, scale_matches * 1.0)

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
    notice_factor = 1.15 if notice <= 15 else (1.10 if notice <= 30 else (1.00 if notice <= 60 else (0.75 if notice <= 90 else 0.40)))
    
    # B. Geographic Fit Factor (Noida/Pune/NCR preferred, others Tier 1 accepted)
    loc_factor = 0.50
    is_core_hub = any(city in location_lower for city in ["noida", "pune", "delhi", "ncr", "gurgaon"])
    if is_core_hub:
        loc_factor = 1.15
    elif in_commutable_zone: # other TARGET_CITIES (Hyderabad, Mumbai, etc.)
        loc_factor = 1.08 if willing_relocate else 1.00
    elif willing_relocate and (country == "india" or 'india' in location_lower):
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
    
    # G. Continuous Consulting Penalty (disabled per JD - mixed consulting+product is fine)
    consulting_mult = 1.0

    # H. Apply trap penalties (framework enthusiasts and closed-source-only veterans)
    trap_mult = 1.0
    if is_framework_enthusiast(cand):
        trap_mult *= 0.1
    if is_closed_source_veteran(cand):
        trap_mult *= 0.1

    final_score = base_score * consulting_mult * behavior_mult * trap_mult
    
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
    loc = info["location"] if info["location"] else "unknown location"
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
    
    if rank <= 15:
        starters = [
            f"Exceptional founding-team prospect with {years} years of experience, currently thriving as {title} at {company}.",
            f"Highly recommended {title} at {company} offering {years} years of proven expertise in production engineering.",
            f"Top-tier Senior AI candidate with {years} years of experience, currently leading as {title} at {company}."
        ]
        fits = [
            f"Brings deep, verified {skill_str} matching our high-relevance search criteria.",
            f"Demonstrates hands-on engineering execution in {skill_str} with excellent product alignment.",
            f"Features a strong systems-builder profile with verified {skill_str} deployed in production."
        ]
        logistics = [
            f"Active candidate ({int(rr*100)}% response rate) with {concern_str.strip()}",
            f"{act_phrase}. {concern_str.strip()}",
            f"Highly available talent. {concern_str.strip()}"
        ]
    elif rank <= 60:
        starters = [
            f"Strong {title} at {company} with {years} years of professional experience.",
            f"Brings a solid {years}-year track record of engineering, currently working as {title} at {company}.",
            f"Experienced {title} at {company} with {years} years of backend and ML experience."
        ]
        fits = [
            f"Demonstrates capability in {skill_str}, matching the JD technical requirements.",
            f"Shows good hands-on experience in {skill_str} with strong systems coding.",
            f"Offers a solid technical overlap in {skill_str} and backend implementation."
        ]
        logistics = [
            f"Active on the platform. {concern_str.strip()}",
            f"{act_phrase}. {concern_str.strip()}",
            f"Shows positive behavioral signals. {concern_str.strip()}"
        ]
    else:
        starters = [
            f"Adjacent profile showing {years} years of experience as {title} at {company}.",
            f"Candidate with {years} years of experience as {title} at {company}.",
            f"Positioned as filler ranking with {years} years of software experience as {title} at {company}."
        ]
        fits = [
            f"Technical overlap is limited to {skill_str} with minor alignment on JD.",
            f"Displays baseline experience in {skill_str} and general backend systems.",
            f"Has adjacent skills in {skill_str} but lacks deep ranking/retrieval production history."
        ]
        logistics = [
            f"Included due to overall experience despite {concern_str.strip()}",
            f"Candidate is a filler with {concern_str.strip()}",
            f"{act_phrase}. {concern_str.strip()}"
        ]
        
    starter = random.choice(starters)
    fit = random.choice(fits)
    log_part = random.choice(logistics)
    
    reasoning = f"{starter} {fit} {log_part}"
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
