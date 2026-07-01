import json
from pathlib import Path
from rank import detect_honeypot, evaluate_candidate, TARGET_CITIES, CONSULTING_FIRMS, DISQUALIFIED_KEYWORDS

sample_path = Path("sample_candidates.json")

with open(sample_path, "r", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"Total candidates in sample: {len(candidates)}")

reasons = {
    "honeypot": 0,
    "location": 0,
    "consulting": 0,
    "disqualified_title": 0,
    "cv_no_nlp": 0,
    "pure_academic": 0,
    "notice_period": 0,
    "passed": 0
}

for cand in candidates:
    profile = cand.get("profile", {})
    career_history = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    
    # 1. Honeypot
    if detect_honeypot(cand):
        reasons["honeypot"] += 1
        continue

    # 2. Location
    location_lower = profile.get("location", "").strip().lower()
    in_commutable_zone = any(city in location_lower for city in TARGET_CITIES)
    willing_relocate = signals.get("willing_to_relocate", False)
    country = profile.get("country", "").strip().lower()
    
    loc_fail = False
    if not in_commutable_zone and not willing_relocate:
        loc_fail = True
    elif country != "india" and 'india' not in location_lower and not willing_relocate:
        loc_fail = True
        
    if loc_fail:
        reasons["location"] += 1
        continue
        
    # 3. Consulting
    companies = [job.get("company", "").strip().lower() for job in career_history]
    if companies:
        all_consulting = True
        for comp in companies:
            is_consulting = any(cf in comp for cf in CONSULTING_FIRMS)
            if not is_consulting:
                all_consulting = False
                break
        if all_consulting:
            reasons["consulting"] += 1
            continue

    # 4. Disqualified Title
    curr_title = profile.get("current_title", "").strip().lower()
    title_fail = False
    for dk in DISQUALIFIED_KEYWORDS:
        if dk in curr_title:
            title_fail = True
            break
    if title_fail:
        reasons["disqualified_title"] += 1
        continue

    # 5. Pure CV / Speech / Robotics without NLP
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
        reasons["cv_no_nlp"] += 1
        continue

    # 6. Academic
    academic_titles = ["postdoc", "research fellow", "professor", "lecturer", "phd candidate", "research assistant", "academic researcher"]
    all_academic = True
    for job in career_history:
        j_title = job.get("title", "").lower()
        is_academic = any(at in j_title for at in academic_titles)
        if not is_academic:
            all_academic = False
            break
    if career_history and all_academic:
        reasons["pure_academic"] += 1
        continue

    # 7. Notice Period
    notice = signals.get("notice_period_days", 90)
    if notice >= 90:
        reasons["notice_period"] += 1
        continue

    reasons["passed"] += 1

print("\nDisqualification Breakdown for sample candidates:")
for reason, count in reasons.items():
    print(f"- {reason}: {count}")
