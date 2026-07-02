import json
from pathlib import Path
from rank import detect_honeypot, evaluate_candidate

sample_path = Path("sample_candidates.json")
with open(sample_path, "r", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"Total candidates in sample: {len(candidates)}")

reasons = {
    "honeypot": 0,
    "all_consulting_or_title_or_cv_or_academic": 0,  # hard disqualifiers still inside evaluate_candidate
    "passed_hard_filters": 0,
    "location_penalized": 0,
    "notice_penalized": 0,
    "gaming_penalty_flagged": 0,
}

for cand in candidates:
    if detect_honeypot(cand):
        reasons["honeypot"] += 1
        continue

    is_valid, score, info = evaluate_candidate(cand)
    if not is_valid:
        reasons["all_consulting_or_title_or_cv_or_academic"] += 1
        continue

    reasons["passed_hard_filters"] += 1
    if not info.get("in_commutable_zone") and not info.get("willing_relocate"):
        reasons["location_penalized"] += 1
    if info.get("notice") >= 90:
        reasons["notice_penalized"] += 1
    if info.get("has_gaming_penalty"):
        reasons["gaming_penalty_flagged"] += 1

print("\nBreakdown:")
for reason, count in reasons.items():
    print(f"- {reason}: {count}")
