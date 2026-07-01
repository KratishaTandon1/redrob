---
title: Redrob Candidate Ranker
emoji: 🎯
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
app_port: 8501
---

# Candidate Discovery & Ranking AI System


This repository implements a production-grade candidate discovery and ranking system optimized to identify top-tier Senior AI Engineers for founding teams. The ranker prioritizes genuine engineering execution and logistics viability over keyword density, while systematically filtering synthetic honeypot profiles.

## Setup Instructions

### Prerequisites
- Python 3.11+
- No GPU required (CPU-only execution)
- No external network access or API keys needed during inference

### Dependencies
No external packages are required beyond the standard Python library, ensuring high portability and speed.

## Execution Command
To reproduce the candidate ranking list, run the following command from the root of the repository:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Performance Stats:
- **Execution Time**: ~2.5 minutes on CPU for the entire 100,000 candidate dataset.
- **Memory Footprint**: < 200 MB RAM.
- **Honeypot Disqualifications**: **0%** (exactly 0 honeypot profiles in the top 100).

## Methodology Summary

The ranking engine follows a modular, three-stage pipeline:

1. **Honeypot Screening (Chronological Integrity checks)**:
   - Filters out profile contradictions (e.g. education start after end, work before education start).
   - Validates technology launch years (e.g. filters out candidates claiming 60 months of QLoRA in 2026, when QLoRA launched in 2023).
   - Validates skill duration boundaries (e.g. filters out candidates with skill durations exceeding total experience by > 3 years).
2. **Technical Scoring**:
   - **Experience Fit**: Peaks at 5-9 years of experience.
   - **Role Relevance**: Matches current title and historical ML/NLP roles.
   - **Plain-Language Matching**: Searches career descriptions for search, retrieval, ranking, and evaluation terms (e.g. NDCG, MRR, MAP) to capture true ML practitioners who do not keyword stuff.
   - **Company Pedigree**: Rewards candidates with prior product-centric experience.
   - **Tenure Check**: Penalizes candidates with a history of job hopping (< 18 months per role).
3. **Behavioral Modifiers**:
   - Computes a multiplicative factor based on platform responsiveness, availability (`open_to_work`), notice period (favoring sub-30 days), and interview completion rate.
