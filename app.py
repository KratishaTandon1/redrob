import streamlit as st
import json
import pandas as pd
import io
import csv
from pathlib import Path
from rank import detect_honeypot, evaluate_candidate, generate_reasoning

# Page configuration for premium aesthetic
st.set_page_config(
    page_title="Candidate Ranker Sandbox",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling and premium feel
st.markdown("""
<style>
    .main-title {
        font-family: 'Inter', 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🎯 Candidate Discovery & Ranking AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Hackathon Sandbox — Screener & Ranker for Founding Team Senior AI Engineers</p>', unsafe_allow_html=True)

# Find sample candidates file
possible_paths = [
    Path("sample_candidates.json"),
    Path("[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"),
    Path("[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"),
    Path("India_runs_data_and_ai_challenge/sample_candidates.json")
]

sample_file_path = None
for p in possible_paths:
    if p.exists():
        sample_file_path = p
        break

# Sidebar Configuration
st.sidebar.header("Data Source")
data_option = st.sidebar.radio(
    "Choose Input Data:",
    ("Use Sample Dataset", "Upload Custom JSON/JSONL")
)

uploaded_file = None
if data_option == "Upload Custom JSON/JSONL":
    uploaded_file = st.sidebar.file_uploader(
        "Upload candidates.jsonl or candidates.json",
        type=["jsonl", "json"],
        help="Make sure the file structure conforms to the candidate schema."
    )

# Candidate processor
def process_candidates(file_obj, is_jsonl=True):
    candidates = []
    scanned = 0
    honeypots = 0
    
    if is_jsonl:
        # Read JSON Lines
        for line in file_obj:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line.strip():
                continue
            try:
                cand = json.loads(line)
                scanned += 1
                if detect_honeypot(cand):
                    honeypots += 1
                    continue
                candidates.append(cand)
            except Exception as e:
                st.sidebar.error(f"Error parsing line: {e}")
    else:
        # Read standard JSON list
        try:
            if isinstance(file_obj, bytes):
                data = json.loads(file_obj.decode("utf-8"))
            elif hasattr(file_obj, "read"):
                data = json.load(file_obj)
            else:
                data = json.loads(file_obj)
                
            for cand in data:
                scanned += 1
                if detect_honeypot(cand):
                    honeypots += 1
                    continue
                candidates.append(cand)
        except Exception as e:
            st.sidebar.error(f"Error parsing JSON: {e}")
            
    return candidates, scanned, honeypots

candidates = []
scanned = 0
honeypots = 0

# Load Data
if data_option == "Use Sample Dataset" and sample_file_path:
    with open(sample_file_path, "r", encoding="utf-8") as f:
        candidates, scanned, honeypots = process_candidates(f, is_jsonl=False)
elif data_option == "Upload Custom JSON/JSONL" and uploaded_file:
    is_jsonl = uploaded_file.name.endswith(".jsonl")
    candidates, scanned, honeypots = process_candidates(uploaded_file, is_jsonl=is_jsonl)
else:
    if data_option == "Use Sample Dataset":
        st.warning("Sample dataset file (sample_candidates.json) was not found in the workspace. Please upload a file via the sidebar.")

# Run Evaluation and Ranking
if candidates:
    ranked_results = []
    for cand in candidates:
        is_valid, score, info = evaluate_candidate(cand)
        if is_valid:
            ranked_results.append((score, cand["candidate_id"], info))
            
    # Sort: score descending, then candidate_id ascending to break ties
    ranked_results.sort(key=lambda x: (-x[0], x[1]))
    top_100 = ranked_results[:100]
    
    # Create final dataset
    final_rows = []
    for idx, (score, cid, info) in enumerate(top_100):
        rank = idx + 1
        reason = generate_reasoning(info, rank)
        final_rows.append({
            "Rank": rank,
            "Candidate ID": cid,
            "Name": info["name"],
            "Score": round(score, 4),
            "Current Title": info["title"],
            "Current Company": info["company"],
            "Experience (YoE)": info["years"],
            "Location": info["location"],
            "Notice Period (Days)": info["notice"],
            "Reasoning": reason
        })
        
    df_top_100 = pd.DataFrame(final_rows)
    
    # Layout statistics
    st.subheader("📊 Ranking Execution Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{scanned}</div>
            <div class="metric-label">Total Scanned</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{honeypots}</div>
            <div class="metric-label">Honeypots Screened</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(ranked_results)}</div>
            <div class="metric-label">Passed Filters</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(top_100)}</div>
            <div class="metric-label">Final Ranked List</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Download section
    st.subheader("📥 Export Results")
    
    # Generate CSV in memory matching the validator spec
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for idx, (score, cid, info) in enumerate(top_100):
        rank = idx + 1
        reason = generate_reasoning(info, rank)
        writer.writerow([cid, rank, round(score, 4), reason])
        
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="Download submission.csv (Verified Spec)",
        data=csv_data,
        file_name="submission.csv",
        mime="text/csv",
        help="Downloads the CSV formatted exactly according to the hackathon validation specification."
    )
    
    st.markdown("---")
    
    # Interactive Table
    st.subheader("🔍 Top Ranked Candidates (Interactive Preview)")
    
    # Text search
    search_query = st.text_input("Filter candidates by Title, Company, or Location:", "")
    if search_query:
        df_filtered = df_top_100[
            df_top_100["Current Title"].str.contains(search_query, case=False, na=False) |
            df_top_100["Current Company"].str.contains(search_query, case=False, na=False) |
            df_top_100["Location"].str.contains(search_query, case=False, na=False)
        ]
    else:
        df_filtered = df_top_100
        
    st.dataframe(
        df_filtered,
        width="stretch",
        hide_index=True
    )
    
else:
    st.info("Please choose a candidate dataset to display the ranking statistics.")
