# app.py
import os
import re
import json
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO
from PyPDF2 import PdfReader

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="AI Skills Radar", layout="wide")

# ---------------------------
# Load Gemini API Key
# ---------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API key not found. Please set it in Streamlit Secrets or environment variables.")

# ---------------------------
# Custom CSS and JavaScript (includes fixed top navbar)
# ---------------------------
st.markdown("""
<style>
/* Basic layout */
body {
    background-color: #f8f8f8;
    color: #000;
    font-family: 'Poppins', sans-serif;
    margin: 0;
}

/* Fixed top navbar */
.top-navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 9999;
    background-color: #000;
    color: white;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.nav-left { display:flex; align-items:center; gap:14px; }
.nav-title { font-weight:700; font-size:18px; color:white; }
.nav-links { display:flex; gap:12px; align-items:center; }
.nav-link {
    color: #ddd;
    text-decoration: none;
    padding: 6px 10px;
    border-radius: 8px;
    transition: background 0.18s, color 0.18s, transform 0.12s;
    cursor: pointer;
}
.nav-link:hover {
    background: rgba(255,255,255,0.06);
    color: white;
    transform: translateY(-2px);
}

/* Main container to avoid navbar overlap */
.main-container {
    padding-top: 86px;
    padding-bottom: 64px;
    max-width: 1200px;
    margin: 0 auto;
}

/* Cards */
.upload-section, .dashboard-section {
    background-color: white;
    border-radius: 12px;
    padding: 22px;
    margin: 18px 0;
    box-shadow: 0 6px 24px rgba(0,0,0,0.06);
    transition: transform 0.28s ease-in-out, box-shadow 0.28s;
}
.upload-section:hover, .dashboard-section:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 36px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton>button {
    background-color: #000 !important;
    color: #fff !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
}
.stButton>button:hover {
    transform: scale(1.03);
}

/* Footer fixed */
.app-footer {
    background-color: #000;
    color: #fff;
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 10px 12px;
    text-align: center;
    font-size: 13px;
    border-top: 1px solid rgba(255,255,255,0.04);
}

/* Smooth fade-in for sections */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(6px);}
    to {opacity: 1; transform: translateY(0);}
}
.upload-section, .dashboard-section {
    animation: fadeIn 0.9s ease both;
}

/* Small screens */
@media (max-width: 700px) {
    .nav-links { display: none; }
    .main-container { padding-left: 12px; padding-right: 12px; }
}
</style>

<script>
function scrollToId(id) {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({behavior: 'smooth', block:'start'});
}
</script>
""", unsafe_allow_html=True)

# ---------------------------
# Render fixed top navbar (keeps always visible)
# ---------------------------
st.markdown("""
<div class="top-navbar">
  <div class="nav-left">
    <div class="nav-title">📊 AI Skills Radar</div>
  </div>
  <div class="nav-links">
    <a class="nav-link" href="javascript:scrollToId('upload')">Upload</a>
    <a class="nav-link" href="javascript:scrollToId('dashboard')">Dashboard</a>
    <a class="nav-link" href="javascript:scrollToId('account')">Account</a>
    <a class="nav-link" href="javascript:scrollToId('about')">About</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Skill Extract & Analyzer Logic (unchanged)
# ---------------------------
COMMON_SKILLS = [
    'python','sql','excel','data analysis','communication','project management',
    'leadership','product management','machine learning','nlp','aws','gcp','react',
    'java','c#','sales','negotiation','recruiting','interviewing','coaching','training'
]

def extract_skills_from_text(text):
    text_lower = (text or "").lower()
    found = [s for s in COMMON_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]
    if not found:
        tokens = re.findall(r"[a-zA-Z]{4,}", text_lower)
        found = tokens[:8]
    return sorted(set(found))

def aggregate_team_skills(df):
    counts = {}
    if df is None:
        return counts
    if 'skills' in [c.lower() for c in df.columns]:
        col = next(c for c in df.columns if c.lower() == 'skills')
        for row in df[col].dropna():
            for s in [x.strip().lower() for x in re.split('[,;|/\\\\n]', str(row)) if x.strip()]:
                counts[s] = counts.get(s, 0) + 1
    else:
        for _, r in df.iterrows():
            combined = ' '.join([str(x) for x in r.values if pd.notna(x)])
            for s in extract_skills_from_text(combined):
                counts[s] = counts.get(s, 0) + 1
    return counts

def compute_skill_match(jd_skills, team_skill_counts):
    jd_set = set(jd_skills)
    team_set = set(team_skill_counts.keys())
    matched = jd_set & team_set
    missing = jd_set - team_set
    score = int(100 * len(matched) / max(1, len(jd_set)))
    missing_detail = [{'skill': s, 'team_count': team_skill_counts.get(s, 0)} for s in sorted(missing)]
    return score, sorted(matched), missing_detail

def radar_chart(skills, values, title='Team Skills Radar'):
    N = len(skills)
    if N == 0:
        return None
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals = list(values)
    vals += vals[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, linewidth=2, color='black')
    ax.fill(angles, vals, alpha=0.25, color='gray')
    ax.set_thetagrids(np.degrees(angles[:-1]), skills)
    ax.set_ylim(0,100)
    ax.set_title(title)
    return fig

def call_gemini(prompt, system_prompt="You are an AI HR analyst generating concise workforce insights.", max_tokens=400):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}]}
    params = {"key": GEMINI_API_KEY}
    resp = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=30)
    data = resp.json()
    try:
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error parsing response: {e}\n\nRaw: {data}"

# ---------------------------
# Main content container (below fixed navbar)
# ---------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Upload Section (unchanged)
st.markdown('<div id="upload" class="upload-section">', unsafe_allow_html=True)
st.subheader("📁 Upload JD & Team Profiles")
jd_file = st.file_uploader("Upload Job Description (TXT/PDF)", type=['txt','pdf'])
team_file = st.file_uploader("Upload Team Profiles (CSV/Excel)", type=['csv','xlsx','xls'])
jd_text = ""

if jd_file:
    if jd_file.type == "text/plain":
        jd_text = jd_file.getvalue().decode("utf-8")
    elif jd_file.type == "application/pdf":
        reader = PdfReader(jd_file)
        jd_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    st.text_area("Job Description Text (editable)", jd_text, height=200)

if team_file:
    try:
        team_df = pd.read_csv(team_file) if team_file.name.endswith(".csv") else pd.read_excel(team_file)
        st.markdown("Team profiles preview:")
        st.dataframe(team_df.head())
    except Exception as e:
        st.error(f"Failed to read team file: {e}")
st.markdown('</div>', unsafe_allow_html=True)

# Dashboard / Analysis Section (unchanged)
st.markdown('<div id="dashboard" class="dashboard-section">', unsafe_allow_html=True)
st.subheader("🔍 Analyze & Dashboard")

if st.button("🔎 Analyze Skills"):
    if not jd_text and 'jd_text' in st.session_state:
        jd_text = st.session_state['jd_text']
    if not jd_text:
        st.warning("Please provide a Job Description (upload or paste).")
    elif not team_file and 'team_df' not in st.session_state:
        st.warning("Please upload team profiles (CSV/Excel).")
    else:
        # prefer session_state if available
        if 'team_df' in st.session_state:
            team_df = st.session_state['team_df']
        else:
            # ensure team_df variable exists from upload processing
            try:
                team_df = pd.read_csv(team_file) if team_file.name.endswith(".csv") else pd.read_excel(team_file)
            except Exception:
                team_df = None

        # compute
        jd_skills = extract_skills_from_text(jd_text)
        team_skill_counts = aggregate_team_skills(team_df)
        score, matched, missing_detail = compute_skill_match(jd_skills, team_skill_counts)

        st.metric(label="Team → JD Match Score", value=f"{score}%")
        st.write("**Matched skills:**", ", ".join(matched) if matched else "None")
        st.write("**Missing / Low coverage skills:**")
        st.table(missing_detail if missing_detail else [{"skill":"-", "team_count":0}])

        # radar
        viz_skills = list(jd_skills)[:8]
        team_size = max(1, sum(team_skill_counts.values()))
        viz_values = [int(100 * team_skill_counts.get(s,0) / team_size) for s in viz_skills]
        fig = radar_chart(viz_skills, viz_values)
        if fig:
            st.pyplot(fig)

        # call Gemini for structured summary (200-word bullet points)
        prompt = f"""
Job Description skills: {', '.join(jd_skills)}
Matched skills: {', '.join(matched)}
Missing skills: {', '.join([m['skill'] for m in missing_detail])}

Provide a concise summary in bullet/point form (~200 words) focusing on:
- overall readiness
- major skill gaps
- recommended training/hiring actions
- 6-month outlook
"""
        with st.spinner("Generating AI summary..."):
            summary = call_gemini(prompt, system_prompt="You are an expert HR AI. Output in bullet points, ~200 words.")
        st.markdown("### 🤖 AI Summary (bullet points, ~200 words)")
        st.write(summary)
st.markdown('</div>', unsafe_allow_html=True)

# Account section (Login/Signup combined) — keep existing behavior but present in page
st.markdown('<div id="account" class="upload-section">', unsafe_allow_html=True)
st.subheader("🔐 Account (Login / Signup)")

col1, col2 = st.columns(2)
with col1:
    st.write("Login (demo)")
    user = st.text_input("Username", key="login_user")
    pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Login", key="login_btn"):
        if user == "admin" and pwd == "1234":
            st.success("Logged in as admin (demo).")
            st.session_state['user'] = user
        else:
            st.error("Invalid credentials (demo).")
with col2:
    st.write("Signup (demo)")
    new_user = st.text_input("New username", key="signup_user")
    new_pwd = st.text_input("New password", type="password", key="signup_pwd")
    if st.button("Create account", key="signup_btn"):
        if new_user and new_pwd:
            st.success(f"Account '{new_user}' created (demo).")
            st.session_state['user'] = new_user
        else:
            st.error("Please provide username and password.")
st.markdown('</div>', unsafe_allow_html=True)

# About section
st.markdown('<div id="about" class="upload-section">', unsafe_allow_html=True)
st.subheader("About")
st.write("AI Skills Radar helps HR & L&D teams align job requirements with current team capabilities and plan upskilling.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer (fixed)
# ---------------------------
st.markdown("""
<div class="app-footer">
    © 2025 AI Skills Radar | Designed by HR AI Application Builder
</div>
""", unsafe_allow_html=True)
