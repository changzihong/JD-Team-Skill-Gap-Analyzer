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
# Custom CSS and JavaScript
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #f8f8f8;
    color: #000;
    font-family: 'Poppins', sans-serif;
}
.navbar {
    background-color: #000;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 10;
    padding: 15px 0;
    text-align: center;
    color: white;
    font-size: 22px;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.footer {
    background-color: #000;
    color: white;
    text-align: center;
    position: fixed;
    bottom: 0;
    width: 100%;
    padding: 10px;
    font-size: 14px;
}
.main-container {
    padding-top: 100px;
    padding-bottom: 60px;
}
.upload-section, .dashboard-section {
    background-color: white;
    border-radius: 16px;
    padding: 25px;
    margin: 20px auto;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    transition: transform 0.3s ease-in-out;
}
.upload-section:hover, .dashboard-section:hover {
    transform: scale(1.01);
}
button[kind="primary"] {
    background-color: #000 !important;
    color: white !important;
    border-radius: 8px;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('.upload-section, .dashboard-section');
    sections.forEach((section, idx) => {
        section.style.opacity = 0;
        setTimeout(() => {
            section.style.transition = 'opacity 1.2s';
            section.style.opacity = 1;
        }, 300 * idx);
    });
});
</script>
""", unsafe_allow_html=True)

# ---------------------------
# Navbar
# ---------------------------
st.markdown('<div class="navbar">📊 AI Skills Radar (JD Team Skill Gap Analyzer)</div>', unsafe_allow_html=True)

# ---------------------------
# Skill Extract & Analyzer Logic
# ---------------------------
COMMON_SKILLS = [
    'python','sql','excel','data analysis','communication','project management',
    'leadership','product management','machine learning','nlp','aws','gcp','react',
    'java','c#','sales','negotiation','recruiting','interviewing','coaching','training'
]

def extract_skills_from_text(text):
    text_lower = text.lower()
    found = [s for s in COMMON_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]
    if not found:
        tokens = re.findall(r"[a-zA-Z]{4,}", text_lower)
        found = tokens[:8]
    return sorted(set(found))

def aggregate_team_skills(df):
    counts = {}
    if 'skills' in df.columns:
        for row in df['skills'].dropna():
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
    if N == 0: return None
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    stats = values + values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, linewidth=2, color='black')
    ax.fill(angles, stats, alpha=0.25, color='gray')
    ax.set_thetagrids(np.degrees(angles[:-1]), skills)
    ax.set_ylim(0,100)
    ax.set_title(title, size=14)
    return fig

def call_gemini(prompt, system_prompt="You are an AI HR analyst.", max_tokens=400):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}]
    }
    params = {"key": GEMINI_API_KEY}
    response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
    data = response.json()
    try:
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error parsing response: {e}\n\nRaw: {data}"

# ---------------------------
# Main App Container
# ---------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Upload Section
st.subheader("📁 Upload JD & Team Profiles")
jd_file = st.file_uploader("Upload Job Description (TXT/PDF)", type=['txt','pdf'])
team_file = st.file_uploader("Upload Team Profiles (CSV/Excel)", type=['csv','xlsx'])
jd_text = ""

if jd_file:
    if jd_file.type == "text/plain":
        jd_text = jd_file.getvalue().decode("utf-8")
    elif jd_file.type == "application/pdf":
        reader = PdfReader(jd_file)
        jd_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    st.text_area("Job Description Text", jd_text, height=200)

if team_file:
    team_df = pd.read_csv(team_file) if team_file.name.endswith(".csv") else pd.read_excel(team_file)
    st.dataframe(team_df.head())

# Dashboard
if st.button("🔍 Analyze Skills"):
    if jd_text and team_file:
        jd_skills = extract_skills_from_text(jd_text)
        team_skill_counts = aggregate_team_skills(team_df)
        score, matched, missing_detail = compute_skill_match(jd_skills, team_skill_counts)
        st.metric(label="Team-JD Match Score", value=f"{score}%")

        # Radar Visualization
        skills_to_plot = jd_skills[:8]
        team_size = max(1, sum(team_skill_counts.values()))
        values = [int(100 * team_skill_counts.get(s,0)/team_size) for s in skills_to_plot]
        fig = radar_chart(skills_to_plot, values)
        if fig: st.pyplot(fig)

        # AI Summary (Gemini)
        with st.spinner("🤖 Generating AI Summary..."):
            analysis_prompt = f"""
Job Description Skills: {', '.join(jd_skills)}
Matched Skills: {', '.join(matched)}
Missing Skills: {[m['skill'] for m in missing_detail]}

Summarize this in point form (around 200 words) focusing on:
- Key strengths
- Major skill gaps
- Suggested training directions
"""
            summary = call_gemini(analysis_prompt, system_prompt="You are an AI HR Analyst generating concise workforce insights.")
            st.markdown("### 📊 AI-Generated Skill Gap Insights")
            st.write(summary)
    else:
        st.warning("Please upload both JD and team profile files before analysis.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
<div class="footer">
    © 2025 AI Skills Radar | Designed by HR AI Application Builder
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
