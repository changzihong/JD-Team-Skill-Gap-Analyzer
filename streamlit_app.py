# app.py
import os
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import StringIO

# Install via: pip install streamlit-community-navigation-bar
from streamlit_community_navigation_bar import st_navbar

st.set_page_config(page_title="AI Skills Radar", layout="wide")

# ----------------------
# Backend: read API key from environment or secrets
# ----------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # hide key in backend
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

if not GEMINI_API_KEY:
    st.error("⚠️ API key not configured. Set GEMINI_API_KEY as env variable.")

# ----------------------
# Navigation bar
# ----------------------
page = st_navbar(
    ["Home", "Upload & Profiles", "Dashboard", "Training Suggestions", "About"],
    selected="Home"
)

# ----------------------
# Helper functions
# ----------------------
COMMON_SKILLS = [
    'python','sql','excel','data analysis','communication','project management',
    'leadership','product management','machine learning','nlp','aws','gcp','react',
    'java','c#','sales','negotiation','recruiting','interviewing','coaching','training'
]

def extract_skills_from_text(text, skill_list=COMMON_SKILLS):
    text_lower = text.lower()
    found = set()
    for s in skill_list:
        if re.search(r'\b' + re.escape(s) + r'\b', text_lower):
            found.add(s)
    if not found:
        tokens = re.findall(r"[a-zA-Z]{4,}", text_lower)
        found = set(tokens[:8])
    return sorted(found)

def aggregate_team_skills(df_profiles):
    counts = {}
    if 'skills' in df_profiles.columns:
        for row in df_profiles['skills'].dropna():
            for s in [x.strip().lower() for x in re.split('[,;|/\\\\n]', str(row)) if x.strip()]:
                counts[s] = counts.get(s, 0) + 1
    else:
        for _, r in df_profiles.iterrows():
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
    missing_detail = [{ 'skill': s, 'team_count': team_skill_counts.get(s, 0) } for s in sorted(missing)]
    return score, sorted(matched), missing_detail

def radar_chart(skills, values, title='Team Skills Radar'):
    labels = skills
    stats = values
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    stats = list(stats)
    stats += stats[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,100)
    ax.set_title(title)
    return fig

def call_llm(prompt, max_tokens=512, temperature=0.0):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }
    payload = {
        "prompt": [{"content": prompt, "type": "text"}],
        "temperature": temperature,
        "candidate_count": 1,
        "max_output_tokens": max_tokens
    }
    resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]

def suggest_training_for_skills(skills):
    suggestions = []
    for s in skills:
        suggestions.append({
            'skill': s,
            'recommended_course': f'Intro to {s.title()} (external course)',
            'internal_mentor': f'Senior {s.title()} Specialist'
        })
    return suggestions

# ----------------------
# Page logic
# ----------------------
if page == "Home":
    st.title("Welcome to AI Skills Radar")
    st.write("Use the navigation bar above to move between pages.")

elif page == "Upload & Profiles":
    st.header("Upload Job Description & Team Profiles")
    jd_file = st.file_uploader('Upload Job Description (txt or pdf)', type=['txt','pdf'])
    jd_text = st.text_area('Or paste JD text here', height=160)
    team_file = st.file_uploader('Upload Team Profiles (CSV/Excel)', type=['csv','xlsx','xls'])
    if st.button("Proceed to Dashboard"):
        if jd_file is not None:
            if jd_file.type == 'text/plain':
                jd_text = jd_file.getvalue().decode('utf-8')
            else:
                st.warning("PDF extraction not implemented in this demo.")
        if team_file is not None:
            try:
                if team_file.name.endswith('.csv'):
                    team_df = pd.read_csv(team_file)
                else:
                    team_df = pd.read_excel(team_file)
                st.session_state['team_df'] = team_df
                st.session_state['jd_text'] = jd_text
                st.success("Uploaded successfully! Now go to Dashboard.")
            except Exception as e:
                st.error("Failed to load team file: " + str(e))

elif page == "Dashboard":
    st.header("Analysis Dashboard")
    jd_text = st.session_state.get('jd_text', None)
    team_df = st.session_state.get('team_df', None)
    if not jd_text or team_df is None:
        st.warning("Please upload JD & team profiles under the “Upload & Profiles” page.")
    else:
        jd_skills = extract_skills_from_text(jd_text)
        team_skill_counts = aggregate_team_skills(team_df)
        score, matched, missing_detail = compute_skill_match(jd_skills, team_skill_counts)
        st.metric(label='Match Score', value=f'{score}%')
        st.write("Matched skills:", matched)
        st.write("Missing skills details:", missing_detail)
        viz_skills = list(jd_skills)[:8]
        team_size = max(1, sum(team_skill_counts.values()))
        viz_values = [int(100 * team_skill_counts.get(s,0) / team_size) for s in viz_skills]
        fig = radar_chart(viz_skills, viz_values)
        st.pyplot(fig)
        st.session_state['missing_detail'] = missing_detail

elif page == "Training Suggestions":
    st.header("Suggested Trainings & Mentoring")
    missing_detail = st.session_state.get('missing_detail', [])
    if not missing_detail:
        st.info("Go to Dashboard first to compute gaps.")
    else:
        df_suggestions = pd.DataFrame(suggest_training_for_skills([m['skill'] for m in missing_detail]))
        st.dataframe(df_suggestions)

elif page == "About":
    st.header("About this App")
    st.write("Built by your AI application builder. Version 0.1. Uses Gemini API for skill analysis.")