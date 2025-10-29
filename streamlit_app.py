# app.py
import os
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import StringIO
from PyPDF2 import PdfReader
from streamlit_community_navigation_bar import st_navbar

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="AI Skills Radar", layout="wide")

# ----------------------
# Custom CSS & JS (Light Purple Theme)
# ----------------------
st.markdown("""
    <style>
        body {
            background-color: #f3f0ff;
        }
        .stApp {
            background-color: #f8f5ff;
        }
        h1, h2, h3, h4 {
            color: #4b0082;
        }
        .stButton>button {
            background-color: #6f42c1 !important;
            color: white !important;
            border-radius: 10px;
            font-weight: 600;
            border: none;
            transition: 0.2s ease-in-out;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background-color: #5a32a3 !important;
        }
        .stMetric {
            background-color: #ede7f6;
            border-radius: 10px;
            padding: 10px;
        }
    </style>

    <script>
    document.addEventListener('DOMContentLoaded', function() {
      document.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('mouseenter', () => btn.style.transform = 'scale(1.05)');
        btn.addEventListener('mouseleave', () => btn.style.transform = 'scale(1)');
      });
    });
    </script>
""", unsafe_allow_html=True)

# ----------------------
# Backend: Gemini API Setup
# ----------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set via environment variable or Streamlit secrets
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

if not GEMINI_API_KEY:
    st.warning("⚠️ Gemini API key not configured. Please set GEMINI_API_KEY as an environment variable or in Streamlit secrets.")

# ----------------------
# Navigation Bar
# ----------------------
page = st_navbar(
    ["Home", "Upload & Profiles", "Dashboard", "Training Suggestions", "About"],
    selected="Home"
)

# ----------------------
# Helper Functions
# ----------------------
COMMON_SKILLS = [
    'python', 'sql', 'excel', 'data analysis', 'communication', 'project management',
    'leadership', 'product management', 'machine learning', 'nlp', 'aws', 'gcp', 'react',
    'java', 'c#', 'sales', 'negotiation', 'recruiting', 'interviewing', 'coaching', 'training'
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
    missing_detail = [{'skill': s, 'team_count': team_skill_counts.get(s, 0)} for s in sorted(missing)]
    return score, sorted(matched), missing_detail

def radar_chart(skills, values, title='Team Skills Radar'):
    labels = skills
    stats = values
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    stats = list(stats)
    stats += stats[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, linewidth=2, color='#6f42c1')
    ax.fill(angles, stats, alpha=0.25, color='#6f42c1')
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    return fig

def call_llm(prompt, max_tokens=512, temperature=0.2):
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"❌ LLM request failed: {e}"

def suggest_training_for_skills(skills):
    suggestions = []
    for s in skills:
        suggestions.append({
            'skill': s,
            'recommended_course': f'Intro to {s.title()} (External course)',
            'internal_mentor': f'Senior {s.title()} Specialist'
        })
    return suggestions

# ----------------------
# Page Logic
# ----------------------
if page == "Home":
    st.title("💼 Welcome to AI Skills Radar")
    st.markdown("""
        This demo helps HR or L&D teams analyze **team skills** versus **job descriptions**.
        - Upload your job description & team profiles
        - View AI-powered skill match scores
        - Identify skill gaps & training suggestions  
        
        Use the navigation bar above to explore features.
    """)

elif page == "Upload & Profiles":
    st.header("📂 Upload Job Description & Team Profiles")

    jd_file = st.file_uploader('Upload Job Description (TXT or PDF)', type=['txt', 'pdf'])
    jd_text = st.text_area('Or paste JD text here', height=180)

    team_file = st.file_uploader('Upload Team Profiles (CSV or Excel)', type=['csv', 'xlsx', 'xls'])

    if st.button("📊 Proceed to Dashboard"):
        if jd_file is not None:
            if jd_file.type == 'text/plain':
                jd_text = jd_file.getvalue().decode('utf-8')
            elif jd_file.type == 'application/pdf':
                reader = PdfReader(jd_file)
                jd_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        if team_file is not None:
            try:
                if team_file.name.endswith('.csv'):
                    team_df = pd.read_csv(team_file)
                else:
                    team_df = pd.read_excel(team_file)
                st.session_state['team_df'] = team_df
                st.session_state['jd_text'] = jd_text
                st.success("✅ Uploaded successfully! Now go to **Dashboard**.")
            except Exception as e:
                st.error(f"Failed to load team file: {e}")

elif page == "Dashboard":
    st.header("📈 Analysis Dashboard")

    jd_text = st.session_state.get('jd_text', "")
    team_df = st.session_state.get('team_df', None)

    if not jd_text.strip() or team_df is None:
        st.warning("Please upload JD & team profiles under the 'Upload & Profiles' page.")
    else:
        jd_skills = extract_skills_from_text(jd_text)
        team_skill_counts = aggregate_team_skills(team_df)
        score, matched, missing_detail = compute_skill_match(jd_skills, team_skill_counts)

        col1, col2, col3 = st.columns(3)
        col1.metric("Match Score", f"{score}%")
        col2.metric("Matched Skills", len(matched))
        col3.metric("Missing Skills", len(missing_detail))

        st.subheader("🔹 Matched Skills")
        st.write(", ".join(matched) if matched else "None found.")

        st.subheader("🔸 Missing Skills")
        st.dataframe(pd.DataFrame(missing_detail))

        viz_skills = list(jd_skills)[:8]
        team_size = max(1, sum(team_skill_counts.values()))
        viz_values = [int(100 * team_skill_counts.get(s, 0) / team_size) for s in viz_skills]

        fig = radar_chart(viz_skills, viz_values)
        st.pyplot(fig)

        st.session_state['missing_detail'] = missing_detail

elif page == "Training Suggestions":
    st.header("🎯 Suggested Trainings & Mentoring")

    missing_detail = st.session_state.get('missing_detail', [])
    if not missing_detail:
        st.info("Go to Dashboard first to compute skill gaps.")
    else:
        df_suggestions = pd.DataFrame(suggest_training_for_skills([m['skill'] for m in missing_detail]))
        st.dataframe(df_suggestions, use_container_width=True)

        if st.button("🤖 Generate AI-Based Training Recommendations"):
            skills_str = ", ".join([m['skill'] for m in missing_detail])
            ai_prompt = f"Suggest short, professional training programs or resources for improving these skills: {skills_str}. Use concise bullet points."
            ai_response = call_llm(ai_prompt)
            st.markdown("### 🧠 AI-Generated Suggestions")
            st.write(ai_response)

elif page == "About":
    st.header("ℹ️ About This App")
    st.markdown("""
        **AI Skills Radar** helps HR teams identify skill gaps, visualize strengths,
        and suggest targeted development actions.  
        - Built using **Streamlit**
        - Powered by **Gemini API**  
        - Created as a demonstration of intelligent HR analytics (v0.2)
    """)
