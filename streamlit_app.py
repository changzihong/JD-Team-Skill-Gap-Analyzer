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

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Skills Radar", layout="wide")

# -----------------------------------------------------------------------------
# LOAD API KEY SAFELY
# -----------------------------------------------------------------------------
# Read API key from Streamlit secrets or environment variable
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

# Use correct model and endpoint
MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API key not configured. Please set GEMINI_API_KEY in Streamlit secrets or env variables.")
    st.stop()

# -----------------------------------------------------------------------------
# NAVIGATION BAR
# -----------------------------------------------------------------------------
page = st_navbar(["Home", "Upload & Profiles", "Dashboard", "Training Suggestions", "About"], selected="Home")

# -----------------------------------------------------------------------------
# COMMON SKILLS DATABASE
# -----------------------------------------------------------------------------
COMMON_SKILLS = [
    'python','sql','excel','data analysis','communication','project management',
    'leadership','product management','machine learning','nlp','aws','gcp','react',
    'java','c#','sales','negotiation','recruiting','interviewing','coaching','training'
]

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
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
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,100)
    ax.set_title(title)
    return fig


def call_gemini_api(prompt, max_output_tokens=512, temperature=0.2):
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens}
    }
    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        st.error(f"❌ Error calling Gemini API: {e}")
        st.write("Raw:", resp.text)
        return ""


def suggest_training_for_skills(skills):
    suggestions = []
    for s in skills:
        suggestions.append({
            'Skill': s,
            'Recommended Course': f'Advanced {s.title()} Masterclass',
            'Internal Mentor': f'Senior {s.title()} Specialist'
        })
    return suggestions

# -----------------------------------------------------------------------------
# PAGE LOGIC
# -----------------------------------------------------------------------------
if page == "Home":
    st.title("AI Skills Radar")
    st.write("Analyze your team's skills vs. job requirements with intelligent insights powered by Gemini.")

elif page == "Upload & Profiles":
    st.header("Upload Job Description & Team Profiles")

    jd_file = st.file_uploader('Upload Job Description (TXT or PDF)', type=['txt','pdf'])
    jd_text = st.text_area('Or paste JD text here', height=160)

    team_file = st.file_uploader('Upload Team Profiles (CSV/Excel)', type=['csv','xlsx','xls'])

    if st.button("Proceed to Dashboard"):
        if jd_file is not None:
            if jd_file.type == 'text/plain':
                jd_text = jd_file.getvalue().decode('utf-8')
            elif jd_file.type == 'application/pdf':
                reader = PdfReader(jd_file)
                jd_text = "\n".join([page.extract_text() for page in reader.pages])
            else:
                st.warning("Unsupported file type.")

        if team_file is not None:
            try:
                if team_file.name.endswith('.csv'):
                    team_df = pd.read_csv(team_file)
                else:
                    team_df = pd.read_excel(team_file)
                st.session_state['team_df'] = team_df
                st.session_state['jd_text'] = jd_text
                st.success("Uploaded successfully! Go to Dashboard.")
            except Exception as e:
                st.error("Failed to load team file: " + str(e))

elif page == "Dashboard":
    st.header("📊 AI-Generated Skill Gap Insights")

    jd_text = st.session_state.get('jd_text')
    team_df = st.session_state.get('team_df')

    if not jd_text or team_df is None:
        st.warning("Please upload Job Description and team profiles first.")
    else:
        jd_skills = extract_skills_from_text(jd_text)
        team_skill_counts = aggregate_team_skills(team_df)
        score, matched, missing_detail = compute_skill_match(jd_skills, team_skill_counts)

        st.metric(label='Match Score', value=f'{score}%')
        st.write("**Matched Skills:**", ", ".join(matched))
        st.write("**Missing Skills:**", [m['skill'] for m in missing_detail])

        viz_skills = list(jd_skills)[:8]
        team_size = max(1, sum(team_skill_counts.values()))
        viz_values = [int(100 * team_skill_counts.get(s,0) / team_size) for s in viz_skills]
        fig = radar_chart(viz_skills, viz_values)
        st.pyplot(fig)

        prompt = f"Analyze these missing skills: {missing_detail} and suggest overall development strategy."
        ai_feedback = call_gemini_api(prompt)
        if ai_feedback:
            st.subheader("Gemini Insights")
            st.write(ai_feedback)

        st.session_state['missing_detail'] = missing_detail

elif page == "Training Suggestions":
    st.header("🎯 Suggested Trainings & Mentorship")
    missing_detail = st.session_state.get('missing_detail', [])

    if not missing_detail:
        st.info("Please go to Dashboard first to identify missing skills.")
    else:
        df_suggestions = pd.DataFrame(suggest_training_for_skills([m['skill'] for m in missing_detail]))
        st.dataframe(df_suggestions, use_container_width=True)

elif page == "About":
    st.header("About")
    st.write("AI Skills Radar helps HR and team leads identify skill gaps and plan targeted upskilling paths.")
