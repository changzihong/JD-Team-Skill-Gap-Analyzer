import os
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import requests
from PyPDF2 import PdfReader

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Skills Radar", layout="wide")

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    body {
        background-color: #f9f9ff;
        font-family: 'Poppins', sans-serif;
    }
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 100;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 0.8rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 0 0 10px 10px;
    }
    .navbar a {
        color: white;
        text-decoration: none;
        margin: 0 1rem;
        font-weight: 500;
    }
    .navbar a:hover {
        text-decoration: underline;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f1f1f1;
        text-align: center;
        padding: 0.8rem;
        color: #555;
        border-top: 1px solid #ddd;
        font-size: 14px;
    }
    .main {
        margin-top: 80px;
        margin-bottom: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GEMINI SETUP
# -----------------------------------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API key not configured. Please set it in Streamlit secrets.")
    st.stop()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
COMMON_SKILLS = [
    'python','sql','excel','data analysis','communication','project management',
    'leadership','product management','machine learning','nlp','aws','gcp','react',
    'java','c#','sales','negotiation','recruiting','interviewing','coaching','training'
]

def extract_skills_from_text(text, skill_list=COMMON_SKILLS):
    text_lower = text.lower()
    found = {s for s in skill_list if re.search(r'\\b' + re.escape(s) + r'\\b', text_lower)}
    if not found:
        tokens = re.findall(r"[a-zA-Z]{4,}", text_lower)
        found = set(tokens[:8])
    return sorted(found)

def aggregate_team_skills(df):
    counts = {}
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
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    stats = values + values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), skills)
    ax.set_ylim(0,100)
    ax.set_title(title)
    return fig

def call_gemini_api(prompt, max_output_tokens=512, temperature=0.3):
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
        st.error(f"❌ Gemini API error: {e}")
        return ""

# -----------------------------------------------------------------------------
# NAVBAR (STATIC)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="navbar">
  <div style="font-weight:600;font-size:18px;">🤖 AI Skills Radar</div>
  <div>
    <a href="#" id="home-link">Home</a>
    <a href="#" id="upload-link">Upload</a>
    <a href="#" id="dashboard-link">Dashboard</a>
    <a href="#" id="login-link">Login</a>
    <a href="#" id="signup-link">Signup</a>
    <a href="#" id="about-link">About</a>
  </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN CONTENT (SINGLE PAGE)
# -----------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"

st.markdown('<div class="main">', unsafe_allow_html=True)

nav_option = st.radio("", ["Home", "Upload", "Dashboard", "Login", "Signup", "About"], horizontal=True, label_visibility="collapsed")

if nav_option == "Home":
    st.title("Welcome to AI Skills Radar")
    st.write("Your AI-powered solution to identify skill gaps and team readiness.")

elif nav_option == "Login":
    st.header("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "admin" and password == "1234":
                st.success("Welcome back, Admin!")
            else:
                st.error("Invalid credentials")

elif nav_option == "Signup":
    st.header("🆕 Signup")
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account")
        if submitted:
            if password == confirm:
                st.success(f"Account created for {username}! You can now login.")
            else:
                st.error("Passwords do not match.")

elif nav_option == "Upload":
    st.header("📁 Upload Job Description & Team Profiles")
    jd_file = st.file_uploader('Upload JD (TXT/PDF)', type=['txt','pdf'])
    jd_text = st.text_area('Or paste JD text here', height=160)
    team_file = st.file_uploader('Upload Team Profiles (CSV/Excel)', type=['csv','xlsx','xls'])

    if st.button("Analyze Skills"):
        if jd_file:
            if jd_file.type == 'text/plain':
                jd_text = jd_file.getvalue().decode('utf-8')
            elif jd_file.type == 'application/pdf':
                reader = PdfReader(jd_file)
                jd_text = "\n".join([page.extract_text() for page in reader.pages])
        if team_file:
            team_df = pd.read_csv(team_file) if team_file.name.endswith('.csv') else pd.read_excel(team_file)
            st.session_state['team_df'] = team_df
            st.session_state['jd_text'] = jd_text
            st.success("✅ Uploaded successfully! Go to Dashboard to view results.")

elif nav_option == "Dashboard":
    st.header("📊 AI Skill Gap Dashboard")
    jd_text = st.session_state.get('jd_text')
    team_df = st.session_state.get('team_df')

    if not jd_text or team_df is None:
        st.warning("Please upload data first from the Upload section.")
    else:
        jd_skills = extract_skills_from_text(jd_text)
        team_skill_counts = aggregate_team_skills(team_df)
        score, matched, missing_detail = compute_skill_match(jd_skills, team_skill_counts)
        st.metric("Skill Match Score", f"{score}%")
        st.write("Matched Skills:", matched)
        st.write("Missing Skills:", [m['skill'] for m in missing_detail])

        viz_skills = list(jd_skills)[:8]
        team_size = max(1, sum(team_skill_counts.values()))
        viz_values = [int(100 * team_skill_counts.get(s,0) / team_size) for s in viz_skills]
        fig = radar_chart(viz_skills, viz_values)
        st.pyplot(fig)

        prompt = f"Analyze missing skills: {missing_detail}. Suggest improvement plan."
        insights = call_gemini_api(prompt)
        if insights:
            st.subheader("Gemini AI Insights")
            st.write(insights)

elif nav_option == "About":
    st.header("ℹ️ About")
    st.write("""
        **AI Skills Radar** helps HR, L&D, and management teams assess capability gaps, 
        visualize skill readiness, and get AI-powered insights for upskilling plans.
    """)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FOOTER (ALWAYS FIXED)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    © 2025 AI Skills Radar | Built with ❤️ for HR Tech | support@aiskillsradar.com
</div>
""", unsafe_allow_html=True)
