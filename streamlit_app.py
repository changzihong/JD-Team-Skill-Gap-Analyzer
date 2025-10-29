import os
import re
import time
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

# -----------------------------------
# Streamlit Config
# -----------------------------------
st.set_page_config(page_title="AI Skills Radar", layout="wide")

# -----------------------------------
# Initialize Session State
# -----------------------------------
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Home"

# -----------------------------------
# Load Gemini API Key
# -----------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)

# -----------------------------------
# Custom CSS (Modern Black & White)
# -----------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body {
  font-family: 'Inter', sans-serif;
  background: #f8f8f8;
  color: #111;
  margin: 0;
}

/* --- NAVBAR --- */
.top-navbar {
  position: fixed;
  top: 0; left: 0;
  width: 100%;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background-color: #000;
  color: white;
  padding: 14px 40px;
  box-shadow: 0 3px 8px rgba(0,0,0,0.4);
}

.nav-left {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 18px;
}

.nav-links {
  display: flex;
  gap: 16px;
  list-style: none;
  margin: 0;
  padding: 0;
}

.nav-link {
  padding: 8px 18px;
  border-radius: 8px;
  background: rgba(255,255,255,0.1);
  color: white;
  text-decoration: none;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.nav-link:hover {
  background: rgba(255,255,255,0.25);
  transform: translateY(-1px);
}

.nav-link.active {
  background: rgba(255,255,255,0.35);
}

/* --- MAIN BODY --- */
.main-container {
  padding: 100px 60px 40px 60px;
  max-width: 1400px;
  margin: 0 auto;
}

/* --- CARDS --- */
.section-card {
  background: white;
  border-radius: 16px;
  padding: 40px;
  margin-bottom: 25px;
  box-shadow: 0 4px 18px rgba(0,0,0,0.05);
}

/* --- HERO SECTION --- */
.hero-section {
  background: linear-gradient(135deg,#000,#333);
  color: white;
  padding: 60px 40px;
  border-radius: 16px;
  text-align: center;
  margin-bottom: 30px;
}

.hero-title {
  font-size: 46px;
  font-weight: 700;
  margin-bottom: 16px;
}

.hero-subtitle {
  font-size: 18px;
  color: #ccc;
}

/* --- FOOTER --- */
.app-footer {
  background: #000;
  color: #ddd;
  padding: 40px 60px;
  text-align: center;
  margin-top: 60px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------
# JavaScript Navbar Logic
# -----------------------------------
st.markdown(
    """
<script>
function navigate(page) {
  const streamlitWindow = window.parent;
  streamlitWindow.postMessage({type: 'streamlit:setSessionState', key: 'current_page', value: page}, '*');
  window.location.reload();
}
</script>
""",
    unsafe_allow_html=True,
)

# -----------------------------------
# Navigation Function
# -----------------------------------
def set_page(page_name: str):
    st.session_state["current_page"] = page_name
    st.rerun()

# -----------------------------------
# Render Top Navbar
# -----------------------------------
current_page = st.session_state["current_page"]
navbar_html = f"""
<div class="top-navbar">
  <div class="nav-left">📊 <span>AI Skills Radar</span></div>
  <ul class="nav-links">
    <li><a class="nav-link {'active' if current_page=='Home' else ''}" href="#" onclick="navigate('Home')">🏠 Home</a></li>
    <li><a class="nav-link {'active' if current_page=='Account' else ''}" href="#" onclick="navigate('Account')">🔐 Account</a></li>
    <li><a class="nav-link {'active' if current_page=='Upload & Analyze' else ''}" href="#" onclick="navigate('Upload & Analyze')">📁 Upload & Analyze</a></li>
  </ul>
</div>
"""
st.markdown(navbar_html, unsafe_allow_html=True)

# -----------------------------------
# Skills Functions
# -----------------------------------
COMMON_SKILLS = [
    "python",
    "sql",
    "excel",
    "data analysis",
    "communication",
    "project management",
    "leadership",
    "machine learning",
    "aws",
    "gcp",
    "nlp",
    "training",
    "coaching",
]

def extract_skills_from_text(text):
    if not text:
        return []
    text_lower = text.lower()
    found = [s for s in COMMON_SKILLS if s in text_lower]
    return sorted(set(found))

def aggregate_team_skills(df):
    skills_count = {}
    if df is None:
        return skills_count
    colnames = [c.lower() for c in df.columns]
    skill_col = next((c for c in df.columns if "skill" in c.lower()), None)
    if skill_col:
        for s in df[skill_col].dropna():
            for sk in re.split(r"[,;/]", str(s).lower()):
                sk = sk.strip()
                if sk:
                    skills_count[sk] = skills_count.get(sk, 0) + 1
    return skills_count

def compute_skill_match(jd_skills, team_skill_counts):
    jd_set = set(jd_skills)
    team_set = set(team_skill_counts.keys())
    matched = jd_set & team_set
    missing = jd_set - team_set
    score = int(100 * len(matched) / max(1, len(jd_set)))
    return score, matched, missing

# -----------------------------------
# MAIN PAGE LOGIC
# -----------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ---------- HOME ----------
if current_page == "Home":
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-title">Welcome to AI Skills Radar</div>
            <div class="hero-subtitle">Empower HR with data-driven skill insights.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
        <h3>🚀 Quick Start</h3>
        <ul>
        <li>Go to <b>Upload & Analyze</b> to upload your job description and team data.</li>
        <li>Let AI detect <b>skills, gaps, and strengths</b>.</li>
        <li>View visual charts and AI recommendations instantly.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- ACCOUNT ----------
elif current_page == "Account":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🔐 Account")
    if "user" in st.session_state:
        st.success(f"Logged in as **{st.session_state['user']}**")
        if st.button("Logout"):
            del st.session_state["user"]
            st.success("Logged out successfully!")
            st.rerun()
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state["user"] = username
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- UPLOAD & ANALYZE ----------
elif current_page == "Upload & Analyze":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("📁 Upload & Analyze Skills")
    jd_file = st.file_uploader("Upload Job Description (TXT/PDF)", type=["txt", "pdf"])
    team_file = st.file_uploader("Upload Team Data (CSV/Excel)", type=["csv", "xlsx"])

    jd_text = ""
    if jd_file:
        if jd_file.type == "text/plain":
            jd_text = jd_file.getvalue().decode("utf-8")
        elif jd_file.type == "application/pdf":
            reader = PdfReader(jd_file)
            jd_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        st.text_area("📄 JD Preview", jd_text, height=200)

    team_df = None
    if team_file:
        try:
            team_df = pd.read_csv(team_file) if team_file.name.endswith(".csv") else pd.read_excel(team_file)
            st.dataframe(team_df.head())
        except Exception as e:
            st.error(f"Error loading file: {e}")

    if st.button("🚀 Run Analysis"):
        if not jd_text:
            st.warning("Please upload or paste a Job Description.")
        elif team_df is None:
            st.warning("Please upload a team profile file.")
        else:
            jd_skills = extract_skills_from_text(jd_text)
            team_skill_counts = aggregate_team_skills(team_df)
            score, matched, missing = compute_skill_match(jd_skills, team_skill_counts)

            st.metric("Skill Match Score", f"{score}%")
            st.write("✅ Matched Skills:", ", ".join(matched))
            st.write("⚠️ Missing Skills:", ", ".join(missing) or "None")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    """
<div class="app-footer">
<p>© 2025 AI Skills Radar | Built for HR by AI 🤖</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
