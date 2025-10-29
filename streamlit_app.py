import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import requests
import os
from io import StringIO
from PyPDF2 import PdfReader

# --- SETUP ---
st.set_page_config(page_title="AI Skill Gap Analyzer", layout="wide")

# --- GEMINI API CONFIG ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- CSS STYLING ---
st.markdown("""
    <style>
    body {
        background-color: #0d0d0d;
        color: white;
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
    }
    .navbar {
        position: fixed;
        top: 0;
        width: 100%;
        background-color: black;
        padding: 12px 0;
        text-align: center;
        z-index: 1000;
        border-bottom: 1px solid #333;
    }
    .navbar a {
        color: white;
        text-decoration: none;
        margin: 0 20px;
        font-weight: 500;
        transition: color 0.3s ease, transform 0.3s ease;
    }
    .navbar a:hover {
        color: #aaa;
        transform: scale(1.1);
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: black;
        color: #bbb;
        text-align: center;
        padding: 8px;
        font-size: 0.85rem;
        border-top: 1px solid #333;
    }
    .main-content {
        padding-top: 80px;
        padding-bottom: 80px;
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .stButton>button {
        background-color: white;
        color: black;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #333;
        color: white;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- JAVASCRIPT (scroll animation) ---
st.markdown("""
    <script>
    document.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 30) {
            navbar.style.backgroundColor = '#1a1a1a';
        } else {
            navbar.style.backgroundColor = 'black';
        }
    });
    </script>
""", unsafe_allow_html=True)

# --- NAVBAR ---
st.markdown("""
<div class="navbar">
    <a href="#home">Home</a>
    <a href="#analyzer">Analyzer</a>
    <a href="#login">Login / Signup</a>
    <a href="#about">About</a>
</div>
""", unsafe_allow_html=True)

# --- MAIN CONTENT START ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# ---------------------- HOME ----------------------
st.markdown('<a name="home"></a>', unsafe_allow_html=True)
st.title("🧭 AI Skill Gap Analyzer")
st.write("Identify, analyze, and bridge skill gaps between your team and job descriptions — powered by Gemini AI.")

# ---------------------- LOGIN / SIGNUP ----------------------
st.markdown('<a name="login"></a>', unsafe_allow_html=True)
st.subheader("🔐 Login / Signup")

with st.form("login_form"):
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username / Email")
        password = st.text_input("Password", type="password")
    with col2:
        mode = st.radio("Select Mode", ["Login", "Signup"])
    submitted = st.form_submit_button("Submit")

if submitted:
    if username and password:
        st.success(f"{mode} successful! Welcome, {username}.")
    else:
        st.error("Please enter both username and password.")

# ---------------------- ANALYZER ----------------------
st.markdown('<a name="analyzer"></a>', unsafe_allow_html=True)
st.subheader("📊 JD & Team Skill Analyzer")

jd_file = st.file_uploader("Upload Job Description (txt or pdf)", type=["txt", "pdf"])
jd_text = st.text_area("Or paste JD text here", height=150)
team_file = st.file_uploader("Upload Team Profiles (CSV or Excel)", type=["csv", "xlsx"])

def extract_skills(text):
    skills = ['python', 'sql', 'excel', 'data analysis', 'communication', 'project management',
              'leadership', 'product management', 'machine learning', 'nlp', 'aws', 'gcp', 'react',
              'java', 'sales', 'negotiation', 'recruiting', 'training']
    text_lower = text.lower()
    found = [s for s in skills if s in text_lower]
    return list(set(found))

def analyze_gap(jd_text, team_df):
    jd_skills = extract_skills(jd_text)
    team_skills = []
    for s in team_df.columns:
        for val in team_df[s].astype(str):
            team_skills.extend(extract_skills(val))
    jd_set = set(jd_skills)
    team_set = set(team_skills)
    matched = jd_set & team_set
    missing = jd_set - team_set
    return matched, missing

def call_gemini_summary(matched, missing):
    """Call Gemini API to generate summarized insight."""
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured."
    
    prompt = f"""
    You are an AI HR assistant. 
    Based on the following skill analysis, summarize insights in **point form**, around **200 words**.
    
    - Matched skills: {', '.join(matched)}
    - Missing skills: {', '.join(missing)}
    
    Focus on:
    • Overall readiness of the team  
    • Key missing skill areas  
    • Recommendations for training or hiring  
    • Predictive note on 6-month skill needs  
    Keep it professional and concise.
    """

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    resp = requests.post(
        GEMINI_API_URL,
        headers=headers,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=30,
    )
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return f"Error parsing response: {data}"

if st.button("Run Skill Gap Analysis"):
    if jd_file or jd_text:
        if jd_file and jd_file.type == "application/pdf":
            reader = PdfReader(jd_file)
            jd_text = "\n".join(page.extract_text() for page in reader.pages)
        if team_file:
            team_df = pd.read_csv(team_file) if team_file.name.endswith(".csv") else pd.read_excel(team_file)
            matched, missing = analyze_gap(jd_text, team_df)
            st.success("✅ Analysis Complete")
            st.write("**Matched Skills:**", list(matched))
            st.write("**Missing Skills:**", list(missing))

            # Radar chart
            all_skills = list(matched) + list(missing)
            values = [100 if s in matched else 40 for s in all_skills]
            N = len(all_skills)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), all_skills)
            ax.set_title("Team Readiness Radar")
            st.pyplot(fig)

            # Call Gemini summary
            st.markdown("### 📊 AI-Generated Skill Gap Insights")
            with st.spinner("Generating AI summary..."):
                summary = call_gemini_summary(matched, missing)
                st.markdown(f"💡 **Summary:**\n\n{summary}")
        else:
            st.error("Please upload team profiles file.")
    else:
        st.error("Please provide a JD text or upload file.")

# ---------------------- ABOUT ----------------------
st.markdown('<a name="about"></a>', unsafe_allow_html=True)
st.subheader("💡 About This Application")
st.write("This AI Skill Gap Analyzer helps HR and L&D managers quickly identify skill mismatches using generative AI insights.")

# --- END CONTENT ---
st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    © 2025 AI Skill Gap Analyzer | Powered by Gemini API & Streamlit
</div>
""", unsafe_allow_html=True)
